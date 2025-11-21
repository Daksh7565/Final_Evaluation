import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from sqlalchemy.orm import sessionmaker
from langchain.tools import tool
from langgraph.graph import StateGraph, END

# Optional LLM (used only for improved natural-language explanation)
try:
    from langchain_groq import ChatGroq
    _HAS_GROQ = True
except Exception:
    _HAS_GROQ = False

# Project DB imports
from DB import engine, Route, SessionLocal, get_per_minute_counts, get_class_distribution, get_rolling_5min

# Load environment variables
load_dotenv()

SessionLocal = sessionmaker(bind=engine)

# ------------
# Helper utilities
# ------------

def _mean_numeric_values(d: dict) -> float:
    vals = [v for k, v in d.items() if isinstance(v, (int, float))]
    return float(np.mean(vals)) if vals else 0.0


def _compute_confidence(best_score: float, second_score: float) -> float:
    """Map gap between best and second-best into [0.85, 0.98]. Larger gap → higher confidence."""
    if np.isinf(best_score):
        return 0.0
    if np.isinf(second_score):
        # no competitor, high confidence
        return 0.98
    gap = second_score - best_score
    # normalize gap relative to second_score (avoid div by zero)
    denom = max(abs(second_score), 1.0)
    normalized = max(0.0, min(gap / denom, 1.0))
    base = 0.85
    conf = base + normalized * (0.98 - base)
    return round(float(conf), 2)


# ------------
# Tools
# ------------
@tool
def fetch_route_statistics(route_id: str, window: int = 10) -> dict:
    """
    Fetches per-minute counts, recent window counts, 5-minute rolling averages, and class distribution.
    Returns a dict with keys: route_id, total_counts, recent_window_counts, rolling_average_5min (dict), class_distribution (list)
    If no data available, returns {'route_id': route_id, 'message': 'No vehicle data available.'}
    """
    session = SessionLocal()
    try:
        route = session.query(Route).filter_by(name=route_id).first()
        if not route:
            return {"error": f"Route '{route_id}' not found."}

        df_min = get_per_minute_counts(route.id)
        if df_min is None or df_min.empty:
            return {"route_id": route_id, "message": "No vehicle data available."}

        # ensure minute column is datetime and sorted
        df = df_min.copy()
        df["minute"] = pd.to_datetime(df["minute"])
        df = df.sort_values("minute")

        total_counts = int(df["count"].sum())

        # recent window
        cutoff = df["minute"].max() - pd.Timedelta(minutes=window)
        recent = df[df["minute"] >= cutoff]
        recent_window_counts = int(recent["count"].sum())

        # rolling 5-min
        df_roll = get_rolling_5min(df)
        rolling_record = df_roll.tail(1).to_dict("records")[0] if (df_roll is not None and not df_roll.empty) else {}

        class_dist_df = get_class_distribution(route.id)
        class_dist = class_dist_df.to_dict("records") if (class_dist_df is not None and not class_dist_df.empty) else []

        return {
            "route_id": route_id,
            "total_counts": total_counts,
            "recent_window_counts": recent_window_counts,
            "rolling_average_5min": rolling_record,
            "class_distribution": class_dist,
        }

    except Exception as e:
        return {"error": f"Exception fetching stats for {route_id}: {str(e)}"}
    finally:
        session.close()


@tool
def compute_congestion_score(stats: dict) -> dict:
    """
    Computes congestion score from stats dict.
    Score formula (tunable): score = A*recent + B*rolling_avg + C*truck_factor
    Returns dict: {route_id, score, components}
    If stats contains 'error' or 'message', returns route with score=inf to disqualify.
    """
    if stats is None:
        return {"error": "Empty stats provided."}

    if "error" in stats:
        return stats
    if "message" in stats:
        # No data available → disqualify
        return {"route_id": stats.get("route_id"), "score": float("inf"), "components": {"reason": "no_data"}}

    route_id = stats.get("route_id")

    # Tunable weights
    A = 0.6  # recent weight
    B = 0.3  # rolling avg weight
    C = 0.1  # truck weight

    recent = float(stats.get("recent_window_counts", 0))

    roll = stats.get("rolling_average_5min", {}) or {}
    # rolling_record might include 'minute' key and per-class columns; compute mean of numeric values
    rolling_avg = _mean_numeric_values(roll)

    class_dist = stats.get("class_distribution", []) or []
    truck_factor = 0
    for row in class_dist:
        # support either 'class_name' or 'class' keys (robust)
        name = row.get("class_name") or row.get("class")
        cnt = row.get("count") or row.get("cnt") or 0
        if name and str(name).lower() == "truck":
            try:
                truck_factor += int(cnt)
            except Exception:
                truck_factor += 0

    score = A * recent + B * rolling_avg + C * truck_factor

    return {
        "route_id": route_id,
        "score": float(score),
        "components": {
            "recent": recent,
            "rolling_avg": float(rolling_avg),
            "truck_factor": int(truck_factor),
        },
    }


@tool
def recommend_route(context: dict, scores: list) -> dict:
    """
    Picks best route (lowest score), computes a reason and a confidence score.
    context: reserved for future use (e.g., vehicle type)
    scores: list of dicts produced by compute_congestion_score (or error/message dicts)
    Returns: {route_id, score, reason, confidence, all_scores}
    """
    if not isinstance(scores, list):
        return {"error": "Invalid scores input; expected list."}

    # filter out error entries and treat 'message' (no data) as disqualified
    valid = [s for s in scores if isinstance(s, dict) and ("score" in s) and (not np.isinf(s.get("score")))]

    if not valid:
        return {"error": "No valid scores available to recommend."}

    # find best and second-best for confidence calculation
    sorted_scores = sorted(valid, key=lambda x: x["score"])  # ascending
    best = sorted_scores[0]
    second = sorted_scores[1] if len(sorted_scores) > 1 else {"score": float("inf")}

    # compute confidence
    confidence = _compute_confidence(best_score=best["score"], second_score=second.get("score", float("inf")))

    # build deterministic reason summary
    reason_parts = []
    comp = best.get("components", {})
    reason_parts.append(f"lowest congestion score ({best['score']:.2f})")
    reason_parts.append(f"recent={comp.get('recent', 0)}")
    reason_parts.append(f"rolling_avg={comp.get('rolling_avg', 0):.2f}")
    reason_parts.append(f"truck_factor={comp.get('truck_factor', 0)}")
    deterministic_reason = f"Route {best['route_id']} selected because it has the {', '.join(reason_parts)}."

    # Optionally generate LLM explanation if available and key present
    llm_reason = None
    if _HAS_GROQ and os.getenv("key_sal_groq"):
        try:
            llm = ChatGroq(model="llama3-8b-8192", groq_api_key=os.getenv("key_sal_groq"))
            # Craft a concise prompt that requests only the required format body
            prompt = (
                "You are an expert traffic analyst. Provide a one-paragraph explanation (1-2 sentences) "
                "that justifies why the following route is recommended. Use the numeric evidence.\n\n"
                f"Best route: {best['route_id']} (score={best['score']:.2f})\n"
                f"Components: {best.get('components', {})}\n"
                f"Other routes: {[{'route_id': s.get('route_id'), 'score': s.get('score')} for s in sorted_scores[1:]]}\n\n"
                "Respond only with a short justification (no extra framing)."
            )
            resp = llm.invoke(prompt)
            # Accept resp.content or str(resp)
            llm_reason = resp.content if hasattr(resp, "content") else str(resp)
            # sanitize: ensure it's not empty
            if llm_reason and len(llm_reason.strip()) > 10:
                final_reason = llm_reason.strip()
            else:
                final_reason = deterministic_reason
        except Exception:
            final_reason = deterministic_reason
    else:
        final_reason = deterministic_reason

    return {
        "route_id": best["route_id"],
        "score": float(best["score"]),
        "reason": final_reason,
        "confidence": float(confidence),
        "all_scores": scores,
    }


# ---------------------
# Agent workflow (StateGraph)
# ---------------------

def initial_state():
    return {"stats": [], "scores": [], "recommendation": None}


# Nodes
def gather_stats(state):
    session = SessionLocal()
    routes = session.query(Route).all()
    session.close()

    stats = []
    for r in routes:
        res = fetch_route_statistics.invoke({"route_id": r.name})
        stats.append(res)
    state["stats"] = stats
    return state


def calculate_scores(state):
    if "stats" not in state or not state["stats"]:
        state["scores"] = []
        return state

    scores = []
    for s in state["stats"]:
        try:
            sc = compute_congestion_score.invoke({"stats": s})
        except Exception as e:
            sc = {"route_id": s.get("route_id"), "score": float("inf"), "components": {"error": str(e)}}
        scores.append(sc)
    state["scores"] = scores
    return state


def choose_best_route(state):
    scores = state.get("scores", [])
    try:
        rec = recommend_route.invoke({"context": {}, "scores": scores})
    except Exception as e:
        rec = {"error": str(e)}
    state["recommendation"] = rec
    return state


graph = StateGraph(dict)
graph.add_node("gather_stats", gather_stats)
graph.add_node("calculate_scores", calculate_scores)
graph.add_node("choose_best", choose_best_route)

graph.set_entry_point("gather_stats")
graph.add_edge("gather_stats", "calculate_scores")
graph.add_edge("calculate_scores", "choose_best")
graph.add_edge("choose_best", END)

route_agent = graph.compile()


# Entry function used by Streamlit frontend
def get_route_recommendation() -> str:
    """Runs the agent and returns the recommendation in the strict required format."""
    init = initial_state()
    final_state = route_agent.invoke(init)
    rec = final_state.get("recommendation")

    # Handle errors
    if rec is None:
        return "Recommended Route: N/A\n\nReason:\nNo recommendation produced.\n\nConfidence: 0.00"

    if isinstance(rec, dict) and "error" in rec:
        return f"Recommended Route: N/A\n\nReason:\nError generating recommendation: {rec['error']}\n\nConfidence: 0.00"

    if isinstance(rec, dict) and "route_id" in rec:
        return (
            f"Recommended Route: {rec['route_id']}\n\n"
            f"Reason:\n{rec.get('reason', 'N/A')}\n\n"
            f"Confidence: {rec.get('confidence', 'N/A')}"
        )

    # Fallback
    return str(rec)


if __name__ == "__main__":
    print(get_route_recommendation())
