import os
import argparse
from dotenv import load_dotenv
from sqlalchemy.orm import sessionmaker
import numpy as np
import pandas as pd
from langchain_groq import ChatGroq
from langchain.tools import tool
from langgraph.graph import StateGraph, END

# DB Models
from DB import engine ,Route, SessionLocal,get_per_minute_counts,get_class_distribution,get_hourly_counts,get_rolling_5min


# Tools implemented earlier in your project
@tool
def fetch_route_statistics(route_id: str, window: int = 10) -> dict:
    """
    Fetches per-minute counts, 5-minute rolling avg, and
    recent (window-minute) congestion stats for a route.
    """
    print(f"[Tool] fetch_route_statistics(route_id={route_id}, window={window})")

    session = SessionLocal()

    try:
        route = session.query(Route).filter_by(name=route_id).first()
        if not route:
            return {"error": f"Route '{route_id}' not found."}

        # Per-minute
        df_min = get_per_minute_counts(route.id)

        if df_min.empty:
            return {
                "route_id": route_id,
                "message": "No vehicle data available."
            }

        # Rolling 5-min
        df_roll = get_rolling_5min(df_min)

        # last window minutes
        df_min["minute"] = pd.to_datetime(df_min["minute"])
        cutoff = df_min["minute"].max() - pd.Timedelta(minutes=window)
        recent = df_min[df_min["minute"] >= cutoff]

        stats = {
            "route_id": route_id,
            "total_counts": int(df_min["count"].sum()),
            "recent_window_counts": int(recent["count"].sum()),
            "rolling_average_5min": df_roll.tail(1).to_dict("records")[0] if not df_roll.empty else {},
            "class_distribution": get_class_distribution(route.id).to_dict("records")
        }
        return stats

    finally:
        session.close()


# ===================================================
# 2. TOOL: Compute Congestion Score
# ===================================================
@tool
def compute_congestion_score(stats: dict) -> dict:
    """
    Computes congestion score using:
    score = α*(recent_window_counts) + β*(rolling_avg) + γ*(class_weights)
    Lower score means better route.
    """
    print(f"[Tool] compute_congestion_score for route={stats.get('route_id')}")

    if "error" in stats:
        return stats

    # Weights (tunable)
    A = 0.6   # recent congestion weight
    B = 0.3   # rolling avg weight
    C = 0.1   # class distribution weight

    recent = stats.get("recent_window_counts", 0)

    roll = stats.get("rolling_average_5min", {})
    rolling_avg = float(np.mean([v for k, v in roll.items() if k not in ["minute"]])) if roll else 0

    class_dist = stats.get("class_distribution", [])
    truck_factor = sum(row["count"] for row in class_dist if row["class_name"] == "truck")

    score = A*recent + B*rolling_avg + C*truck_factor

    return {
        "route_id": stats["route_id"],
        "score": float(score),
        "components": {
            "recent": recent,
            "rolling_avg": rolling_avg,
            "truck_factor": truck_factor
        }
    }


# ===================================================
# 3. TOOL: Recommend Route
# ===================================================
@tool
def recommend_route(context: dict, scores: list) -> dict:
    """
    Picks the best route (lowest score) and generates natural language reason.
    """
    print("[Tool] recommend_route() executing")

    filtered_scores = [s for s in scores if "error" not in s]

    if not filtered_scores:
        return {"error": "No valid route statistics available."}

    best = min(filtered_scores, key=lambda x: x["score"])

    # This can be expanded with an LLM call to generate a natural language reason
    reason = f"Route {best['route_id']} has the lowest congestion score of {best['score']:.2f}. "
    reason += f"This is based on recent traffic of {best['components']['recent']} vehicles, a rolling average of {best['components']['rolling_avg']:.2f}, and a truck factor of {best['components']['truck_factor']}."

    return {
        "route_id": best["route_id"],
        "score": best["score"],
        "reason": reason,
        "confidence": 0.95, # Example confidence score
        "all_scores": filtered_scores
    }

# Load .env
load_dotenv()

# -------------------------
# DATABASE
# -------------------------
SessionLocal = sessionmaker(bind=engine)

# -------------------------
# LLM
# -------------------------
llm = ChatGroq(
    model="llama3-8b-8192",
    groq_api_key=os.getenv("key_sal_groq")
)

# -------------------------
# STATE DEFINITION
# -------------------------
def initial_state():
    return {
        "stats": [],
        "scores": [],
        "recommendation": None
    }

# -------------------------
# NODES
# -------------------------

def gather_stats(state):
    print("\n--- Step 1: Gathering Stats for All Routes ---")

    session = SessionLocal()
    routes = session.query(Route).all()
    session.close()

    stats = []
    for r in routes:
        print(f"[Tool] fetch_route_statistics(route_id={r.name})")
        result = fetch_route_statistics.invoke({"route_id": r.name})
        stats.append(result)

    # Update full state
    state["stats"] = stats
    return state


def calculate_scores(state):
    print("--- Step 2: Calculating Scores ---")

    if "stats" not in state or not state["stats"]:
        raise ValueError("No route statistics available in state['stats']")

    scores = []
    for st in state["stats"]:
        print(f"[Tool] compute_congestion_score({st})")
        # FIX: The input to invoke must be a dictionary where the keys match
        # the function's argument names. The function expects an argument named "stats".
        sc = compute_congestion_score.invoke({"stats": st})
        scores.append(sc)

    state["scores"] = scores
    return state


def choose_best_route(state):
    print("--- Step 3: Choosing Best Route ---")

    if "scores" not in state or not state["scores"]:
        raise ValueError("No scores available in state['scores']")

    print("[Tool] recommend_route()")
    # Note: This call is correct because the keys "context" and "scores"
    # match the argument names of the recommend_route function.
    recommendation = recommend_route.invoke({
        "context": {},
        "scores": state["scores"]
    })

    state["recommendation"] = recommendation
    return state


# -------------------------
# BUILD LANGGRAPH WORKFLOW
# -------------------------

graph = StateGraph(dict)

graph.add_node("gather_stats", gather_stats)
graph.add_node("calculate_scores", calculate_scores)
graph.add_node("choose_best", choose_best_route)

graph.set_entry_point("gather_stats")
graph.add_edge("gather_stats", "calculate_scores")
graph.add_edge("calculate_scores", "choose_best")
graph.add_edge("choose_best", END)

route_agent = graph.compile()

# -------------------------
# RUNNER FUNCTION
# -------------------------

def get_route_recommendation():
    print("\n--- Running Route Recommendation Agent ---\n")

    init = initial_state()

    final_state = route_agent.invoke(init)
    rec = final_state["recommendation"]

    if rec is None:
        return "No recommendation generated."

    if isinstance(rec, dict):
        if "route_id" in rec:
            return (
                f"\nRecommended Route: {rec['route_id']}\n"
                f"Reason: {rec.get('reason','N/A')}\n"
                f"Confidence: {rec.get('confidence','N/A')}\n"
            )
        else:
            return str(rec)

    return str(rec)


# -------------------------
# MAIN
# -------------------------

if __name__ == "__main__":
    print(get_route_recommendation())

# import os
# from dotenv import load_dotenv
# from sqlalchemy.orm import sessionmaker
# import numpy as np
# import pandas as pd
# from datetime import timedelta # Ensure timedelta is imported

# from langchain_groq import ChatGroq
# from langchain.tools import tool
# from langgraph.graph import StateGraph, END

# # Import your project's modules
# from DB import engine, SessionLocal
# from model import Route
# from analysis import (
#     get_per_minute_counts,
#     get_class_distribution,
#     get_rolling_5min,
#     get_trend_scores,  # <-- NEW
#     detect_anomalies   # <-- NEW
# )

# # Load environment variables
# load_dotenv()

# # --- Database Session ---
# SessionLocal = sessionmaker(bind=engine)

# # --- Large Language Model ---
# llm = ChatGroq(
#     model="llama3-8b-8192",
#     groq_api_key=os.getenv("key_sal_groq")
# )

# # ===================================================
# # 1. TOOL: Fetch Comprehensive Route Statistics
# # ===================================================
# @tool
# def fetch_route_statistics(route_id: str, window: int = 10) -> dict:
#     """
#     Fetches comprehensive statistics for a route, including per-minute counts,
#     rolling averages, recent congestion, trend scores, and anomaly detection.
#     """
#     print(f"[Tool] fetch_route_statistics(route_id={route_id}, window={window})")
#     session = SessionLocal()
#     try:
#         route = session.query(Route).filter_by(name=route_id).first()
#         if not route:
#             return {"error": f"Route '{route_id}' not found."}

#         df_min = get_per_minute_counts(route.id)
#         if df_min.empty:
#             return {"route_id": route_id, "message": "No vehicle data available."}

#         # --- Standard Calculations ---
#         df_roll = get_rolling_5min(df_min)
#         df_min["minute"] = pd.to_datetime(df_min["minute"])
#         cutoff = df_min["minute"].max() - pd.Timedelta(minutes=window)
#         recent = df_min[df_min["minute"] >= cutoff]

#         # --- NEW ADVANCED ANALYTICS ---
#         trend_scores = get_trend_scores(df_min, window_minutes=15)
#         anomalies = detect_anomalies(df_min)

#         stats = {
#             "route_id": route_id,
#             "total_counts": int(df_min["count"].sum()),
#             "recent_window_counts": int(recent["count"].sum()),
#             "rolling_average_5min": df_roll.tail(1).to_dict("records")[0] if not df_roll.empty else {},
#             "class_distribution": get_class_distribution(route.id).to_dict("records"),
#             # --- ADD NEW DATA TO OUTPUT ---
#             "trend_scores": trend_scores,
#             "has_anomaly": not anomalies.empty
#         }
#         return stats
#     finally:
#         session.close()


# # ===================================================
# # 2. TOOL: Compute Intelligent Congestion Score
# # ===================================================
# @tool
# def compute_congestion_score(stats: dict) -> dict:
#     """
#     Computes an intelligent congestion score. A lower score is better.
#     The score is heavily penalized if an anomaly is detected or if there is a
#     strong positive (increasing) traffic trend.
#     """
#     print(f"[Tool] compute_congestion_score for route={stats.get('route_id')}")
    
#     if "error" in stats or "message" in stats:
#         return {**stats, "score": float('inf'), "components": {"reason": "No data available"}}

#     # --- Anomaly Check ---
#     # If there's an anomaly (potential incident), disqualify the route.
#     if stats.get("has_anomaly", False):
#         return {
#             "route_id": stats["route_id"],
#             "score": 9999.0,
#             "components": {"reason": "Anomaly detected", "has_anomaly": True}
#         }

#     # --- Define Weights for Scoring Formula ---
#     W_RECENT = 0.5   # Weight for recent traffic volume
#     W_ROLLING = 0.3  # Weight for rolling average stability
#     W_TRUCK = 0.1    # Weight for heavy vehicles (trucks)
#     W_TREND = 0.1    # Weight for traffic trend

#     # --- Extract Components ---
#     recent_count = stats.get("recent_window_counts", 0)
#     rolling_avg_dict = stats.get("rolling_average_5min", {})
#     rolling_avg = float(np.mean([v for k, v in rolling_avg_dict.items() if k != "minute"])) if rolling_avg_dict else 0
#     class_dist = stats.get("class_distribution", [])
#     truck_factor = sum(row["count"] for row in class_dist if row["class_name"] == "truck")
    
#     # --- Trend Component Calculation ---
#     # Penalize positive (increasing) trends. We ignore negative trends as they are favorable.
#     trends = stats.get("trend_scores", {})
#     # Multiply by 10 to scale the impact of the raw slope value
#     trend_factor = sum(max(0, trend) for trend in trends.values()) * 10

#     # --- Final Score Calculation ---
#     score = (
#         (W_RECENT * recent_count) +
#         (W_ROLLING * rolling_avg) +
#         (W_TRUCK * truck_factor) +
#         (W_TREND * trend_factor)
#     )

#     return {
#         "route_id": stats["route_id"],
#         "score": float(score),
#         "components": {
#             "recent": recent_count,
#             "rolling_avg": round(rolling_avg, 2),
#             "truck_factor": truck_factor,
#             "trend_factor": round(trend_factor, 2),
#             "has_anomaly": False
#         }
#     }


# # ===================================================
# # 3. TOOL: Recommend Route (Simplified)
# # ===================================================
# @tool
# def recommend_route(scores: list) -> dict:
#     """
#     Picks the best route by selecting the one with the minimum congestion score.
#     """
#     print("[Tool] recommend_route() executing")
    
#     valid_scores = [s for s in scores if "error" not in s and "message" not in s]
#     if not valid_scores:
#         return {"error": "No valid route scores were provided."}

#     best_route = min(valid_scores, key=lambda x: x["score"])
#     return best_route


# # ===================================================
# # 4. AGENT STATE AND GRAPH WORKFLOW
# # ===================================================

# class AgentState(dict):
#     pass

# # --- Graph Nodes ---

# def gather_stats(state: AgentState):
#     print("\n--- AGENT STEP 1: GATHERING STATS FOR ALL ROUTES ---")
#     session = SessionLocal()
#     routes = session.query(Route).all()
#     session.close()

#     all_stats = []
#     for r in routes:
#         stats = fetch_route_statistics.invoke({"route_id": r.name})
#         all_stats.append(stats)

#     state["all_stats"] = all_stats
#     return state

# def calculate_scores(state: AgentState):
#     print("\n--- AGENT STEP 2: CALCULATING CONGESTION SCORES ---")
#     all_stats = state.get("all_stats", [])
#     scores = [compute_congestion_score.invoke({"stats": s}) for s in all_stats]
#     state["scores"] = scores
#     return state

# def choose_best_route(state: AgentState):
#     print("\n--- AGENT STEP 3: CHOOSING BEST ROUTE & GENERATING REASON ---")
#     scores = state.get("scores", [])
    
#     best_route_info = recommend_route.invoke({"scores": scores})
#     if "error" in best_route_info:
#         state["recommendation"] = "Error: Could not determine the best route."
#         return state

#     # --- Generate a high-quality reason using the LLM ---
#     prompt = f"""
#     You are an expert AI traffic analyst. Based on the data below, your task is to provide a clear and concise route recommendation.

#     Here is the complete data for all analyzed routes:
#     {scores}

#     The recommended route is '{best_route_info['route_id']}' with a final congestion score of {best_route_info['score']:.2f}.

#     Justify this recommendation. In your reason, you MUST mention the key factors that made it the best choice. Consider the following:
#     - Was its recent traffic ('recent') lower than others?
#     - Is its rolling average ('rolling_avg') stable and low?
#     - Does it have a favorable traffic trend ('trend_factor' is low or zero)?
#     - Mention if other routes were disqualified due to anomalies (score of 9999.0).

#     Respond ONLY in the required format below, with no extra text before or after.

#     Recommended Route: {best_route_info['route_id']}

#     Reason:
#     <Your one-paragraph, professional, and data-driven explanation here.>

#     Confidence: <Provide a confidence score between 0.85 and 0.98. The score should be higher if the best route's score is significantly lower than the next best option.>
#     """

#     response = llm.invoke(prompt)
#     state["recommendation"] = response.content
#     return state

# # --- Compile the Graph ---

# graph = StateGraph(AgentState)
# graph.add_node("gather_stats", gather_stats)
# graph.add_node("calculate_scores", calculate_scores)
# graph.add_node("choose_best", choose_best_route)

# graph.set_entry_point("gather_stats")
# graph.add_edge("gather_stats", "calculate_scores")
# graph.add_edge("calculate_scores", "choose_best")
# graph.add_edge("choose_best", END)

# route_agent = graph.compile()


# # ===================================================
# # 5. ENTRY FUNCTION FOR EXTERNAL CALLS
# # ===================================================
# def get_route_recommendation():
#     """
#     Runs the full agentic workflow to get the best route recommendation.
#     """
#     print("\n--- RUNNING ROUTE RECOMMENDATION AGENT ---")
#     try:
#         initial_state = {}
#         final_state = route_agent.invoke(initial_state)
#         return final_state.get("recommendation", "Agent did not produce a final recommendation.")
#     except Exception as e:
#         print(f"An error occurred during agent execution: {e}")
#         return "An error occurred while generating the recommendation. Please check the logs."


# if __name__ == "__main__":
#     recommendation = get_route_recommendation()
#     print("\n--- FINAL RECOMMENDATION ---")
#     print(recommendation)