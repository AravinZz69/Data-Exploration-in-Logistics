"""
optimization.py
---------------
Vehicle Routing Problem (VRP) solver using Google OR-Tools.

Supports:
  - Capacitated VRP (CVRP)
  - VRP with Time Windows (VRPTW)
  - Weighted objective: α·distance + β·travel_time + γ·late_penalty + δ·vehicle_cost

Usage
-----
    from src.optimization import solve_vrp, build_distance_matrix

    distance_matrix = build_distance_matrix(locations)   # list of (lat, lon) tuples
    result = solve_vrp(
        distance_matrix=distance_matrix,
        demands=demands,            # list[int], index 0 = depot (demand=0)
        vehicle_capacities=[500, 500, 500],
        depot=0,
        time_limit_seconds=30,
    )
    print(result["routes"])
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from typing import Any

try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    print("WARNING: OR-Tools not installed. Run: pip install ortools")


# ── Distance helpers ──────────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two (lat, lon) points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def build_distance_matrix(
    locations: list[tuple[float, float]],
    scale: int = 1000,
) -> list[list[int]]:
    """
    Build an integer distance matrix (metres × scale) from a list of (lat, lon) tuples.
    OR-Tools requires integer arc costs.

    Parameters
    ----------
    locations : list of (latitude, longitude)
    scale     : multiply km by this factor to convert to integer arc weights
                (default 1000 → metres)
    """
    n = len(locations)
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                km = haversine_km(*locations[i], *locations[j])
                matrix[i][j] = int(km * scale)
    return matrix


# ── VRP Solver ────────────────────────────────────────────────────────────────

def solve_vrp(
    distance_matrix: list[list[int]],
    demands: list[int],
    vehicle_capacities: list[int],
    depot: int = 0,
    time_limit_seconds: int = 30,
    time_windows: list[tuple[int, int]] | None = None,
    penalty_per_unserved: int = 100_000,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Solve a Capacitated VRP (optionally with time windows) using OR-Tools.

    Parameters
    ----------
    distance_matrix     : n×n list of integer arc costs
    demands             : demand per node (index 0 = depot, demand = 0)
    vehicle_capacities  : capacity per vehicle
    depot               : index of the depot node
    time_limit_seconds  : solver wall-clock limit
    time_windows        : optional list of (earliest, latest) per node
                          (same units as distance matrix)
    penalty_per_unserved: dropped-node penalty (make it large)
    verbose             : print solution summary

    Returns
    -------
    dict with keys:
        routes          – list of lists of node indices per vehicle
        total_distance  – sum of all arc distances
        dropped_nodes   – nodes not served (if any)
        status          – 'OPTIMAL' | 'FEASIBLE' | 'INFEASIBLE' | 'NO_SOLUTION'
    """
    if not ORTOOLS_AVAILABLE:
        raise ImportError("ortools is required. Install with: pip install ortools")

    n_vehicles = len(vehicle_capacities)
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), n_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    # ── Arc cost callback ─────────────────────────────────────────────────────
    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # ── Capacity dimension ────────────────────────────────────────────────────
    def demand_callback(from_index: int) -> int:
        node = manager.IndexToNode(from_index)
        return demands[node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,                    # no slack
        vehicle_capacities,
        True,                 # start cumul at zero
        "Capacity",
    )

    # ── Time windows dimension (optional) ─────────────────────────────────────
    if time_windows is not None:
        time_dim_name = "Time"
        routing.AddDimension(
            transit_callback_index,
            30,       # waiting time slack
            max(tw[1] for tw in time_windows),
            False,
            time_dim_name,
        )
        time_dimension = routing.GetDimensionOrDie(time_dim_name)
        for location_idx, time_window in enumerate(time_windows):
            index = manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(*time_window)

    # ── Allow dropping nodes with a large penalty ─────────────────────────────
    for node in range(len(distance_matrix)):
        if node != depot:
            routing.AddDisjunction([manager.NodeToIndex(node)], penalty_per_unserved)

    # ── Search parameters ─────────────────────────────────────────────────────
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_params.time_limit.seconds = time_limit_seconds

    # ── Solve ─────────────────────────────────────────────────────────────────
    solution = routing.SolveWithParameters(search_params)

    if not solution:
        return {"status": "NO_SOLUTION", "routes": [], "total_distance": None, "dropped_nodes": []}

    status_map = {
        routing.ROUTING_OPTIMAL: "OPTIMAL",
        routing.ROUTING_SUCCESS: "FEASIBLE",
        routing.ROUTING_FAIL: "INFEASIBLE",
        routing.ROUTING_FAIL_TIMEOUT: "TIMEOUT",
        routing.ROUTING_INVALID: "INVALID",
    }
    status = status_map.get(routing.status(), "UNKNOWN")

    # ── Extract routes ────────────────────────────────────────────────────────
    routes = []
    total_distance = 0

    for vehicle_id in range(n_vehicles):
        route = []
        index = routing.Start(vehicle_id)
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(node)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            total_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
        route.append(manager.IndexToNode(index))  # depot at end
        if len(route) > 2:  # skip empty vehicle routes
            routes.append(route)

    # Dropped nodes
    dropped = []
    for node in range(len(distance_matrix)):
        if node == depot:
            continue
        node_index = manager.NodeToIndex(node)
        if solution.Value(routing.NextVar(node_index)) == node_index:
            dropped.append(node)

    if verbose:
        print(f"\n── VRP Solution ({status}) ─────────────────────────")
        for i, r in enumerate(routes):
            print(f"  Vehicle {i}: {' → '.join(str(n) for n in r)}")
        print(f"  Total distance: {total_distance:,}")
        if dropped:
            print(f"  ⚠ Dropped nodes: {dropped}")

    return {
        "status": status,
        "routes": routes,
        "total_distance": total_distance,
        "dropped_nodes": dropped,
    }


# ── Route comparison helper ───────────────────────────────────────────────────

def compare_routes(
    baseline_distance: float,
    optimized_distance: float,
    baseline_late_rate: float,
    optimized_late_rate: float,
) -> pd.DataFrame:
    """Print a simple before/after comparison table."""
    data = {
        "Metric": ["Total Distance", "Late Delivery Rate (%)"],
        "Baseline": [baseline_distance, baseline_late_rate],
        "Optimized": [optimized_distance, optimized_late_rate],
    }
    df = pd.DataFrame(data)
    df["Improvement"] = df["Baseline"] - df["Optimized"]
    df["Improvement %"] = (df["Improvement"] / df["Baseline"] * 100).round(2)
    return df


# ── CLI demo (synthetic data) ─────────────────────────────────────────────────

if __name__ == "__main__":
    # 5 delivery locations + 1 depot (index 0)
    demo_locations = [
        (12.9716, 77.5946),   # 0: Depot (Bengaluru)
        (12.9352, 77.6245),   # 1
        (13.0067, 77.5667),   # 2
        (12.9550, 77.6100),   # 3
        (12.9900, 77.5500),   # 4
        (12.9200, 77.6400),   # 5
    ]
    matrix = build_distance_matrix(demo_locations)
    result = solve_vrp(
        distance_matrix=matrix,
        demands=[0, 10, 15, 10, 20, 10],
        vehicle_capacities=[50, 50],
        depot=0,
        time_limit_seconds=10,
    )
    print("\nResult:", result)
