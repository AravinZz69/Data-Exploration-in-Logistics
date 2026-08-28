"""
optimization.py
---------------
Vehicle Routing Problem (VRP) solver using Google OR-Tools.
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


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2); dphi = math.radians(lat2 - lat1); dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def build_distance_matrix(locations: list[tuple[float, float]], scale: int = 1000) -> list[list[int]]:
    n = len(locations); matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j: matrix[i][j] = int(haversine_km(*locations[i], *locations[j]) * scale)
    return matrix


def solve_vrp(distance_matrix: list[list[int]], demands: list[int], vehicle_capacities: list[int],
              depot: int = 0, time_limit_seconds: int = 30,
              time_windows: list[tuple[int, int]] | None = None,
              penalty_per_unserved: int = 100_000, verbose: bool = True) -> dict[str, Any]:
    if not ORTOOLS_AVAILABLE: raise ImportError("ortools is required. Install with: pip install ortools")
    n_vehicles = len(vehicle_capacities); manager = pywrapcp.RoutingIndexManager(len(distance_matrix), n_vehicles, depot); routing = pywrapcp.RoutingModel(manager)
    def distance_callback(from_index: int, to_index: int) -> int:
        return distance_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]
    transit_callback_index = routing.RegisterTransitCallback(distance_callback); routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    def demand_callback(from_index: int) -> int:
        return demands[manager.IndexToNode(from_index)]
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(demand_callback_index, 0, vehicle_capacities, True, "Capacity")
    if time_windows is not None:
        routing.AddDimension(transit_callback_index, 30, max(tw[1] for tw in time_windows), False, "Time")
        time_dimension = routing.GetDimensionOrDie("Time")
        for location_idx, time_window in enumerate(time_windows):
            time_dimension.CumulVar(manager.NodeToIndex(location_idx)).SetRange(*time_window)
    for node in range(len(distance_matrix)):
        if node != depot: routing.AddDisjunction([manager.NodeToIndex(node)], penalty_per_unserved)
    search_params = pywrapcp.DefaultRoutingSearchParameters(); search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC; search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH; search_params.time_limit.seconds = time_limit_seconds
    solution = routing.SolveWithParameters(search_params)
    if not solution: return {"status": "NO_SOLUTION", "routes": [], "total_distance": None, "dropped_nodes": []}
    status_map = {routing.ROUTING_OPTIMAL: "OPTIMAL", routing.ROUTING_SUCCESS: "FEASIBLE", routing.ROUTING_FAIL: "INFEASIBLE", routing.ROUTING_FAIL_TIMEOUT: "TIMEOUT", routing.ROUTING_INVALID: "INVALID"}; status = status_map.get(routing.status(), "UNKNOWN")
    routes = []; total_distance = 0
    for vehicle_id in range(n_vehicles):
        route = []; index = routing.Start(vehicle_id)
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index); route.append(node); previous_index = index; index = solution.Value(routing.NextVar(index)); total_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
        route.append(manager.IndexToNode(index))
        if len(route) > 2: routes.append(route)
    dropped = []
    for node in range(len(distance_matrix)):
        if node == depot: continue
        node_index = manager.NodeToIndex(node)
        if solution.Value(routing.NextVar(node_index)) == node_index: dropped.append(node)
    if verbose:
        print(f"\n── VRP Solution ({status}) ─────────────────────────")
        for i, r in enumerate(routes): print(f"  Vehicle {i}: {' → '.join(str(n) for n in r)}")
        print(f"  Total distance: {total_distance:,}")
        if dropped: print(f"  ⚠ Dropped nodes: {dropped}")
    return {"status": status, "routes": routes, "total_distance": total_distance, "dropped_nodes": dropped}


def compare_routes(baseline_distance: float, optimized_distance: float, baseline_late_rate: float, optimized_late_rate: float) -> pd.DataFrame:
    data = {"Metric": ["Total Distance", "Late Delivery Rate (%)"], "Baseline": [baseline_distance, baseline_late_rate], "Optimized": [optimized_distance, optimized_late_rate]}
    df = pd.DataFrame(data); df["Improvement"] = df["Baseline"] - df["Optimized"]; df["Improvement %"] = (df["Improvement"] / df["Baseline"] * 100).round(2); return df

if __name__ == "__main__":
    demo_locations = [(12.9716, 77.5946), (12.9352, 77.6245), (13.0067, 77.5667), (12.9550, 77.6100), (12.9900, 77.5500), (12.9200, 77.6400)]
    matrix = build_distance_matrix(demo_locations)
    result = solve_vrp(distance_matrix=matrix, demands=[0, 10, 15, 10, 20, 10], vehicle_capacities=[50, 50], depot=0, time_limit_seconds=10)
    print("\nResult:", result)
