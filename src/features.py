"""
features.py
-----------
Feature engineering functions for the logistics analytics project.
Produces derived columns used in predictive models, clustering, and KPI dashboards.
"""

import pandas as pd
import numpy as np


def add_delivery_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["on_time"] = df["actual_delivery_date"] <= df["promised_date"]
    df["late_delivery_flag"] = (~df["on_time"]).astype(int)
    df["delivery_delay_hours"] = ((df["actual_delivery_date"] - df["promised_date"]).dt.total_seconds() / 3600)
    df["lead_time_hours"] = ((df["actual_delivery_date"] - df["order_date"]).dt.total_seconds() / 3600)
    df["promised_lead_hours"] = ((df["promised_date"] - df["order_date"]).dt.total_seconds() / 3600)
    df["day_of_week"] = df["order_date"].dt.dayofweek
    df["is_weekend_order"] = df["day_of_week"].isin([5, 6])
    df["week_number"] = df["order_date"].dt.isocalendar().week.astype(int)
    df["month"] = df["order_date"].dt.month
    return df


def add_route_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"total_distance", "stop_count"}.issubset(df.columns):
        df["route_distance_per_stop"] = df["total_distance"] / df["stop_count"].replace(0, np.nan)
    if {"load_weight", "vehicle_capacity"}.issubset(df.columns):
        df["vehicle_utilization"] = df["load_weight"] / df["vehicle_capacity"].replace(0, np.nan)
    if {"total_distance", "total_time"}.issubset(df.columns):
        df["avg_speed_kmh"] = df["total_distance"] / df["total_time"].replace(0, np.nan)
    return df


def add_inventory_features(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    df = df.copy().sort_values(["warehouse_id", "product_id", "date"])
    grp = df.groupby(["warehouse_id", "product_id"])
    df["rolling_demand_7d"] = grp["demand"].transform(lambda s: s.rolling(window, min_periods=1).mean())
    df["demand_std_7d"] = grp["demand"].transform(lambda s: s.rolling(window, min_periods=2).std())
    df["demand_cv"] = df["demand_std_7d"] / df["rolling_demand_7d"].replace(0, np.nan)
    if "closing_stock" in df.columns:
        df["days_of_cover"] = df["closing_stock"] / df["rolling_demand_7d"].replace(0, np.nan)
    return df


def build_model_features(orders: pd.DataFrame, routes: pd.DataFrame | None = None) -> pd.DataFrame:
    feat = add_delivery_features(orders)
    if routes is not None:
        route_cols = ["route_id", "vehicle_id", "warehouse_id", "date", "total_distance", "stop_count", "total_time", "load_weight", "vehicle_capacity", "fuel_cost"]
        route_subset = routes[[c for c in route_cols if c in routes.columns]].copy()
        route_subset = add_route_features(route_subset)
        agg_cols = {k: v for k, v in {"route_distance_per_stop": "mean", "vehicle_utilization": "mean", "avg_speed_kmh": "mean", "stop_count": "sum", "total_distance": "sum"}.items() if k in route_subset.columns}
        route_agg = route_subset.groupby(["warehouse_id", "date"]).agg(agg_cols).reset_index()
        route_agg.rename(columns={"date": "order_date"}, inplace=True)
        feat = feat.merge(route_agg, on=["warehouse_id", "order_date"], how="left")
    return feat
