"""
features.py
-----------
Feature engineering functions for the logistics analytics project.
Produces derived columns used in predictive models, clustering, and
KPI dashboards.
"""

import pandas as pd
import numpy as np


# ── Delivery / Lead-time features ─────────────────────────────────────────────

def add_delivery_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Requires columns: order_date, promised_date, actual_delivery_date.
    Adds:
        on_time              – bool: delivered on or before promised date
        late_delivery_flag   – int (0/1): inverse of on_time
        delivery_delay_hours – float: actual - promised (negative = early)
        lead_time_hours      – float: actual_delivery - order
        promised_lead_hours  – float: promised - order
        day_of_week          – int (0=Monday … 6=Sunday) of order_date
        is_weekend_order     – bool
        week_number          – ISO week number
        month                – calendar month (1-12)
    """
    df = df.copy()
    df["on_time"] = df["actual_delivery_date"] <= df["promised_date"]
    df["late_delivery_flag"] = (~df["on_time"]).astype(int)

    df["delivery_delay_hours"] = (
        (df["actual_delivery_date"] - df["promised_date"])
        .dt.total_seconds() / 3600
    )
    df["lead_time_hours"] = (
        (df["actual_delivery_date"] - df["order_date"])
        .dt.total_seconds() / 3600
    )
    df["promised_lead_hours"] = (
        (df["promised_date"] - df["order_date"])
        .dt.total_seconds() / 3600
    )
    df["day_of_week"] = df["order_date"].dt.dayofweek
    df["is_weekend_order"] = df["day_of_week"].isin([5, 6])
    df["week_number"] = df["order_date"].dt.isocalendar().week.astype(int)
    df["month"] = df["order_date"].dt.month
    return df


# ── Route / Transportation features ──────────────────────────────────────────

def add_route_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Requires columns: total_distance, stop_count, load_weight, vehicle_capacity.
    Adds:
        route_distance_per_stop  – distance / number of stops
        vehicle_utilization      – load_weight / vehicle_capacity (0-1)
        avg_speed_kmh            – total_distance / total_time (hours)
    """
    df = df.copy()

    if {"total_distance", "stop_count"}.issubset(df.columns):
        df["route_distance_per_stop"] = df["total_distance"] / df["stop_count"].replace(0, np.nan)

    if {"load_weight", "vehicle_capacity"}.issubset(df.columns):
        df["vehicle_utilization"] = df["load_weight"] / df["vehicle_capacity"].replace(0, np.nan)

    if {"total_distance", "total_time"}.issubset(df.columns):
        # total_time expected in hours
        df["avg_speed_kmh"] = df["total_distance"] / df["total_time"].replace(0, np.nan)

    return df


# ── Inventory features ────────────────────────────────────────────────────────

def add_inventory_features(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    Expects a DataFrame indexed by (date, warehouse_id, product_id) or with
    those columns present.  Adds rolling demand and demand variability.

    Requires columns: date, warehouse_id, product_id, demand, closing_stock.
    Adds:
        rolling_demand_7d    – rolling 7-day mean demand per SKU-warehouse
        demand_cv            – coefficient of variation of demand
        stockout_risk        – closing_stock / rolling_demand_7d  (days of cover)
    """
    df = df.copy().sort_values(["warehouse_id", "product_id", "date"])

    grp = df.groupby(["warehouse_id", "product_id"])

    df["rolling_demand_7d"] = (
        grp["demand"]
        .transform(lambda s: s.rolling(window, min_periods=1).mean())
    )
    df["demand_std_7d"] = (
        grp["demand"]
        .transform(lambda s: s.rolling(window, min_periods=2).std())
    )
    df["demand_cv"] = df["demand_std_7d"] / df["rolling_demand_7d"].replace(0, np.nan)

    if "closing_stock" in df.columns:
        df["days_of_cover"] = df["closing_stock"] / df["rolling_demand_7d"].replace(0, np.nan)

    return df


# ── Combined feature pipeline ─────────────────────────────────────────────────

def build_model_features(orders: pd.DataFrame,
                          routes: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Build the model-ready feature table by joining orders with (optional)
    route-level aggregates and adding all derived features.

    Returns a DataFrame ready for train/test split.
    """
    feat = add_delivery_features(orders)

    if routes is not None:
        route_cols = ["route_id", "vehicle_id", "warehouse_id", "date",
                      "total_distance", "stop_count", "total_time",
                      "load_weight", "vehicle_capacity", "fuel_cost"]
        route_subset = routes[[c for c in route_cols if c in routes.columns]].copy()
        route_subset = add_route_features(route_subset)

        # Aggregate to warehouse-date level for joining to orders
        agg_cols = {
            "route_distance_per_stop": "mean",
            "vehicle_utilization": "mean",
            "avg_speed_kmh": "mean",
            "stop_count": "sum",
            "total_distance": "sum",
        }
        agg_cols = {k: v for k, v in agg_cols.items() if k in route_subset.columns}
        route_agg = route_subset.groupby(["warehouse_id", "date"]).agg(agg_cols).reset_index()
        route_agg.rename(columns={"date": "order_date"}, inplace=True)

        feat = feat.merge(route_agg, on=["warehouse_id", "order_date"], how="left")

    return feat
