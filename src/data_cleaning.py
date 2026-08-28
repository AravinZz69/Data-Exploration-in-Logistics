"""
data_cleaning.py
----------------
Utilities for loading and cleaning raw logistics data tables.
Handles schema validation, duplicate detection, missing values,
timestamp parsing, range checks, and key integrity checks.
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def load_orders(filepath: str | Path = RAW_DIR / "orders.csv") -> pd.DataFrame:
    df = pd.read_csv(filepath)
    for col in ["order_date", "promised_date", "actual_delivery_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_inventory(filepath: str | Path = RAW_DIR / "inventory.csv") -> pd.DataFrame:
    df = pd.read_csv(filepath)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def load_routes(filepath: str | Path = RAW_DIR / "routes.csv") -> pd.DataFrame:
    df = pd.read_csv(filepath)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def check_schema(df: pd.DataFrame, required_cols: list[str], table_name: str = "") -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{table_name}] Missing columns: {missing}")


def report_missing(df: pd.DataFrame, table_name: str = "") -> pd.Series:
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    if miss.empty:
        print(f"[{table_name}] No missing values found.")
    else:
        print(f"\n[{table_name}] Missing value rates:")
        print(miss.to_string())
    return miss


def drop_duplicates_report(df: pd.DataFrame, subset: list[str], table_name: str = "") -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=subset)
    removed = before - len(df)
    print(f"[{table_name}] Duplicates removed: {removed} (kept {len(df)} rows)")
    return df


def validate_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    if {"order_date", "actual_delivery_date"}.issubset(df.columns):
        df["timestamp_error"] = df["actual_delivery_date"] < df["order_date"]
        n_errors = df["timestamp_error"].sum()
        if n_errors > 0:
            print(f"  ⚠ Timestamp errors (delivery before order): {n_errors} rows")
    return df


def validate_ranges(df: pd.DataFrame) -> pd.DataFrame:
    numeric_non_negative = ["quantity", "shipping_cost", "distance_km", "stop_count", "load_weight", "vehicle_capacity"]
    for col in numeric_non_negative:
        if col in df.columns:
            n_neg = (df[col] < 0).sum()
            if n_neg > 0:
                print(f"  ⚠ Negative values in '{col}': {n_neg} rows")
    for lat_col in [c for c in df.columns if "lat" in c.lower()]:
        invalid = (~df[lat_col].between(-90, 90)).sum()
        if invalid:
            print(f"  ⚠ Invalid latitude values in '{lat_col}': {invalid} rows")
    for lon_col in [c for c in df.columns if "lon" in c.lower()]:
        invalid = (~df[lon_col].between(-180, 180)).sum()
        if invalid:
            print(f"  ⚠ Invalid longitude values in '{lon_col}': {invalid} rows")
    return df


def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    required = ["order_id", "order_date", "promised_date", "actual_delivery_date", "product_id", "warehouse_id", "quantity", "shipping_cost"]
    check_schema(df, required, "orders")
    report_missing(df, "orders")
    df = drop_duplicates_report(df, subset=["order_id"], table_name="orders")
    return validate_ranges(validate_timestamps(df))


def clean_inventory(df: pd.DataFrame) -> pd.DataFrame:
    required = ["date", "warehouse_id", "product_id", "opening_stock", "receipts", "demand", "closing_stock", "stockout_flag"]
    check_schema(df, required, "inventory")
    report_missing(df, "inventory")
    df = drop_duplicates_report(df, subset=["date", "warehouse_id", "product_id"], table_name="inventory")
    expected = df["opening_stock"] + df["receipts"] - df["demand"]
    errors = ((df["closing_stock"] - expected).abs() > 0.01).sum()
    if errors > 0:
        print(f"  ⚠ Inventory balance errors: {errors} rows")
    return df


def clean_routes(df: pd.DataFrame) -> pd.DataFrame:
    required = ["route_id", "date", "vehicle_id", "warehouse_id", "total_distance", "total_time", "stop_count"]
    check_schema(df, required, "routes")
    report_missing(df, "routes")
    df = drop_duplicates_report(df, subset=["route_id"], table_name="routes")
    return validate_ranges(df)


def save_processed(df: pd.DataFrame, filename: str) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / filename
    df.to_csv(out, index=False)
    print(f"  ✓ Saved: {out}")


if __name__ == "__main__":
    print("Loading raw data …")
    orders = clean_orders(load_orders()); save_processed(orders, "orders_clean.csv")
    inventory = clean_inventory(load_inventory()); save_processed(inventory, "inventory_clean.csv")
    routes = clean_routes(load_routes()); save_processed(routes, "routes_clean.csv")
    print("\nData cleaning complete.")
