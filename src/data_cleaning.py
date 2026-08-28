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


# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_orders(filepath: str | Path = RAW_DIR / "orders.csv") -> pd.DataFrame:
    """Load and parse the orders table."""
    df = pd.read_csv(filepath)
    date_cols = ["order_date", "promised_date", "actual_delivery_date"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_inventory(filepath: str | Path = RAW_DIR / "inventory.csv") -> pd.DataFrame:
    """Load and parse the daily inventory snapshot table."""
    df = pd.read_csv(filepath)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def load_routes(filepath: str | Path = RAW_DIR / "routes.csv") -> pd.DataFrame:
    """Load and parse the routes table."""
    df = pd.read_csv(filepath)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


# ── Validators ────────────────────────────────────────────────────────────────

def check_schema(df: pd.DataFrame, required_cols: list[str], table_name: str = "") -> None:
    """Raise ValueError if any required column is missing."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{table_name}] Missing columns: {missing}")


def report_missing(df: pd.DataFrame, table_name: str = "") -> pd.Series:
    """Print and return missing-value rates per column (descending)."""
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    if miss.empty:
        print(f"[{table_name}] No missing values found.")
    else:
        print(f"\n[{table_name}] Missing value rates:")
        print(miss.to_string())
    return miss


def drop_duplicates_report(df: pd.DataFrame, subset: list[str], table_name: str = "") -> pd.DataFrame:
    """Drop duplicate rows on *subset* and report how many were removed."""
    before = len(df)
    df = df.drop_duplicates(subset=subset)
    removed = before - len(df)
    print(f"[{table_name}] Duplicates removed: {removed} (kept {len(df)} rows)")
    return df


# ── Timestamp checks ──────────────────────────────────────────────────────────

def validate_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag rows where actual_delivery_date < order_date (impossible sequence).
    Returns the DataFrame with a new boolean column `timestamp_error`.
    """
    if {"order_date", "actual_delivery_date"}.issubset(df.columns):
        df["timestamp_error"] = df["actual_delivery_date"] < df["order_date"]
        n_errors = df["timestamp_error"].sum()
        if n_errors > 0:
            print(f"  ⚠ Timestamp errors (delivery before order): {n_errors} rows")
    return df


# ── Range checks ──────────────────────────────────────────────────────────────

def validate_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for negative quantities and other out-of-range values.
    Prints warnings; does not remove rows (caller decides how to handle).
    """
    numeric_non_negative = ["quantity", "shipping_cost", "distance_km",
                            "stop_count", "load_weight", "vehicle_capacity"]
    for col in numeric_non_negative:
        if col in df.columns:
            n_neg = (df[col] < 0).sum()
            if n_neg > 0:
                print(f"  ⚠ Negative values in '{col}': {n_neg} rows")

    # Coordinate checks
    for lat_col in [c for c in df.columns if "lat" in c.lower()]:
        invalid = (~df[lat_col].between(-90, 90)).sum()
        if invalid:
            print(f"  ⚠ Invalid latitude values in '{lat_col}': {invalid} rows")
    for lon_col in [c for c in df.columns if "lon" in c.lower()]:
        invalid = (~df[lon_col].between(-180, 180)).sum()
        if invalid:
            print(f"  ⚠ Invalid longitude values in '{lon_col}': {invalid} rows")

    return df


# ── Main cleaning pipeline ────────────────────────────────────────────────────

def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline for the orders table."""
    required = ["order_id", "order_date", "promised_date", "actual_delivery_date",
                "product_id", "warehouse_id", "quantity", "shipping_cost"]
    check_schema(df, required, "orders")
    report_missing(df, "orders")
    df = drop_duplicates_report(df, subset=["order_id"], table_name="orders")
    df = validate_timestamps(df)
    df = validate_ranges(df)
    return df


def clean_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline for the inventory snapshot table."""
    required = ["date", "warehouse_id", "product_id", "opening_stock",
                "receipts", "demand", "closing_stock", "stockout_flag"]
    check_schema(df, required, "inventory")
    report_missing(df, "inventory")
    df = drop_duplicates_report(df, subset=["date", "warehouse_id", "product_id"],
                                table_name="inventory")
    # Closing stock should equal opening_stock + receipts - demand (tolerance 0.01)
    if all(c in df.columns for c in ["opening_stock", "receipts", "demand", "closing_stock"]):
        expected = df["opening_stock"] + df["receipts"] - df["demand"]
        balance_errors = ((df["closing_stock"] - expected).abs() > 0.01).sum()
        if balance_errors > 0:
            print(f"  ⚠ Inventory balance errors: {balance_errors} rows")
    return df


def clean_routes(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline for the routes table."""
    required = ["route_id", "date", "vehicle_id", "warehouse_id",
                "total_distance", "total_time", "stop_count"]
    check_schema(df, required, "routes")
    report_missing(df, "routes")
    df = drop_duplicates_report(df, subset=["route_id"], table_name="routes")
    df = validate_ranges(df)
    return df


def save_processed(df: pd.DataFrame, filename: str) -> None:
    """Save a cleaned DataFrame to data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / filename
    df.to_csv(out, index=False)
    print(f"  ✓ Saved: {out}")


# ── CLI entry ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading raw data …")
    orders = load_orders()
    inventory = load_inventory()
    routes = load_routes()

    print("\nCleaning orders …")
    orders = clean_orders(orders)
    save_processed(orders, "orders_clean.csv")

    print("\nCleaning inventory …")
    inventory = clean_inventory(inventory)
    save_processed(inventory, "inventory_clean.csv")

    print("\nCleaning routes …")
    routes = clean_routes(routes)
    save_processed(routes, "routes_clean.csv")

    print("\nData cleaning complete.")
