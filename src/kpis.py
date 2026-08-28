"""
kpis.py
-------
Compute and visualise logistics KPIs.

KPIs implemented
----------------
1. On-Time Delivery Rate
2. Late Delivery Rate
3. Average Delivery Lead Time (hours)
4. Stockout Rate
5. Inventory Turnover
6. Transportation Cost per Delivery
7. Route Distance per Delivery
8. Vehicle Utilization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ── Individual KPI functions ──────────────────────────────────────────────────

def on_time_delivery_rate(orders: pd.DataFrame) -> float:
    """Percentage of orders delivered on or before the promised date."""
    return orders["on_time"].mean() * 100


def late_delivery_rate(orders: pd.DataFrame) -> float:
    """Percentage of orders delivered after the promised date."""
    return 100 - on_time_delivery_rate(orders)


def avg_lead_time_hours(orders: pd.DataFrame) -> float:
    """Mean elapsed hours from order placement to actual delivery."""
    return orders["lead_time_hours"].mean()


def stockout_rate(inventory: pd.DataFrame) -> float:
    """
    Percentage of (date, warehouse, product) periods with a stockout.
    Requires a binary column `stockout_flag`.
    """
    return inventory["stockout_flag"].mean() * 100


def inventory_turnover(inventory: pd.DataFrame,
                        period_days: int = 30) -> float:
    """
    Annualised inventory turnover:
        Total demand over the period / average closing stock
    Multiplied by (365 / period_days) to annualise.
    """
    total_demand = inventory["demand"].sum()
    avg_stock = inventory["closing_stock"].mean()
    if avg_stock == 0:
        return np.nan
    turnover_period = total_demand / avg_stock
    return turnover_period * (365 / period_days)


def cost_per_delivery(orders: pd.DataFrame) -> float:
    """Average shipping cost per completed delivery."""
    if "shipping_cost" not in orders.columns:
        return np.nan
    return orders["shipping_cost"].mean()


def distance_per_delivery(routes: pd.DataFrame) -> float:
    """Average route distance (km) divided by the number of stops."""
    if {"total_distance", "stop_count"}.issubset(routes.columns):
        routes = routes.copy()
        routes["dist_per_stop"] = routes["total_distance"] / routes["stop_count"].replace(0, np.nan)
        return routes["dist_per_stop"].mean()
    return np.nan


def avg_vehicle_utilization(routes: pd.DataFrame) -> float:
    """Mean vehicle utilization across all routes (0–100 %)."""
    if "vehicle_utilization" in routes.columns:
        return routes["vehicle_utilization"].mean() * 100
    return np.nan


# ── Summary table ─────────────────────────────────────────────────────────────

def compute_kpi_summary(orders: pd.DataFrame,
                         inventory: pd.DataFrame | None = None,
                         routes: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Return a tidy DataFrame with all KPIs and their current values.
    Pass None for tables that are not yet available.
    """
    rows = [
        ("On-Time Delivery Rate (%)", on_time_delivery_rate(orders)),
        ("Late Delivery Rate (%)", late_delivery_rate(orders)),
        ("Avg Lead Time (hours)", avg_lead_time_hours(orders)),
        ("Transportation Cost / Delivery", cost_per_delivery(orders)),
    ]

    if inventory is not None:
        rows += [
            ("Stockout Rate (%)", stockout_rate(inventory)),
            ("Inventory Turnover (annualised)", inventory_turnover(inventory)),
        ]

    if routes is not None:
        rows += [
            ("Route Distance / Delivery (km)", distance_per_delivery(routes)),
            ("Avg Vehicle Utilization (%)", avg_vehicle_utilization(routes)),
        ]

    df = pd.DataFrame(rows, columns=["KPI", "Value"])
    df["Value"] = df["Value"].round(2)
    return df


# ── Time-series KPI plots ─────────────────────────────────────────────────────

def plot_daily_late_rate(orders: pd.DataFrame, save: bool = True) -> None:
    """Line chart of daily late-delivery rate."""
    daily = (
        orders.groupby(orders["order_date"].dt.date)
              .agg(late_rate=("on_time", lambda x: 1 - x.mean()))
              .reset_index()
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=daily, x="order_date", y="late_rate", ax=ax, color="#e05c5c")
    ax.set_title("Daily Late-Delivery Rate", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Late-delivery rate")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "daily_late_rate.png", dpi=150)
        print(f"Saved: {FIGURES_DIR / 'daily_late_rate.png'}")
    plt.show()


def plot_kpi_summary(kpi_df: pd.DataFrame, save: bool = True) -> None:
    """Horizontal bar chart of KPI values."""
    fig, ax = plt.subplots(figsize=(10, len(kpi_df) * 0.7 + 1))
    colors = ["#4c9be8"] * len(kpi_df)
    ax.barh(kpi_df["KPI"], kpi_df["Value"], color=colors)
    ax.set_xlabel("Value")
    ax.set_title("Logistics KPI Baseline", fontsize=14, fontweight="bold")
    for i, v in enumerate(kpi_df["Value"]):
        ax.text(v + 0.5, i, str(v), va="center", fontsize=9)
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "kpi_summary.png", dpi=150)
        print(f"Saved: {FIGURES_DIR / 'kpi_summary.png'}")
    plt.show()


# ── CLI entry ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path
    PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"

    orders = pd.read_csv(PROCESSED / "orders_clean.csv", parse_dates=["order_date", "promised_date", "actual_delivery_date"])
    # Ensure derived cols exist
    orders["on_time"] = orders["actual_delivery_date"] <= orders["promised_date"]
    orders["lead_time_hours"] = (orders["actual_delivery_date"] - orders["order_date"]).dt.total_seconds() / 3600

    kpi_df = compute_kpi_summary(orders)
    print("\nKPI Baseline:")
    print(kpi_df.to_string(index=False))

    plot_daily_late_rate(orders)
    plot_kpi_summary(kpi_df)
