"""
models.py
---------
Predictive modelling pipelines for late-delivery classification,
demand forecasting, and customer/route clustering.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, roc_auc_score, mean_absolute_error, mean_squared_error, silhouette_score
from sklearn.cluster import KMeans

MODELS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

LATE_DELIVERY_FEATURES = ["distance_km", "stop_count", "loading_time_hours", "congestion_level", "vehicle_capacity", "order_quantity", "day_of_week", "is_weekend_order"]
LATE_DELIVERY_CATEGORICALS: list[str] = []


def train_late_delivery_model(df: pd.DataFrame, features: list[str] | None = None, target: str = "late_delivery_flag", test_size: float = 0.20, random_state: int = 42, save: bool = True) -> tuple[Pipeline, dict]:
    features = features or [c for c in LATE_DELIVERY_FEATURES if c in df.columns]
    if not features:
        raise ValueError("No valid feature columns found in DataFrame.")
    X, y = df[features].copy(), df[target].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    cat_cols = [c for c in features if c in LATE_DELIVERY_CATEGORICALS]
    num_cols = [c for c in features if c not in cat_cols]
    preprocessor = ColumnTransformer([("num", StandardScaler(), num_cols), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)], remainder="drop")
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", RandomForestClassifier(n_estimators=300, random_state=random_state, class_weight="balanced", n_jobs=-1))])
    pipeline.fit(X_train, y_train)
    pred, prob = pipeline.predict(X_test), pipeline.predict_proba(X_test)[:, 1]
    report, roc_auc = classification_report(y_test, pred, output_dict=True), roc_auc_score(y_test, prob)
    metrics = {"precision": report["1"]["precision"], "recall": report["1"]["recall"], "f1": report["1"]["f1-score"], "roc_auc": roc_auc}
    print(classification_report(y_test, pred)); print(f"ROC-AUC: {roc_auc:.4f}")
    if save: joblib.dump(pipeline, MODELS_DIR / "late_delivery_model.joblib")
    return pipeline, metrics


DEMAND_FEATURES = ["rolling_demand_7d", "demand_cv", "days_of_cover", "month", "week_number", "day_of_week"]


def train_demand_model(df: pd.DataFrame, features: list[str] | None = None, target: str = "demand", test_size: float = 0.20, random_state: int = 42, save: bool = True) -> tuple[Pipeline, dict]:
    features = features or [c for c in DEMAND_FEATURES if c in df.columns]
    if not features: raise ValueError("No valid feature columns found in DataFrame.")
    df_model = df.dropna(subset=features + [target])
    X, y = df_model[features], df_model[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    pipeline = Pipeline([("scaler", StandardScaler()), ("regressor", RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1))])
    pipeline.fit(X_train, y_train); pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, pred); rmse = np.sqrt(mean_squared_error(y_test, pred)); mape = np.mean(np.abs((y_test - pred) / y_test.replace(0, np.nan))) * 100
    metrics = {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE_%": round(mape, 2)}
    print(metrics)
    if save: joblib.dump(pipeline, MODELS_DIR / "demand_forecast_model.joblib")
    return pipeline, metrics


def find_best_k(X_scaled: np.ndarray, k_range: range = range(2, 8)) -> int:
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto"); labels = km.fit_predict(X_scaled); scores[k] = silhouette_score(X_scaled, labels)
    return max(scores, key=scores.get)


def cluster_customers(customer_df: pd.DataFrame, features: list[str] | None = None, k_range: range = range(2, 7), save: bool = True) -> tuple[pd.DataFrame, KMeans, StandardScaler]:
    default_features = ["order_frequency", "avg_order_value", "avg_distance_km"]
    features = features or [c for c in default_features if c in customer_df.columns]
    if not features: raise ValueError("No valid feature columns found in customer_df.")
    X = customer_df[features].dropna(); scaler = StandardScaler(); X_scaled = scaler.fit_transform(X); best_k = find_best_k(X_scaled, k_range)
    model = KMeans(n_clusters=best_k, random_state=42, n_init="auto"); customer_df = customer_df.copy(); customer_df.loc[X.index, "customer_cluster"] = model.fit_predict(X_scaled); customer_df["customer_cluster"] = customer_df["customer_cluster"].astype("Int64")
    if save: joblib.dump({"model": model, "scaler": scaler, "features": features}, MODELS_DIR / "customer_cluster_model.joblib")
    return customer_df, model, scaler


def cluster_routes(routes_df: pd.DataFrame, features: list[str] | None = None, k_range: range = range(2, 7), save: bool = True) -> tuple[pd.DataFrame, KMeans, StandardScaler]:
    default_features = ["total_distance", "stop_count", "vehicle_utilization", "route_distance_per_stop", "avg_speed_kmh"]
    features = features or [c for c in default_features if c in routes_df.columns]
    if not features: raise ValueError("No valid feature columns found in routes_df.")
    X = routes_df[features].dropna(); scaler = StandardScaler(); X_scaled = scaler.fit_transform(X); best_k = find_best_k(X_scaled, k_range)
    model = KMeans(n_clusters=best_k, random_state=42, n_init="auto"); routes_df = routes_df.copy(); routes_df.loc[X.index, "route_cluster"] = model.fit_predict(X_scaled); routes_df["route_cluster"] = routes_df["route_cluster"].astype("Int64")
    if save: joblib.dump({"model": model, "scaler": scaler, "features": features}, MODELS_DIR / "route_cluster_model.joblib")
    return routes_df, model, scaler
