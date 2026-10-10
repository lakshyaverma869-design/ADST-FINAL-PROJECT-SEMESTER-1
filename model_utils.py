"""Shared prediction feature definitions and fitted preprocessing pipeline."""

from datetime import datetime
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline

TARGET_COLUMN = "charging_demand"
INPUT_COLUMNS = [
    "city",
    "hour",
    "day_of_week",
    "traffic_status",
    "temperature",
    "rainfall",
    "weather_condition",
    "relative_humidity",
]
FEATURE_COLUMNS = INPUT_COLUMNS + ["peak_period"]
CATEGORICAL_COLUMNS = ["city", "traffic_status", "weather_condition", "peak_period"]
NUMERIC_COLUMNS = ["hour", "day_of_week", "temperature", "rainfall", "relative_humidity"]


def peak_period_for_hour(hour):
    hour = int(hour)
    if 7 <= hour <= 10 or 16 <= hour <= 20:
        return "commute_peak"
    if 11 <= hour <= 15:
        return "daytime"
    return "off_peak"


def add_engineered_features(frame):
    result = frame.copy()
    result["peak_period"] = result["hour"].astype(int).map(peak_period_for_hour)
    return result


def traffic_status_from_ratio(ratio):
    if ratio is None:
        return "unknown"
    if ratio < 0.55:
        return "heavy"
    if ratio < 0.82:
        return "moderate"
    return "free_flow"


def traffic_status_from_speed(speed):
    if speed is None or pd.isna(speed):
        return "unknown"
    return traffic_status_from_ratio(float(speed) / 65.0)


def make_feature_row(city, local_time, weather, traffic_status="unknown"):
    """Use only fields available from city/time plus retrieved weather/traffic."""
    return {
        "city": city,
        "hour": int(local_time.hour),
        "day_of_week": int(local_time.weekday()),
        "traffic_status": traffic_status or "unknown",
        "temperature": weather.get("temperature"),
        "rainfall": weather.get("rainfall"),
        "weather_condition": weather.get("weather_condition") or "Unknown",
        "relative_humidity": weather.get("relative_humidity"),
    }


def prepare_model_features(payload):
    frame = pd.DataFrame([{key: payload.get(key) for key in INPUT_COLUMNS}])
    return add_engineered_features(frame)[FEATURE_COLUMNS]


def build_model():
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categories",
                make_pipeline(
                    SimpleImputer(strategy="constant", fill_value="Unknown"),
                    OneHotEncoder(handle_unknown="ignore"),
                ),
                CATEGORICAL_COLUMNS,
            ),
            (
                "numbers",
                make_pipeline(SimpleImputer(strategy="median")),
                NUMERIC_COLUMNS,
            ),
        ],
        remainder="drop",
    )
    regressor = RandomForestRegressor(
        n_estimators=240,
        max_depth=18,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocessing", preprocessing), ("regressor", regressor)])
