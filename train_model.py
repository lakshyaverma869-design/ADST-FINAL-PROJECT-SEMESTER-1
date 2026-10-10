"""Train and evaluate the Random Forest using the shared live-input features."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from config import DATA_PATH, IMPORTANCE_PATH, METRICS_PATH, MODEL_PATH, TRAINING_INSIGHTS_PATH
from model_utils import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    add_engineered_features,
    build_model,
    traffic_status_from_speed,
)

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "static" / "images"


def calculate_feature_importance(model):
    """Aggregate one-hot columns back to the original user-facing features."""
    transformer = model.named_steps["preprocessing"]
    names = transformer.get_feature_names_out()
    values = model.named_steps["regressor"].feature_importances_
    result = {name: 0.0 for name in FEATURE_COLUMNS}
    for name, value in zip(names, values):
        raw_name = name.split("__", 1)[-1]
        for column in FEATURE_COLUMNS:
            if raw_name == column or raw_name.startswith(column + "_"):
                result[column] += float(value)
                break
    return dict(sorted(result.items(), key=lambda item: item[1], reverse=True))


def train_and_evaluate():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}. Run python generate_data.py first.")
    data = pd.read_csv(DATA_PATH)
    rows_before = len(data)
    data = data.drop_duplicates().dropna(subset=[TARGET_COLUMN]).copy()
    if data.empty:
        raise ValueError("No rows with a target value remain after cleaning.")

    data["traffic_status"] = data["average_speed"].map(traffic_status_from_speed)
    # In 20% of synthetic training rows, traffic is intentionally unavailable.
    # This teaches a transparent Unknown category for API/key outages.
    mask = np.random.default_rng(42).random(len(data)) < 0.20
    data.loc[mask, "traffic_status"] = "unknown"
    data["hour"] = data["hour"].astype(int)
    data["day_of_week"] = data["day_of_week"].astype(int)
    data = add_engineered_features(data)

    x = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN].astype(float)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42
    )
    model = build_model()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    mse = float(mean_squared_error(y_test, predictions))

    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_test, predictions)),
        "rows_before_cleaning": int(rows_before),
        "rows_after_cleaning": int(len(data)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "target": TARGET_COLUMN,
        "target_unit": "model-estimated charging sessions per city-hour",
        "dataset_type": "synthetic",
        "missing_traffic_augmentation_share": 0.20,
        "random_state": 42,
        "features": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_COLUMNS,
        "numeric_features": NUMERIC_COLUMNS,
        "model": "RandomForestRegressor",
    }
    importance = calculate_feature_importance(model)
    insights = calculate_training_insights(data)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    IMPORTANCE_PATH.write_text(json.dumps(importance, indent=2), encoding="utf-8")
    TRAINING_INSIGHTS_PATH.write_text(json.dumps(insights, indent=2), encoding="utf-8")
    save_visualizations(data, y_test, predictions, importance)

    print("Model training and evaluation finished.")
    print(f"Rows: {len(data):,} | train: {len(x_train):,} | test: {len(x_test):,}")
    print(f"Features: {', '.join(FEATURE_COLUMNS)}")
    print(f"MAE: {metrics['mae']:.3f}")
    print(f"MSE: {metrics['mse']:.3f}")
    print(f"RMSE: {metrics['rmse']:.3f}")
    print(f"R2: {metrics['r2']:.3f}")
    print(f"Feature importance: {importance}")
    print("Training observations:")
    for item in insights["observations"]:
        print(f"- {item}")
    return model, metrics, importance


def calculate_training_insights(data):
    """Summarize generated training patterns without presenting them as city facts."""
    hourly = data.groupby("hour")[TARGET_COLUMN].mean()
    day_type = data.assign(
        day_type=np.where(data["day_of_week"] >= 5, "weekend", "weekday")
    ).groupby("day_type")[TARGET_COLUMN].mean()
    weather = data.groupby("weather_condition")[TARGET_COLUMN].mean().sort_values(ascending=False)
    correlation = data[["traffic_volume", TARGET_COLUMN]].corr().iloc[0, 1]
    observations = [
        f"Generated examples have their highest hourly mean at {int(hourly.idxmax()):02d}:00 "
        f"({hourly.max():.1f} sessions/hour) and lowest at {int(hourly.idxmin()):02d}:00 "
        f"({hourly.min():.1f} sessions/hour).",
    ]
    if "weekday" in day_type and "weekend" in day_type:
        difference = float(day_type["weekend"] - day_type["weekday"])
        direction = "higher" if difference >= 0 else "lower"
        observations.append(
            f"The generated weekend mean is {abs(difference):.1f} sessions/hour {direction} than the weekday mean."
        )
    if not weather.empty:
        observations.append(
            f"{weather.index[0]} has the highest generated mean among weather categories "
            f"({weather.iloc[0]:.1f} sessions/hour)."
        )
    if np.isfinite(correlation):
        observations.append(
            f"Generated traffic volume and target have Pearson r={correlation:.2f}; volume is an EDA-only field, not a prediction input."
        )
    return {
        "dataset_type": "synthetic",
        "observations": observations,
        "hourly_means": {str(int(hour)): round(float(value), 3) for hour, value in hourly.items()},
        "weather_means": {str(name): round(float(value), 3) for name, value in weather.items()},
        "weekday_weekend_means": {str(name): round(float(value), 3) for name, value in day_type.items()},
        "traffic_volume_target_correlation": float(correlation),
    }


def save_visualizations(data, actual, predicted, importance=None):
    sns.set_theme(style="whitegrid", palette="deep")
    sample = data.sample(min(1800, len(data)), random_state=42)

    fig, ax = plt.subplots(figsize=(8, 5))
    hourly = data.groupby("hour", as_index=False)[TARGET_COLUMN].mean()
    sns.lineplot(data=hourly, x="hour", y=TARGET_COLUMN, marker="o", color="#2b7864", ax=ax)
    ax.set(xticks=range(0, 24, 2), title="Synthetic Training Demand by Hour",
           xlabel="Hour of day", ylabel="Mean sessions/hour")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "demand_by_hour.png", dpi=150)
    plt.close(fig)

    hourly_traffic = data.groupby("hour", as_index=False)["traffic_volume"].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=hourly_traffic, x="hour", y="traffic_volume", marker="o", color="#b76d47", ax=ax)
    ax.set(xticks=range(0, 24, 2), title="Synthetic Training Traffic Volume by Hour",
           xlabel="Hour of day", ylabel="Generated vehicles/hour")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "traffic_by_hour.png", dpi=150)
    plt.close(fig)

    if "date" in data.columns:
        dated = data.copy()
        dated["date"] = pd.to_datetime(dated["date"], errors="coerce")
        daily = dated.groupby("date", as_index=False)[TARGET_COLUMN].mean()
        fig, ax = plt.subplots(figsize=(9, 4.8))
        sns.lineplot(data=daily, x="date", y=TARGET_COLUMN, color="#2b7864", ax=ax)
        ax.set(title="Generated Training Demand by Date", xlabel="Date",
               ylabel="Mean synthetic sessions/hour")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(IMAGE_DIR / "demand_over_time.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    order = ["free_flow", "moderate", "heavy", "unknown"]
    sns.barplot(data=data, x="traffic_status", y=TARGET_COLUMN, order=order,
                errorbar=None, color="#83b9a5", ax=ax)
    ax.set(title="Synthetic Training Demand by Traffic Indicator",
           xlabel="Traffic status", ylabel="Mean sessions/hour")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "traffic_impact.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(data=sample, x="traffic_volume", y=TARGET_COLUMN,
                scatter_kws={"alpha": 0.25, "s": 16},
                line_kws={"color": "#c25a36", "linewidth": 2}, ax=ax)
    ax.set(title="Synthetic Traffic Volume and Demand (EDA Only)",
           xlabel="Generated vehicles/hour", ylabel="Generated sessions/hour")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "traffic_vs_demand.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(data=sample, x="temperature", y=TARGET_COLUMN,
                scatter_kws={"alpha": 0.25, "s": 16},
                line_kws={"color": "#c25a36", "linewidth": 2}, ax=ax)
    ax.set(title="Synthetic Temperature and Demand",
           xlabel="Temperature (°C)", ylabel="Generated sessions/hour")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "temperature_vs_demand.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    weather_order = ["Clear", "Cloudy", "Rainy", "Foggy", "Other"]
    sns.barplot(data=data, x="weather_condition", y=TARGET_COLUMN,
                order=weather_order, errorbar=None, color="#83b9a5", ax=ax)
    ax.set(title="Synthetic Training Demand by Weather",
           xlabel="Weather condition", ylabel="Mean sessions/hour")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "demand_by_weather.png", dpi=150)
    plt.close(fig)

    data = data.assign(day_type=np.where(data["day_of_week"] >= 5, "Weekend", "Weekday"))
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=data, x="day_type", y=TARGET_COLUMN,
                order=["Weekday", "Weekend"], errorbar=None, color="#b6cfbf", ax=ax)
    ax.set(title="Synthetic Weekday vs Weekend Demand",
           xlabel="Day type", ylabel="Mean sessions/hour")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "weekday_weekend_demand.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual, predicted, alpha=0.35, color="#2b7864", edgecolor="none")
    upper = max(float(np.max(actual)), float(np.max(predicted))) * 1.05
    ax.plot([0, upper], [0, upper], "--", color="#c25a36", linewidth=1.5)
    ax.set(xlim=(0, upper), ylim=(0, upper), title="Actual vs Predicted (Synthetic Test Split)",
           xlabel="Synthetic target sessions/hour", ylabel="Predicted sessions/hour")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)

    if importance:
        labels = list(importance)
        values = [importance[label] for label in labels]
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=values, y=labels, color="#2b7864", ax=ax)
        ax.set(title="Random Forest Feature Importance",
               xlabel="Aggregated impurity importance", ylabel="Model feature")
        fig.tight_layout()
        fig.savefig(IMAGE_DIR / "feature_importance.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    train_and_evaluate()
