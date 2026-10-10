"""Summary and chart helpers for retrieved historical weather with model estimates."""

from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "static" / "images"


def summarize_records(records):
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("No historical weather rows are available to summarize.")
    frame["predicted_demand"] = pd.to_numeric(frame["predicted_demand"], errors="coerce")
    demand = frame["predicted_demand"].dropna()
    hour_means = frame.groupby("hour")["predicted_demand"].mean()
    day_means = frame.groupby("date")["predicted_demand"].mean()
    traffic_means = frame[frame["traffic_status"] != "unknown"].groupby("traffic_status")["predicted_demand"].mean()
    weather_means = frame.groupby("weather_condition")["predicted_demand"].mean().dropna()
    daily = frame.assign(day_type=frame["day_of_week"].map(lambda day: "Weekend" if int(day) >= 5 else "Weekday"))
    weekday_means = daily.groupby("day_type")["predicted_demand"].mean()
    valid_traffic = frame[frame["traffic_status"] != "unknown"]["traffic_status"]

    return {
        "row_count": int(len(frame)),
        "average_demand": float(demand.mean()),
        "maximum_demand": float(demand.max()),
        "minimum_demand": float(demand.min()),
        "peak_hour": int(hour_means.idxmax()),
        "peak_hour_average": float(hour_means.max()),
        "most_active_day": str(day_means.idxmax()),
        "most_active_day_average": float(day_means.max()),
        "average_traffic": "Unavailable" if valid_traffic.empty else valid_traffic.mode().iloc[0].replace("_", " "),
        "traffic_coverage": int(len(valid_traffic)),
        "average_temperature": _mean(frame.get("temperature")),
        "average_humidity": _mean(frame.get("relative_humidity")),
        "weather_means": {str(key): float(value) for key, value in weather_means.items()},
        "traffic_means": {str(key): float(value) for key, value in traffic_means.items()},
        "weekday_means": {str(key): float(value) for key, value in weekday_means.items()},
    }


def _mean(values):
    if values is None:
        return None
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else None


def analyze_records(records):
    summary = summarize_records(records)
    frame = pd.DataFrame(records)
    observations = [
        f"The highest mean model-estimated demand occurred at {summary['peak_hour']:02d}:00 "
        f"({summary['peak_hour_average']:.1f} sessions/hour) in this selected dataset."
    ]
    if summary["weather_means"]:
        weather_name, weather_value = max(summary["weather_means"].items(), key=lambda item: item[1])
        observations.append(
            f"{weather_name} has the highest mean model estimate in these retrieved records "
            f"({weather_value:.1f} sessions/hour); this is an association in this sample, not a causal finding."
        )
    if summary["traffic_coverage"]:
        traffic_name, traffic_value = max(summary["traffic_means"].items(), key=lambda item: item[1])
        observations.append(
            f"Among rows with a traffic indicator, {traffic_name.replace('_', ' ')} has the highest mean estimate "
            f"({traffic_value:.1f} sessions/hour)."
        )
    else:
        observations.append(
            "Historical traffic was unavailable from the configured public sources, so traffic-vs-demand "
            "cannot be evaluated for this date range."
        )
    try:
        correlation = frame[["temperature", "predicted_demand"]].corr().iloc[0, 1]
        if math.isfinite(float(correlation)):
            direction = "positive" if correlation > 0.1 else ("negative" if correlation < -0.1 else "weak")
            observations.append(
                f"Temperature and model-estimated demand have a {direction} correlation "
                f"(r={float(correlation):.2f}) in this selected sample."
            )
    except (KeyError, ValueError, TypeError):
        pass
    return summary, observations


def create_historical_charts(records, prefix):
    """Save six charts; traffic chart explicitly says unavailable if no feed exists."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")
    frame = pd.DataFrame(records)
    frame["predicted_demand"] = pd.to_numeric(frame["predicted_demand"], errors="coerce")
    frame["local_time"] = pd.to_datetime(frame["local_time"])
    outputs = {}

    def save(fig, name):
        path = IMAGE_DIR / f"{prefix}_{name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs[name] = f"images/{path.name}"

    fig, ax = plt.subplots(figsize=(8, 4.8))
    hourly = frame.groupby("hour", as_index=False)["predicted_demand"].mean()
    sns.lineplot(data=hourly, x="hour", y="predicted_demand", marker="o", color="#2b7864", ax=ax)
    ax.set(xticks=range(0, 24, 2), title="Model-Estimated Demand by Hour",
           xlabel="Hour of day", ylabel="Estimated sessions/hour")
    save(fig, "demand_by_hour")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    daily = frame.groupby("date", as_index=False)["predicted_demand"].mean()
    sns.lineplot(data=daily, x="date", y="predicted_demand", marker="o", color="#2b7864", ax=ax)
    ax.tick_params(axis="x", rotation=35)
    ax.set(title="Model-Estimated Demand over the Selected Dates",
           xlabel="Date", ylabel="Mean estimated sessions/hour")
    save(fig, "demand_over_time")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    traffic = frame[frame["traffic_status"] != "unknown"]
    if traffic.empty:
        ax.text(0.5, 0.5, "Historical traffic data is unavailable",
                ha="center", va="center", fontsize=13, color="#65756f")
        ax.set_axis_off()
    else:
        sns.boxplot(data=traffic, x="traffic_status", y="predicted_demand",
                    color="#83b9a5", ax=ax)
        ax.set(title="Estimated Demand by Retrieved Traffic Indicator",
               xlabel="Traffic status", ylabel="Estimated sessions/hour")
    save(fig, "traffic_vs_demand")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    valid = frame.dropna(subset=["temperature", "predicted_demand"])
    if valid.empty:
        ax.text(0.5, 0.5, "Temperature data is unavailable", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.scatter(valid["temperature"], valid["predicted_demand"], alpha=0.45, color="#2b7864")
        ax.set(title="Temperature vs Model-Estimated Demand",
               xlabel="Temperature (°C)", ylabel="Estimated sessions/hour")
    save(fig, "temperature_vs_demand")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    weather = frame.groupby("weather_condition", as_index=False)["predicted_demand"].mean()
    sns.barplot(data=weather, x="weather_condition", y="predicted_demand",
                color="#83b9a5", ax=ax)
    ax.set(title="Estimated Demand by Retrieved Weather",
           xlabel="Weather condition", ylabel="Mean estimated sessions/hour")
    save(fig, "weather_vs_demand")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    frame = frame.assign(day_type=frame["day_of_week"].map(lambda day: "Weekend" if int(day) >= 5 else "Weekday"))
    day_type = frame.groupby("day_type", as_index=False)["predicted_demand"].mean()
    sns.barplot(data=day_type, x="day_type", y="predicted_demand",
                order=["Weekday", "Weekend"], color="#b6cfbf", ax=ax)
    ax.set(title="Weekday vs Weekend Model Estimates",
           xlabel="Day type", ylabel="Mean estimated sessions/hour")
    save(fig, "weekday_weekend")
    return outputs
