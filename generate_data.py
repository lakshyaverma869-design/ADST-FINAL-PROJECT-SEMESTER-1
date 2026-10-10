"""Generate the reproducible synthetic training data used by this student project."""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from model_utils import traffic_status_from_speed

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "ev_charging_data.csv"
CITY_SETTINGS = {
    "Bengaluru": {"traffic": 880, "temperature": 25.0, "ev_share": 0.050, "market": 1.4},
    "Delhi": {"traffic": 1180, "temperature": 19.0, "ev_share": 0.032, "market": 1.0},
    "Mumbai": {"traffic": 1080, "temperature": 28.0, "ev_share": 0.043, "market": 1.6},
    "Pune": {"traffic": 720, "temperature": 24.0, "ev_share": 0.046, "market": 0.9},
}
WEATHER = ["Clear", "Cloudy", "Rainy", "Foggy"]
WEATHER_PROBABILITY = [0.56, 0.24, 0.14, 0.06]


def generate_dataset(seed=42):
    """Create 90 days of hourly synthetic observations for four cities.

    Traffic volume, speed, EV count, weather, and the target are generated
    solely to create a training exercise. They are not live or measured data.
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2025-01-01", periods=90 * 24, freq="h")
    records = []

    for city, settings in CITY_SETTINGS.items():
        for timestamp in timestamps:
            hour = int(timestamp.hour)
            day_of_week = int(timestamp.dayofweek)
            is_weekend = day_of_week >= 5
            morning_peak = np.exp(-((hour - 8) / 2.2) ** 2)
            evening_peak = np.exp(-((hour - 18) / 2.7) ** 2)
            midday_activity = np.exp(-((hour - 13) / 4.2) ** 2)
            hourly_factor = 0.28 + 0.78 * morning_peak + 0.98 * evening_peak + 0.35 * midday_activity
            weekend_factor = 0.78 if is_weekend else 1.0
            traffic_volume = int(
                np.clip(
                    rng.normal(
                        settings["traffic"] * hourly_factor * weekend_factor,
                        settings["traffic"] * 0.075,
                    ),
                    80,
                    3600,
                )
            )

            average_speed = float(
                np.clip(66 - 0.018 * traffic_volume + rng.normal(0, 4.0), 8, 68)
            )
            traffic_status = traffic_status_from_speed(average_speed)
            weather_condition = str(rng.choice(WEATHER, p=WEATHER_PROBABILITY))
            rainfall = (
                float(rng.gamma(shape=1.4, scale=1.8))
                if weather_condition == "Rainy"
                else 0.0
            )
            daily_temperature = 3.2 * np.sin((hour - 8) * np.pi / 12)
            seasonal_adjustment = 1.2 * np.sin((timestamp.dayofyear - 20) * 2 * np.pi / 365)
            weather_temperature = -1.0 if weather_condition == "Rainy" else (
                -2.2 if weather_condition == "Foggy" else 0.0
            )
            temperature = float(
                np.clip(
                    settings["temperature"] + daily_temperature + seasonal_adjustment
                    + weather_temperature + rng.normal(0, 2.0),
                    5,
                    43,
                )
            )
            humidity_base = {
                "Clear": 44,
                "Cloudy": 64,
                "Rainy": 84,
                "Foggy": 90,
            }[weather_condition]
            relative_humidity = float(
                np.clip(humidity_base - 0.35 * (temperature - 24) + rng.normal(0, 8), 20, 100)
            )

            ev_count = int(rng.binomial(traffic_volume, settings["ev_share"]))
            peak_period = (
                "commute_peak"
                if 7 <= hour <= 10 or 16 <= hour <= 20
                else ("daytime" if 11 <= hour <= 15 else "off_peak")
            )
            hour_component = 3.6 if peak_period == "commute_peak" else (
                1.2 if peak_period == "daytime" else 0.15
            )
            weather_component = {
                "Clear": 0.8,
                "Cloudy": 0.35,
                "Rainy": -0.9,
                "Foggy": -0.45,
            }[weather_condition]
            traffic_component = {
                "free_flow": 0.0,
                "moderate": 1.1,
                "heavy": 2.0,
            }[traffic_status]
            weekend_component = 0.45 if is_weekend else 0.0
            comfort_component = max(0.0, abs(temperature - 24) - 8) * 0.08
            humidity_component = max(0.0, relative_humidity - 75) * 0.025
            mean_sessions = max(
                0.25,
                1.7
                + 0.27 * ev_count
                + 0.0015 * traffic_volume
                + hour_component
                + weather_component
                + traffic_component
                + weekend_component
                + comfort_component
                + humidity_component
                + settings["market"],
            )
            charging_demand = int(rng.poisson(mean_sessions))

            records.append(
                {
                    "date": timestamp.date().isoformat(),
                    "city": city,
                    "hour": hour,
                    "day_of_week": day_of_week,
                    "traffic_volume": traffic_volume,
                    "average_speed": round(average_speed, 1),
                    "traffic_status": traffic_status,
                    "temperature": round(temperature, 1),
                    "rainfall": round(rainfall, 1),
                    "relative_humidity": round(relative_humidity, 1),
                    "weather_condition": weather_condition,
                    "ev_count": ev_count,
                    "charging_demand": charging_demand,
                }
            )

    dataset = pd.DataFrame(records)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(DATA_PATH, index=False)
    print(f"Generated {len(dataset):,} synthetic records at {DATA_PATH}")
    print("No live or measured city traffic, EV count, or charging-session data is included.")
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    generate_dataset(seed=arguments.seed)
