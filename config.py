"""Project configuration and the supported city reference points."""
from pathlib import Path
import os

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

CITY_INFO = {
    "Bengaluru": {"latitude": 12.9716, "longitude": 77.5946, "timezone": "Asia/Kolkata"},
    "Delhi": {"latitude": 28.6139, "longitude": 77.2090, "timezone": "Asia/Kolkata"},
    "Mumbai": {"latitude": 19.0760, "longitude": 72.8777, "timezone": "Asia/Kolkata"},
    "Pune": {"latitude": 18.5204, "longitude": 73.8567, "timezone": "Asia/Kolkata"},
}
CITIES = list(CITY_INFO)
DATABASE_PATH = Path(os.getenv("EV_DB_PATH", str(BASE_DIR / "ev_charging.db")))
MODEL_PATH = BASE_DIR / "models" / "ev_charging_model.pkl"
METRICS_PATH = BASE_DIR / "models" / "model_metrics.json"
IMPORTANCE_PATH = BASE_DIR / "models" / "feature_importance.json"
TRAINING_INSIGHTS_PATH = BASE_DIR / "models" / "training_insights.json"
DATA_PATH = BASE_DIR / "data" / "ev_charging_data.csv"
WEATHER_TTL_SECONDS = 20 * 60
TRAFFIC_TTL_SECONDS = 5 * 60
FORECAST_TTL_SECONDS = 30 * 60
HISTORICAL_TTL_SECONDS = 12 * 60 * 60
MAX_HISTORICAL_DAYS = 31
MAX_FORECAST_DAYS = 16
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
TOMTOM_FLOW_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY", "").strip()
