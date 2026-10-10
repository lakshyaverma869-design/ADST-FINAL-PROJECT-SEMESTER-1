"""Public weather and optional traffic-data clients with SQLite caching.

Weather uses Open-Meteo's non-commercial public endpoints. Optional live road
flow uses TomTom's Flow Segment Data API when TOMTOM_API_KEY is configured.
No volume, EV count, historical traffic, or charging-session observations are
invented when a source does not provide them.
"""

from collections import Counter
from datetime import date, datetime, timedelta
import json
import math
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from config import (
    CITY_INFO,
    FORECAST_TTL_SECONDS,
    HISTORICAL_TTL_SECONDS,
    MAX_FORECAST_DAYS,
    MAX_HISTORICAL_DAYS,
    OPEN_METEO_ARCHIVE_URL,
    OPEN_METEO_FORECAST_URL,
    TOMTOM_API_KEY,
    TOMTOM_FLOW_URL,
    TRAFFIC_TTL_SECONDS,
    WEATHER_TTL_SECONDS,
)
from database import get_cache, set_cache
from model_utils import traffic_status_from_ratio


class DataSourceError(RuntimeError):
    """A user-safe external data-source error."""


def get_city_coordinates(city):
    if city not in CITY_INFO:
        raise DataSourceError("Please choose a supported city.")
    return dict(CITY_INFO[city])


def weather_condition_from_code(code):
    try:
        code = int(code)
    except (TypeError, ValueError):
        return "Unknown"
    if code in {0, 1}:
        return "Clear"
    if code in {2, 3}:
        return "Cloudy"
    if code in {45, 48}:
        return "Foggy"
    if 51 <= code <= 67 or 80 <= code <= 82 or 95 <= code <= 99:
        return "Rainy"
    if code in {71, 73, 75, 77, 85, 86}:
        return "Other"
    return "Unknown"


def _request_json(base_url, parameters, source, timeout=12):
    url = base_url + "?" + urlencode(parameters)
    request = Request(url, headers={"User-Agent": "EVChargingForecastStudentProject/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        if error.code == 429:
            message = "The public service rate limit was reached. Try again later."
        elif error.code in {401, 403}:
            message = "The data-source credential is missing or invalid."
        else:
            message = f"The {source} service returned HTTP {error.code}."
        raise DataSourceError(message) from None
    except (URLError, TimeoutError, OSError, json.JSONDecodeError):
        raise DataSourceError(f"{source} is currently unavailable. Check your internet connection and try again.") from None


def _cached_request(table, city, cache_key, source, ttl_seconds, fetcher, force=False):
    cached = None if force else get_cache(table, city, cache_key)
    if cached:
        return {
            **cached["payload"],
            "_meta": {
                "source": cached["source"],
                "retrieved_at": cached["retrieved_at"],
                "cache_status": "cached",
                "stale": False,
            },
        }
    try:
        payload = fetcher()
        saved = set_cache(table, city, cache_key, source, payload, ttl_seconds)
        return {
            **payload,
            "_meta": {
                "source": source,
                "retrieved_at": saved["retrieved_at"],
                "cache_status": "fresh",
                "stale": False,
            },
        }
    except DataSourceError as error:
        stale = get_cache(table, city, cache_key, allow_expired=True)
        if stale:
            return {
                **stale["payload"],
                "_meta": {
                    "source": f"{stale['source']} (cached; refresh failed)",
                    "retrieved_at": stale["retrieved_at"],
                    "cache_status": "stale",
                    "stale": True,
                    "warning": str(error),
                },
            }
        raise


def _weather_parameters(city, hourly=False):
    location = get_city_coordinates(city)
    parameters = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "timezone": location["timezone"],
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
    }
    if hourly:
        parameters["hourly"] = "temperature_2m,relative_humidity_2m,rain,weather_code"
    return parameters


def _normalize_weather_row(hour, temperature, humidity, rain, weather_code):
    def number(value):
        try:
            result = float(value)
            return result if math.isfinite(result) else None
        except (TypeError, ValueError):
            return None

    return {
        "hour": int(hour),
        "temperature": number(temperature),
        "rainfall": number(rain),
        "weather_condition": weather_condition_from_code(weather_code),
        "weather_code": weather_code,
        "relative_humidity": number(humidity),
    }


def get_current_weather(city, force=False):
    def fetch():
        parameters = _weather_parameters(city)
        parameters["current"] = "temperature_2m,relative_humidity_2m,precipitation,rain,weather_code"
        result = _request_json(OPEN_METEO_FORECAST_URL, parameters, "Open-Meteo current weather")
        current = result.get("current")
        if not current or current.get("time") is None:
            raise DataSourceError("Open-Meteo did not return current weather for this city.")
        parsed = _normalize_weather_row(
            datetime.fromisoformat(current["time"]).hour,
            current.get("temperature_2m"),
            current.get("relative_humidity_2m"),
            current.get("rain", current.get("precipitation")),
            current.get("weather_code"),
        )
        parsed["weather_time"] = current["time"]
        parsed["timezone"] = result.get("timezone", CITY_INFO[city]["timezone"])
        if parsed["temperature"] is None or parsed["weather_condition"] == "Unknown":
            raise DataSourceError("Current weather is incomplete; no weather-based prediction was made.")
        return parsed

    return _cached_request(
        "weather_cache", city, "current", "Open-Meteo current weather",
        WEATHER_TTL_SECONDS, fetch, force=force,
    )


def _fetch_historical_range(city, start_date, end_date, force=False):
    cache_key = f"history:{start_date}:{end_date}"

    def fetch():
        parameters = _weather_parameters(city, hourly=True)
        parameters.update({"start_date": start_date, "end_date": end_date})
        result = _request_json(OPEN_METEO_ARCHIVE_URL, parameters, "Open-Meteo Historical Weather API")
        hourly = result.get("hourly", {})
        times = hourly.get("time", [])
        if not times:
            raise DataSourceError("Open-Meteo returned no hourly records for that date range.")
        records = []
        for index, text_time in enumerate(times):
            parsed_time = datetime.fromisoformat(text_time)
            weather = _normalize_weather_row(
                parsed_time.hour,
                _at(hourly, "temperature_2m", index),
                _at(hourly, "relative_humidity_2m", index),
                _at(hourly, "rain", index),
                _at(hourly, "weather_code", index),
            )
            weather.update({
                "city": city,
                "local_time": text_time,
                "date": parsed_time.date().isoformat(),
                "day_of_week": parsed_time.weekday(),
                "traffic_status": "unknown",
            })
            records.append(weather)
        return {"records": records}

    return _cached_request(
        "weather_cache", city, cache_key,
        "Open-Meteo historical weather reanalysis",
        HISTORICAL_TTL_SECONDS, fetch, force=force,
    )


def _at(hourly, key, index):
    values = hourly.get(key, [])
    return values[index] if index < len(values) else None


def get_historical_weather(city, start_date, end_date, force=False):
    try:
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
    except ValueError:
        raise DataSourceError("Enter valid start and end dates.") from None
    if end < start:
        raise DataSourceError("End date must be on or after the start date.")
    if (end - start).days + 1 > MAX_HISTORICAL_DAYS:
        raise DataSourceError(
            f"Choose a date range of {MAX_HISTORICAL_DAYS} days or fewer for one analysis."
        )
    return _fetch_historical_range(city, start.isoformat(), end.isoformat(), force=force)


def _forecast_window(city, force=False):
    def fetch():
        parameters = _weather_parameters(city, hourly=True)
        parameters["forecast_days"] = MAX_FORECAST_DAYS
        result = _request_json(OPEN_METEO_FORECAST_URL, parameters, "Open-Meteo Weather Forecast API")
        hourly = result.get("hourly", {})
        if not hourly.get("time"):
            raise DataSourceError("Open-Meteo returned no hourly forecast.")
        return {"hourly": hourly, "timezone": result.get("timezone", CITY_INFO[city]["timezone"])}

    return _cached_request(
        "weather_cache", city, "forecast_window",
        "Open-Meteo hourly forecast (up to 16 days)",
        FORECAST_TTL_SECONDS, fetch, force=force,
    )


def get_weather_forecast(city, target_date, target_hour, force=False):
    location = get_city_coordinates(city)
    try:
        target = date.fromisoformat(str(target_date))
        target_hour = int(target_hour)
    except (TypeError, ValueError):
        raise DataSourceError("Enter a valid future date and hour.") from None
    if not 0 <= target_hour <= 23:
        raise DataSourceError("Forecast hour must be between 0 and 23.")
    today = datetime.now(ZoneInfo(location["timezone"])).date()
    if target < today:
        raise DataSourceError("Future Forecast accepts today or a later date.")

    forecast_note = None
    forecast = None
    if (target - today).days < MAX_FORECAST_DAYS:
        try:
            forecast = _forecast_window(city, force=force)
        except DataSourceError:
            forecast = None
            forecast_note = "The forecast service was unavailable; a historical weather analogue was attempted."

    if forecast:
        hourly = forecast["hourly"]
        wanted = f"{target.isoformat()}T{target_hour:02d}:00"
        try:
            index = hourly["time"].index(wanted)
        except ValueError:
            forecast = None
            forecast_note = "The selected hour is beyond the forecast service's returned horizon."

    if forecast:
        weather = _normalize_weather_row(
            target_hour,
            _at(hourly, "temperature_2m", index),
            _at(hourly, "relative_humidity_2m", index),
            _at(hourly, "rain", index),
            _at(hourly, "weather_code", index),
        )
        weather.update({
            "source": forecast["_meta"]["source"],
            "retrieved_at": forecast["_meta"]["retrieved_at"],
            "data_kind": "forecast",
            "notice": forecast_note,
            "stale": forecast["_meta"]["stale"],
        })
        return weather

    analogue = _weather_analogue(city, target, target_hour)
    if (target - today).days >= MAX_FORECAST_DAYS:
        analogue["notice"] = (
            "This date is outside the available forecast horizon. Weather uses a historical same-date/hour "
            "analogue and is not a forecast."
        )
    else:
        analogue["notice"] = (
            "A live forecast was unavailable for this date. Weather uses a historical same-date/hour "
            "analogue and is not a forecast."
        )
    if forecast_note:
        analogue["notice"] += " " + forecast_note
    return analogue


def _weather_analogue(city, target, target_hour):
    rows = []
    for year_offset in range(1, 6):
        prior_year = target.year - year_offset
        try:
            analogue_date = target.replace(year=prior_year)
        except ValueError:
            analogue_date = target.replace(year=prior_year, day=28)
        if analogue_date >= date.today():
            continue
        try:
            result = _fetch_historical_range(
                city, analogue_date.isoformat(), analogue_date.isoformat()
            )
            records = result["records"]
            row = next((item for item in records if item["hour"] == target_hour), None)
            if row:
                rows.append(row)
        except DataSourceError:
            continue
    if not rows:
        raise DataSourceError(
            "No live forecast is available for that date, and historical weather analogues could not be retrieved."
        )
    numeric = {}
    for key in ("temperature", "rainfall", "relative_humidity"):
        values = [row[key] for row in rows if row.get(key) is not None]
        numeric[key] = sum(values) / len(values) if values else None
    conditions = [row["weather_condition"] for row in rows if row["weather_condition"] != "Unknown"]
    numeric["weather_condition"] = Counter(conditions).most_common(1)[0][0] if conditions else "Unknown"
    numeric.update({
        "hour": target_hour,
        "weather_code": None,
        "source": "Open-Meteo historical weather (five prior same-date/hour records; analogue, not forecast)",
        "retrieved_at": datetime.now().astimezone().isoformat(timespec="minutes"),
        "data_kind": "historical_analogue",
        "stale": False,
    })
    return numeric


def get_traffic_data(city, force=False):
    if not TOMTOM_API_KEY:
        return {
            "traffic_status": "unknown",
            "current_speed": None,
            "free_flow_speed": None,
            "speed_ratio": None,
            "confidence": None,
            "source": "Unavailable: optional TomTom key is not configured",
            "retrieved_at": None,
            "data_kind": "unavailable",
            "note": "No live traffic value was substituted. Configure TOMTOM_API_KEY to query the nearest road segment.",
        }

    location = get_city_coordinates(city)
    point = f"{location['latitude']},{location['longitude']}"

    def fetch():
        result = _request_json(
            TOMTOM_FLOW_URL,
            {"point": point, "key": TOMTOM_API_KEY},
            "TomTom Traffic Flow API",
        )
        flow = result.get("flowSegmentData")
        if not flow:
            raise DataSourceError("TomTom returned no road-flow segment for this city reference point.")
        current_speed = _safe_float(flow.get("currentSpeed"))
        free_flow_speed = _safe_float(flow.get("freeFlowSpeed"))
        confidence = _safe_float(flow.get("confidence"))
        if current_speed is None or not free_flow_speed:
            raise DataSourceError("TomTom returned incomplete speed data.")
        ratio = current_speed / free_flow_speed
        status = traffic_status_from_ratio(ratio)
        if confidence is not None and confidence < 0.25:
            status = "unknown"
        return {
            "traffic_status": status,
            "current_speed": current_speed,
            "free_flow_speed": free_flow_speed,
            "speed_ratio": ratio,
            "confidence": confidence,
            "road_closure": bool(flow.get("roadClosure", False)),
        }

    try:
        result = _cached_request(
            "traffic_cache", city, "current",
            "TomTom Traffic Flow (nearest road segment)",
            TRAFFIC_TTL_SECONDS, fetch, force=force,
        )
        result.update({
            "data_kind": (
                "stale" if result["_meta"]["cache_status"] == "stale" else
                ("cached" if result["_meta"]["cache_status"] == "cached" else "live")
            ),
            "source": result["_meta"]["source"],
            "retrieved_at": result["_meta"]["retrieved_at"],
            "note": "Speed and congestion represent the nearest road segment to a city reference point, not citywide traffic volume.",
        })
        if result["_meta"].get("warning"):
            result["note"] += " " + result["_meta"]["warning"]
        return result
    except DataSourceError as error:
        stale = get_cache("traffic_cache", city, "current", allow_expired=True)
        if stale:
            payload = stale["payload"]
            payload.update({
                "data_kind": "stale",
                "source": f"{stale['source']} (cached; refresh failed)",
                "retrieved_at": stale["retrieved_at"],
                "note": f"Using the last cached segment reading. {error}",
            })
            return payload
        return {
            "traffic_status": "unknown",
            "current_speed": None,
            "free_flow_speed": None,
            "speed_ratio": None,
            "confidence": None,
            "source": "TomTom Traffic Flow unavailable",
            "retrieved_at": None,
            "data_kind": "unavailable",
            "note": str(error),
        }


def get_historical_traffic(city, start_date, end_date):
    return {
        "traffic_status": "unknown",
        "source": "Unavailable: no free public historical city traffic feed is configured",
        "data_kind": "unavailable",
        "note": "TomTom's public Flow Segment endpoint provides current road-segment flow; its traffic statistics products are not enabled in this student project.",
    }


def _safe_float(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None
