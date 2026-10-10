"""Flask application for live, historical, and future model estimates."""

from datetime import date, datetime, time
import base64
import json
from pathlib import Path
import sqlite3
import uuid
from zoneinfo import ZoneInfo

import pandas as pd
from flask import Flask, make_response, redirect, render_template, request, url_for
from joblib import load

from config import (
    CITIES,
    CITY_INFO,
    DATA_PATH,
    DATABASE_PATH,
    IMPORTANCE_PATH,
    METRICS_PATH,
    MODEL_PATH,
    TRAINING_INSIGHTS_PATH,
)
from data_sources import (
    DataSourceError,
    get_current_weather,
    get_historical_traffic,
    get_historical_weather,
    get_traffic_data,
    get_weather_forecast,
)
from database import connect, initialize_database
from model_utils import INPUT_COLUMNS, make_feature_row, peak_period_for_hour, prepare_model_features
from report_generator import analyze_records, create_historical_charts

WEATHER_LABEL = "Open-Meteo weather model"
MODEL_LABEL = "Random Forest trained on synthetic data"


def create_app(config=None):
    flask_app = Flask(__name__)
    flask_app.config.update(
        DATABASE=str(DATABASE_PATH),
        MODEL_PATH=str(MODEL_PATH),
        METRICS_PATH=str(METRICS_PATH),
        IMPORTANCE_PATH=str(IMPORTANCE_PATH),
        TRAINING_INSIGHTS_PATH=str(TRAINING_INSIGHTS_PATH),
    )
    if config:
        flask_app.config.update(config)
    try:
        initialize_database(flask_app.config["DATABASE"])
        flask_app.config["DATABASE_ERROR"] = None
    except sqlite3.Error:
        flask_app.config["DATABASE_ERROR"] = (
            "The local database could not be opened. Check its file permissions and available disk space."
        )

    try:
        flask_app.config["MODEL"] = load(flask_app.config["MODEL_PATH"])
        flask_app.config["MODEL_ERROR"] = None
    except Exception:
        flask_app.config["MODEL"] = None
        flask_app.config["MODEL_ERROR"] = (
            "The saved model could not be loaded. Run python train_model.py to rebuild it."
        )

    @flask_app.route("/")
    def index():
        metrics = _read_json(flask_app.config["METRICS_PATH"], {})
        return render_template(
            "index.html",
            cities=CITIES,
            metrics=metrics,
            source_cards=[
                {"name": "Weather", "source": "Open-Meteo current and historical weather APIs"},
                {"name": "Live traffic", "source": "Optional TomTom nearest-road flow; key required"},
                {"name": "Historical traffic", "source": "Unavailable from a free historic feed"},
                {"name": "Training labels", "source": "Clearly synthetic, generated CSV"},
                {"name": "Charging demand", "source": "Model estimate, not measured sessions"},
            ],
        )

    @flask_app.route("/live", methods=["GET", "POST"])
    def live_prediction():
        context = _page_defaults()
        if request.method == "POST":
            city = request.form.get("city", "")
            context["selected_city"] = city
            try:
                _validate_city(city)
                local_now = datetime.now(ZoneInfo(CITY_INFO[city]["timezone"]))
                weather = get_current_weather(city)
                traffic = get_traffic_data(city)
                feature_row = make_feature_row(city, local_now, weather, traffic["traffic_status"])
                prediction = _predict(flask_app, feature_row)
                sources = _live_sources(city, local_now, weather, traffic)
                prediction_id = _save_prediction(
                    flask_app, city, "live", None, feature_row, sources, prediction
                )
                context.update({
                    "weather": weather,
                    "traffic": traffic,
                    "prediction": prediction,
                    "prediction_id": prediction_id,
                    "local_now": local_now,
                    "peak_period": peak_period_for_hour(local_now.hour),
                    "sources": sources,
                })
            except DataSourceError as error:
                context["error"] = str(error)
            except sqlite3.Error:
                context["error"] = "The local database is unavailable; no prediction was saved."
            except RuntimeError as error:
                context["error"] = str(error)
        return render_template("live_prediction.html", cities=CITIES, **context)

    @flask_app.route("/historical", methods=["GET", "POST"])
    def historical_analysis():
        context = _page_defaults()
        context["today"] = datetime.now().date().isoformat()
        if request.method == "POST":
            city = request.form.get("city", "")
            start_text = request.form.get("start_date", "")
            end_text = request.form.get("end_date", "")
            context.update({"selected_city": city, "start_date": start_text, "end_date": end_text})
            try:
                _validate_city(city)
                start = date.fromisoformat(start_text)
                end = date.fromisoformat(end_text)
                if end > datetime.now(ZoneInfo(CITY_INFO[city]["timezone"])).date():
                    raise DataSourceError("Historical Analysis dates must be today or earlier.")
                weather_result = get_historical_weather(city, start_text, end_text)
                historical_traffic = get_historical_traffic(city, start_text, end_text)
                if flask_app.config["MODEL"] is None:
                    raise RuntimeError(flask_app.config["MODEL_ERROR"])

                records = weather_result["records"]
                if not records:
                    raise DataSourceError("No historical weather rows were returned.")
                zone = ZoneInfo(CITY_INFO[city]["timezone"])
                for row in records:
                    local_time = datetime.fromisoformat(row["local_time"]).replace(tzinfo=zone)
                    traffic_status = historical_traffic["traffic_status"]
                    model_input = make_feature_row(city, local_time, row, traffic_status)
                    row["traffic_status"] = traffic_status
                    row["predicted_demand"] = _predict(flask_app, model_input)
                    row["day_type"] = "Weekend" if local_time.weekday() >= 5 else "Weekday"

                summary, observations = analyze_records(records)
                prefix = "history_" + uuid.uuid4().hex[:10]
                charts = create_historical_charts(records, prefix)
                sources = [
                    _source_row(
                        "Historical weather",
                        weather_result["_meta"]["source"],
                        weather_result["_meta"]["retrieved_at"],
                        "Cached" if weather_result["_meta"]["stale"] else "Retrieved",
                    ),
                    _source_row(
                        "Historical traffic",
                        historical_traffic["source"],
                        None,
                        "Unavailable",
                        historical_traffic.get("note"),
                    ),
                    _source_row("Charging sessions", MODEL_LABEL, None, "Model estimate"),
                ]
                created_at = datetime.now().astimezone().isoformat(timespec="seconds")
                with connect(flask_app.config["DATABASE"]) as connection:
                    cursor = connection.execute(
                        """
                        INSERT INTO historical_analysis (
                            city, start_date, end_date, created_at, summary_json,
                            rows_json, sources_json, chart_prefix
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            city, start_text, end_text, created_at,
                            json.dumps(summary), json.dumps(records),
                            json.dumps(sources), prefix,
                        ),
                    )
                    analysis_id = cursor.lastrowid
                    saved_at = datetime.now().astimezone().isoformat(timespec="seconds")
                    connection.executemany(
                        """
                        INSERT INTO predictions (
                            timestamp, city, prediction_mode, forecast_for,
                            input_data_json, sources_json, predicted_demand,
                            traffic_status, temperature, rainfall, relative_humidity,
                            weather_condition, hour, day_of_week, model_version
                        ) VALUES (?, ?, 'historical', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            (
                                saved_at, city, row["local_time"],
                                json.dumps({key: row.get(key) for key in INPUT_COLUMNS}),
                                json.dumps(sources), row["predicted_demand"],
                                row.get("traffic_status"), row.get("temperature"),
                                row.get("rainfall"), row.get("relative_humidity"),
                                row.get("weather_condition"), row.get("hour"),
                                row.get("day_of_week"),
                                "RandomForestRegressor/synthetic-training-v2",
                            )
                            for row in records
                        ],
                    )

                context.update({
                    "analysis_id": analysis_id,
                    "records": records,
                    "summary": summary,
                    "observations": observations,
                    "charts": charts,
                    "sources": sources,
                    "historical_traffic": historical_traffic,
                })
            except (ValueError, DataSourceError) as error:
                context["error"] = str(error)
            except sqlite3.Error:
                context["error"] = "The local database is unavailable; the analysis could not be saved."
            except RuntimeError as error:
                context["error"] = str(error)
        return render_template("historical.html", cities=CITIES, **context)

    @flask_app.route("/future", methods=["GET", "POST"])
    def future_forecast():
        context = _page_defaults()
        context["today"] = datetime.now().date().isoformat()
        if request.method == "POST":
            city = request.form.get("city", "")
            date_text = request.form.get("future_date", "")
            time_text = request.form.get("future_time", "18:00")
            context.update({"selected_city": city, "future_date": date_text, "future_time": time_text})
            try:
                _validate_city(city)
                target_date = date.fromisoformat(date_text)
                target_time = time.fromisoformat(time_text)
                zone = ZoneInfo(CITY_INFO[city]["timezone"])
                target = datetime.combine(target_date, target_time).replace(tzinfo=zone)
                local_now = datetime.now(zone)
                if target <= local_now:
                    raise DataSourceError("Choose a date and time later than the current city time.")
                weather = get_weather_forecast(city, date_text, target.hour)
                traffic = {
                    "traffic_status": "unknown",
                    "source": "Unavailable: no public future traffic forecast configured",
                    "data_kind": "unavailable",
                    "retrieved_at": None,
                    "note": "Live traffic is not used as a substitute for future traffic.",
                }
                feature_row = make_feature_row(city, target, weather, "unknown")
                prediction = _predict(flask_app, feature_row)
                sources = [
                    _source_row(
                        "Weather",
                        weather["source"],
                        weather.get("retrieved_at"),
                        "Historical analogue (not forecast)" if weather["data_kind"] == "historical_analogue"
                        else ("Cached" if weather.get("stale") else "Forecast"),
                        weather.get("notice"),
                    ),
                    _source_row("Traffic", traffic["source"], None, "Unavailable", traffic["note"]),
                    _source_row("Time and peak period", "City timezone and calendar", None, "Derived"),
                    _source_row("Charging demand", MODEL_LABEL, None, "Model estimate"),
                ]
                prediction_id = _save_prediction(
                    flask_app, city, "future", target.isoformat(), feature_row, sources, prediction
                )
                context.update({
                    "weather": weather,
                    "traffic": traffic,
                    "prediction": prediction,
                    "prediction_id": prediction_id,
                    "target": target,
                    "peak_period": peak_period_for_hour(target.hour),
                    "sources": sources,
                })
            except (ValueError, DataSourceError) as error:
                context["error"] = str(error)
            except sqlite3.Error:
                context["error"] = "The local database is unavailable; no prediction was saved."
            except RuntimeError as error:
                context["error"] = str(error)
        return render_template("future_forecast.html", cities=CITIES, **context)

    @flask_app.route("/reports/<int:analysis_id>")
    def report(analysis_id):
        try:
            with connect(flask_app.config["DATABASE"]) as connection:
                result = connection.execute(
                    "SELECT * FROM historical_analysis WHERE id=?", (analysis_id,)
                ).fetchone()
            if result is None:
                return render_template("error.html", message="That saved analysis report was not found."), 404
            data = dict(result)
            summary = json.loads(data["summary_json"])
            records = json.loads(data["rows_json"])
            sources = json.loads(data["sources_json"])
            _, observations = analyze_records(records)
            metrics = _read_json(flask_app.config["METRICS_PATH"], {})
            importance = _read_json(flask_app.config["IMPORTANCE_PATH"], {})
            download_mode = request.args.get("download") == "1"
            chart_names = [
                "demand_by_hour", "demand_over_time", "traffic_vs_demand",
                "temperature_vs_demand", "weather_vs_demand", "weekday_weekend",
            ]
            chart_images = {}
            embedded_css = None
            if download_mode:
                from pathlib import Path
                image_dir = Path(flask_app.static_folder) / "images"
                for name in chart_names:
                    image_path = image_dir / f"{data['chart_prefix']}_{name}.png"
                    try:
                        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
                        chart_images[name] = f"data:image/png;base64,{encoded}"
                    except OSError:
                        chart_images[name] = ""
                try:
                    embedded_css = (Path(flask_app.static_folder) / "css" / "style.css").read_text(encoding="utf-8")
                except OSError:
                    embedded_css = "body{font-family:Arial,sans-serif;margin:2rem;color:#20342f}"
            html = render_template(
                "report.html",
                analysis=data,
                summary=summary,
                records=records,
                sources=sources,
                observations=observations,
                metrics=metrics,
                importance=importance,
                download_mode=download_mode,
                chart_images=chart_images,
                embedded_css=embedded_css,
            )
            if request.args.get("download") == "1":
                response = make_response(html)
                response.headers["Content-Disposition"] = (
                    f'attachment; filename="ev-demand-analysis-{analysis_id}.html"'
                )
                return response
            return html
        except sqlite3.Error:
            return render_template("error.html", message="The local database is unavailable."), 503

    @flask_app.route("/dashboard")
    def dashboard():
        try:
            with connect(flask_app.config["DATABASE"]) as connection:
                summary = connection.execute(
                    """
                    SELECT COUNT(*) AS total,
                           COALESCE(AVG(predicted_demand),0) AS average,
                           COALESCE(MAX(predicted_demand),0) AS highest,
                           COALESCE(MIN(predicted_demand),0) AS lowest,
                           COUNT(DISTINCT city) AS cities,
                           SUM(CASE WHEN prediction_mode='live' THEN 1 ELSE 0 END) AS live_count,
                           SUM(CASE WHEN prediction_mode='historical' THEN 1 ELSE 0 END) AS historical_count,
                           SUM(CASE WHEN prediction_mode='future' THEN 1 ELSE 0 END) AS future_count
                    FROM predictions
                    """
                ).fetchone()
                mode_counts = connection.execute(
                    "SELECT prediction_mode, COUNT(*) AS total FROM predictions GROUP BY prediction_mode"
                ).fetchall()
                recent = connection.execute(
                    "SELECT * FROM predictions ORDER BY id DESC LIMIT 8"
                ).fetchall()
            mode_chart = _mode_distribution_chart(mode_counts)
            return render_template(
                "dashboard.html",
                summary=dict(summary),
                recent=recent,
                mode_chart=mode_chart,
                metrics=_read_json(flask_app.config["METRICS_PATH"], {}),
                importance=_read_json(flask_app.config["IMPORTANCE_PATH"], {}),
                insights=_read_json(flask_app.config["TRAINING_INSIGHTS_PATH"], {"observations": []}),
            )
        except sqlite3.Error:
            return render_template("error.html", message="The local database is unavailable."), 503

    @flask_app.route("/history")
    def history():
        try:
            with connect(flask_app.config["DATABASE"]) as connection:
                predictions = connection.execute(
                    "SELECT * FROM predictions ORDER BY id DESC LIMIT 250"
                ).fetchall()
                analyses = connection.execute(
                    """
                    SELECT id, city, start_date, end_date, created_at
                    FROM historical_analysis ORDER BY id DESC LIMIT 50
                    """
                ).fetchall()
            return render_template("history.html", predictions=predictions, analyses=analyses)
        except sqlite3.Error:
            return render_template("error.html", message="The local database is unavailable."), 503

    @flask_app.route("/health")
    def health():
        ready = flask_app.config["MODEL"] is not None
        return {"status": "ok" if ready else "model_unavailable", "model_loaded": ready}, (200 if ready else 503)

    return flask_app


def _page_defaults():
    return {
        "error": None,
        "selected_city": "Bengaluru",
        "weather": None,
        "traffic": None,
        "prediction": None,
        "prediction_id": None,
        "sources": [],
        "peak_period": None,
    }


def _validate_city(city):
    if city not in CITIES:
        raise DataSourceError("Choose a supported city from the list.")


def _predict(flask_app, feature_row):
    model = flask_app.config["MODEL"]
    if model is None:
        raise RuntimeError(flask_app.config["MODEL_ERROR"])
    features = prepare_model_features(feature_row)
    return max(0, int(round(float(model.predict(features)[0]))))


def _source_row(name, source, retrieved_at, status, note=None):
    return {
        "name": name,
        "source": source,
        "retrieved_at": retrieved_at,
        "status": status,
        "note": note,
    }


def _live_sources(city, local_now, weather, traffic):
    weather_meta = weather.get("_meta", {})
    return [
        _source_row(
            "Weather",
            weather_meta.get("source", WEATHER_LABEL),
            weather_meta.get("retrieved_at"),
            "Cached / stale" if weather_meta.get("stale") else (
                "Cached" if weather_meta.get("cache_status") == "cached" else "Live"
            ),
            weather_meta.get("warning"),
        ),
        _source_row(
            "Traffic indicator",
            traffic["source"],
            traffic.get("retrieved_at"),
            "Unavailable" if traffic["data_kind"] == "unavailable" else (
                "Cached" if traffic["data_kind"] in {"cached", "stale"} else "Live"
            ),
            traffic.get("note"),
        ),
        _source_row("City coordinates and timezone", "Project city reference list", None, "Configured"),
        _source_row("Hour, weekday, peak period", "Derived from current city-local time", None, "Derived"),
        _source_row("EV count / actual charging sessions", "No reliable public hourly feed configured", None, "Not used"),
        _source_row("Charging demand", MODEL_LABEL, None, "Model estimate"),
    ]


def _save_prediction(flask_app, city, mode, forecast_for, feature_row, sources, prediction):
    if flask_app.config["MODEL"] is None:
        raise RuntimeError(flask_app.config["MODEL_ERROR"])
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    with connect(flask_app.config["DATABASE"]) as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions (
                timestamp, city, prediction_mode, forecast_for,
                input_data_json, sources_json, predicted_demand,
                traffic_status, temperature, rainfall, relative_humidity,
                weather_condition, hour, day_of_week, model_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now, city, mode, forecast_for, json.dumps(feature_row),
                json.dumps(sources), prediction, feature_row.get("traffic_status"),
                feature_row.get("temperature"), feature_row.get("rainfall"),
                feature_row.get("relative_humidity"), feature_row.get("weather_condition"),
                feature_row.get("hour"), feature_row.get("day_of_week"),
                "RandomForestRegressor/synthetic-training-v2",
            ),
        )
        return cursor.lastrowid


def _read_json(path, fallback):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return fallback


def _mode_distribution_chart(mode_counts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    from config import BASE_DIR

    path = BASE_DIR / "static" / "images" / "prediction_modes.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [row["prediction_mode"].title() for row in mode_counts]
    counts = [row["total"] for row in mode_counts]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    if counts:
        ax.bar(names, counts, color="#2b7864")
        ax.set(ylabel="Saved records", title="Prediction Mode Distribution")
    else:
        ax.text(0.5, 0.5, "No saved predictions yet", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return "images/prediction_modes.png"


app = create_app()


if __name__ == "__main__":
    app.run(debug=False)
