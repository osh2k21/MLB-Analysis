from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import streamlit as st

from mlb_predictor.api import HttpClient, MLBStatsClient, OddsClient, StructuredFeedClient, WeatherClient
from mlb_predictor.config import load_settings
from mlb_predictor.contracts import DataStatus, Evidence
from mlb_predictor.database import PredictionRepository
from mlb_predictor.dashboard.views import validation_rows
from mlb_predictor.features import FEATURE_NAMES, FeatureCollector, FeatureError, scope_game_evidence
from mlb_predictor.pipeline import PredictionPipeline, PredictionWithheld, load_bundle
from mlb_predictor.validation import GameValidator


st.set_page_config(page_title="Verified MLB Predictor", page_icon="⚾", layout="wide")
st.title("Verified MLB Predictor")
st.caption("Fail-closed data validation · 143 sourced features · five-model calibrated ensemble")

settings = load_settings()
http = HttpClient(settings.cache_path, settings.request_timeout_seconds, settings.max_attempts, settings.requests_per_second)
mlb = MLBStatsClient(http)
repository = PredictionRepository(settings.database_path)


@st.cache_resource
def model_bundle(path: str):
    return load_bundle(Path(path))


def structured(name: str, url: str | None) -> Evidence:
    return StructuredFeedClient(name, url, http).fetch()


def scoped_odds(all_odds: Evidence, away: str, home: str) -> Evidence:
    if not all_odds.usable or not isinstance(all_odds.payload, list):
        return all_odds
    match = next((item for item in all_odds.payload if item.get("away_team") == away and item.get("home_team") == home), None)
    if not match:
        return Evidence(all_odds.source, DataStatus.MISSING, all_odds.observed_at, detail="game absent from odds response")
    books = match.get("bookmakers") or []
    market = next((market for book in books for market in book.get("markets", []) if market.get("key") == "h2h"), None)
    outcomes = market.get("outcomes", []) if market else []
    payload = {"home": next((x for x in outcomes if x.get("name") == home), None), "away": next((x for x in outcomes if x.get("name") == away), None)}
    observed = all_odds.observed_at
    stamps = [book.get("last_update") for book in books if book.get("last_update")]
    if stamps:
        try:
            observed = max(datetime.fromisoformat(stamp.replace("Z", "+00:00")) for stamp in stamps)
        except ValueError:
            return Evidence(all_odds.source, DataStatus.INVALID, all_odds.observed_at, detail="odds timestamp invalid")
    return Evidence(all_odds.source, DataStatus.OK, observed, payload, fetched_at=all_odds.fetched_at, cache_hit=all_odds.cache_hit)


selected_date = st.sidebar.date_input("Slate date", value=date.today()).isoformat()
refresh = st.sidebar.button("Refresh verified data")
if refresh:
    st.cache_data.clear()

schedule = mlb.schedule(selected_date)
if not schedule.usable:
    st.error(f"Schedule unavailable: {schedule.detail}")
    st.stop()

odds_all = OddsClient(http, settings.odds_api_key).games()
feeds = {
    "statcast": structured("Baseball Savant / Statcast", settings.statcast_feed_url),
    "fangraphs": structured("FanGraphs", settings.fangraphs_feed_url),
    "retrosheet": structured("Retrosheet", settings.retrosheet_feed_url),
    "injuries": structured("Injury feed", settings.injury_feed_url),
    "umpire": structured("Umpire assignments", settings.umpire_feed_url),
    "park": structured("Park factors", settings.park_factor_feed_url),
}

try:
    bundle = model_bundle(str(settings.model_path))
    model_error = None
except Exception as exc:
    bundle, model_error = None, str(exc)

if model_error:
    st.warning(f"No production prediction model: {model_error}. Train the ensemble before probabilities can be displayed.")

games = schedule.payload
if not games:
    st.info("No MLB games on this date.")

validator = GameValidator()
collector = FeatureCollector()
for raw in games:
    game = mlb.game_ref(raw)
    with st.container(border=True):
        st.subheader(f"{game.away_team} at {game.home_team}")
        mlb_feed = mlb.game_feed(game.game_pk)
        coordinates = {}
        if mlb_feed.usable:
            coordinates = mlb_feed.payload.get("gameData", {}).get("venue", {}).get("location", {}).get("defaultCoordinates", {})
        weather = WeatherClient(http, settings.weather_api_key, settings.weather_base_url).forecast(
            coordinates.get("latitude"), coordinates.get("longitude"), game.commence_time
        )
        evidence = {
            "mlb_feed": mlb_feed,
            "bullpen": scope_game_evidence(feeds["retrosheet"], game.game_pk, "bullpen"),
            "weather": weather,
            "odds": scoped_odds(odds_all, game.away_team, game.home_team),
            "statcast": scope_game_evidence(feeds["statcast"], game.game_pk, "statcast"),
            "injuries": scope_game_evidence(feeds["injuries"], game.game_pk, "injuries"),
            "umpire": scope_game_evidence(feeds["umpire"], game.game_pk, "umpire"),
        }
        report = validator.validate(game, evidence)
        st.dataframe(validation_rows(report), hide_index=True, use_container_width=True)
        feature_error = None
        snapshot = None
        try:
            snapshot = collector.collect(game, [mlb_feed, odds_all, weather, *feeds.values()])
        except Exception as exc:
            feature_error = str(exc)

        if not report.ready or feature_error or bundle is None:
            st.error("Prediction Withheld")
            if feature_error:
                st.caption(f"Feature contract incomplete: {feature_error}")
            continue

        try:
            result = PredictionPipeline(bundle, repository, validator).predict(game, evidence, snapshot)
        except PredictionWithheld as exc:
            st.error(str(exc))
            continue
        probability = result.prediction.home_probability
        pick = game.home_team if probability >= 0.5 else game.away_team
        pick_probability = max(probability, 1 - probability)
        st.metric("Calibrated prediction", pick, f"{pick_probability:.1%}")
        st.write({name.title(): f"{points:+.1f}%" for name, points in result.explanation.category_points.items()})
        st.caption("Model votes: " + " · ".join(f"{v.model} {v.home_probability:.1%}" for v in result.prediction.votes))
