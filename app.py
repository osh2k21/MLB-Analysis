from __future__ import annotations

from datetime import date
from pathlib import Path
import math

import streamlit as st

from mlb_predictor.api import HttpClient, MLBStatsClient, OddsClient, WeatherClient
from mlb_predictor.config import load_settings
from mlb_predictor.contracts import DataStatus, Evidence
from mlb_predictor.explainability import explain_prediction
from mlb_predictor.features.catalog import FEATURE_NAMES
from mlb_predictor.features.free_live import FreeLiveFeatureBuilder
from mlb_predictor.pipeline import load_bundle


st.set_page_config(page_title="Free MLB Predictor", page_icon="⚾", layout="wide")
st.title("Free MLB Predictor")
st.caption("120 reproducible free-data features · chronological five-model ensemble · no paid stat feeds")


@st.cache_resource
def services():
    settings = load_settings()
    http = HttpClient(settings.cache_path, settings.request_timeout_seconds, settings.max_attempts, settings.requests_per_second)
    return settings, http, MLBStatsClient(http)


@st.cache_resource
def fitted_bundle(path: str):
    return load_bundle(Path(path))


def american_probability(odds):
    value = float(odds)
    return -value / (-value + 100) if value < 0 else 100 / (value + 100)


def game_odds(evidence: Evidence, away: str, home: str):
    if not evidence.usable:
        return None
    event = next((x for x in evidence.payload if x.get("away_team") == away and x.get("home_team") == home), None)
    if not event:
        return None
    books = event.get("bookmakers") or []
    prices = []
    for book in books:
        market = next((x for x in book.get("markets", []) if x.get("key") == "h2h"), None)
        if not market:
            continue
        outcomes = market.get("outcomes", [])
        home_price = next((x.get("price") for x in outcomes if x.get("name") == home), None)
        away_price = next((x.get("price") for x in outcomes if x.get("name") == away), None)
        if home_price is not None and away_price is not None:
            hp, ap = american_probability(home_price), american_probability(away_price)
            prices.append({"book": book.get("title", book.get("key", "book")), "home_odds": home_price, "away_odds": away_price, "home_no_vig": hp / (hp + ap)})
    if not prices:
        return None
    return {
        "home": {"price": prices[0]["home_odds"]}, "away": {"price": prices[0]["away_odds"]},
        "home_no_vig": sum(x["home_no_vig"] for x in prices) / len(prices), "books": prices,
    }


def lineup_ready(feed: Evidence) -> bool:
    if not feed.usable:
        return False
    teams = feed.payload.get("liveData", {}).get("boxscore", {}).get("teams", {})
    return len(teams.get("home", {}).get("battingOrder") or []) >= 9 and len(teams.get("away", {}).get("battingOrder") or []) >= 9


def starter_ready(raw: dict, feed: Evidence) -> bool:
    raw_teams = raw.get("teams", {})
    raw_ok = bool(raw_teams.get("home", {}).get("probablePitcher", {}).get("id") and raw_teams.get("away", {}).get("probablePitcher", {}).get("id"))
    probable = feed.payload.get("gameData", {}).get("probablePitchers", {}) if feed.usable else {}
    return raw_ok or bool(probable.get("home", {}).get("id") and probable.get("away", {}).get("id"))


settings, http, mlb = services()
selected_date = st.sidebar.date_input("Slate date", date.today()).isoformat()
if st.sidebar.button("Refresh data"):
    st.cache_data.clear()

try:
    bundle = fitted_bundle(str(settings.model_path))
except Exception as exc:
    st.error(f"Trained model unavailable: {exc}")
    st.stop()

metrics_path = settings.model_path.with_suffix(".metrics.json")
with st.sidebar.expander("Model audit"):
    st.write(f"Features: {len(bundle.feature_names)}")
    st.write(f"Base-model training through: {bundle.trained_through}")
    st.write("Historical source: Retrosheet 2021–2025")
    if metrics_path.exists():
        import json
        audit = json.loads(metrics_path.read_text(encoding="utf-8"))
        st.write(f"Untouched test games: {audit['samples']:,}")
        st.write(f"Brier score: {audit['brier_score']:.4f}")
        st.write(f"Log loss: {audit['log_loss']:.4f}")

schedule = mlb.schedule(selected_date)
if not schedule.usable:
    st.error(f"MLB schedule unavailable: {schedule.detail}")
    st.stop()
odds_all = OddsClient(http, settings.odds_api_key).games()
if not settings.odds_api_key:
    st.warning("Add ODDS_API_KEY in Streamlit Secrets. It is the only required third-party key.")

for raw in schedule.payload:
    game = mlb.game_ref(raw)
    with st.container(border=True):
        st.subheader(f"{game.away_team} at {game.home_team}")
        feed = mlb.game_feed(game.game_pk)
        coordinates = feed.payload.get("gameData", {}).get("venue", {}).get("location", {}).get("defaultCoordinates", {}) if feed.usable else {}
        roof = str(feed.payload.get("gameData", {}).get("venue", {}).get("fieldInfo", {}).get("roofType", "") if feed.usable else "").lower()
        weather = WeatherClient(http, settings.weather_base_url).forecast(coordinates.get("latitude"), coordinates.get("longitude"), game.commence_time)
        odds = game_odds(odds_all, game.away_team, game.home_team)
        snapshot, coverage, build_error = None, 0.0, None
        try:
            snapshot, coverage = FreeLiveFeatureBuilder(mlb, bundle.metadata).build(game, raw, feed, weather)
        except Exception as exc:
            build_error = str(exc)

        checks = [
            ("Starters confirmed", starter_ready(raw, feed), "MLB Stats API"),
            ("Lineups confirmed", lineup_ready(feed), "MLB game feed"),
            ("Free feature coverage ≥75%", coverage >= 0.75, f"{coverage:.1%} available"),
            ("Weather or closed roof", weather.usable or "closed" in roof or "dome" in roof, weather.source),
            ("Current odds available", odds is not None, "The Odds API"),
            ("Trained calibrated ensemble", bundle is not None, "Retrosheet 2021–2025"),
        ]
        st.dataframe([{"Check": name, "Status": "✓" if ok else "✗", "Source/detail": detail} for name, ok, detail in checks], hide_index=True, use_container_width=True)
        if build_error:
            st.error(f"Feature build failed: {build_error}")
            continue
        if not all(ok for _, ok, _ in checks):
            st.error("Prediction Withheld")
            if not lineup_ready(feed):
                st.caption("MLB normally posts confirmed lineups close to first pitch. Refresh later; the model will not guess a lineup.")
            continue

        row = snapshot.ordered(bundle.feature_names)
        prediction = bundle.predict([row])[0]
        home_probability = prediction.home_probability
        # Free model is modest; suppress unsupported extreme confidence.
        home_probability = max(0.20, min(0.80, home_probability))
        pick_home = home_probability >= 0.5
        pick, pick_probability = (game.home_team, home_probability) if pick_home else (game.away_team, 1 - home_probability)
        market_probability = odds["home_no_vig"] if pick_home else 1 - odds["home_no_vig"]
        edge = pick_probability - market_probability
        left, middle, right = st.columns(3)
        left.metric("Model pick", pick)
        middle.metric("Calibrated probability", f"{pick_probability:.1%}")
        right.metric("No-vig market edge", f"{edge:+.1%}")
        explanation = explain_prediction(bundle, snapshot)
        st.write({name.title(): f"{points:+.1f} pp" for name, points in explanation.category_points.items() if name != "market"})
        st.caption("Votes: " + " · ".join(f"{vote.model} {vote.home_probability:.1%} home" for vote in prediction.votes))
        st.caption("Probabilities are estimates, not guarantees. The untouched test Brier score is shown in Model audit.")

