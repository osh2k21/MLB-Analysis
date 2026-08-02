import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson, skellam
import requests
from datetime import datetime, timezone, timedelta
import random
import math
import re
import unicodedata
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
import time

# The parallel pre-warming used throughout this app (team/pitcher stats,
# Kalshi prop/margin lookups, box-score settlement) runs real network calls
# on background worker threads. Streamlit logs a "missing ScriptRunContext"
# warning for every thread that isn't the main script thread -- this is
# purely cosmetic: it only matters if a thread tries to call an st.* UI
# function (draw something on the page), which none of these background
# calls do. They're pure data-fetching functions, so the warning is 100%
# safe to silence rather than flooding the terminal on every rerun.
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

# Try importing zoneinfo for Central Time conversion (Python 3.9+)
try:
    import zoneinfo
    CENTRAL_TZ = zoneinfo.ZoneInfo("America/Chicago")
except Exception:
    CENTRAL_TZ = None

# -----------------------------------------------------------------------------
# TEAM SHORT-NAME HELPER
# -----------------------------------------------------------------------------
# Just grabbing the last word of a full team name ("Chicago White Sox" ->
# "Sox") works for almost every MLB team, but it collapses Boston Red Sox and
# Chicago White Sox into the same "Sox" label. This explicit map keeps the
# two-word nicknames intact and falls back to last-word splitting for
# everything else.
TEAM_SHORT_NAME_OVERRIDES = {
    "Boston Red Sox": "Red Sox",
    "Chicago White Sox": "White Sox",
}

def team_short_name(full_name):
    """Short/nickname form of a full team name, e.g. 'Los Angeles Dodgers' ->
    'Dodgers', but 'Boston Red Sox' -> 'Red Sox' (not just 'Sox')."""
    if not full_name:
        return "-"
    if full_name in TEAM_SHORT_NAME_OVERRIDES:
        return TEAM_SHORT_NAME_OVERRIDES[full_name]
    return full_name.split()[-1]

# Standard broadcast-style 2-3 letter team codes, for the scorechip's
# "COL" / "SD" style team labels under the score.
TEAM_ABBREVIATIONS = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE", "Colorado Rockies": "COL",
    "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Athletics": "ATH", "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD",
    "San Francisco Giants": "SF", "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB", "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}

def team_abbr(full_name):
    """Standard team code (e.g. 'San Diego Padres' -> 'SD'). Falls back to
    the first 3 letters of the last word if a team isn't in the table, so an
    unexpected name never breaks the display."""
    if not full_name:
        return "-"
    if full_name in TEAM_ABBREVIATIONS:
        return TEAM_ABBREVIATIONS[full_name]
    return full_name.split()[-1][:3].upper()

def pitcher_last_name(full_name):
    """Last name only, for compact display (e.g. 'Gerrit Cole' -> 'Cole'),
    skipping generational suffixes so 'Vladimir Guerrero Jr.' -> 'Guerrero'
    rather than 'Jr.'."""
    if not full_name or "TBD" in full_name:
        return "TBD"
    parts = full_name.replace('.', '').split()
    suffixes = {"jr", "sr", "ii", "iii", "iv", "v"}
    while len(parts) > 1 and parts[-1].lower() in suffixes:
        parts = parts[:-1]
    return parts[-1] if parts else full_name

# City/region-only display name for the scorechip's title (e.g. "Colorado vs
# San Diego"). Most teams can just drop the nickname, but 6 teams share a
# city with another team -- for those, fall back to the nickname instead so
# the title doesn't collapse to the same ambiguous "Chicago vs Chicago" /
# "Los Angeles vs Los Angeles" / "New York vs New York" problem we already
# hit once with "Sox".
TEAM_CITY_DISPLAY_OVERRIDES = {
    "Chicago Cubs": "Cubs", "Chicago White Sox": "White Sox",
    "Los Angeles Angels": "Angels", "Los Angeles Dodgers": "Dodgers",
    "New York Mets": "Mets", "New York Yankees": "Yankees",
}

def team_city_display(full_name):
    """City/region-only name for the scorechip title, e.g. 'Colorado Rockies'
    -> 'Colorado', 'San Diego Padres' -> 'San Diego'. Falls back to the
    team's nickname for the 6 teams that share a city with another team."""
    if not full_name:
        return "-"
    if full_name in TEAM_CITY_DISPLAY_OVERRIDES:
        return TEAM_CITY_DISPLAY_OVERRIDES[full_name]
    return " ".join(full_name.split()[:-1]) or full_name

def ordinal_inning(n):
    """1 -> '1ST', 2 -> '2ND', 3 -> '3RD', 4 -> '4TH', 11 -> '11TH', etc."""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return str(n)
    if 10 <= n % 100 <= 20:
        suffix = "TH"
    else:
        suffix = {1: "ST", 2: "ND", 3: "RD"}.get(n % 10, "TH")
    return f"{n}{suffix}"

def team_logo_html(team_id):
    """Real team logo, linked directly from MLB's own official static CDN
    (mlbstatic.com/team-logos/{team_id}.svg) -- not a copy embedded in this
    app, just a live reference to MLB's own hosted asset, keyed by the same
    real numeric team_id the MLB Stats API already gives us. Falls back to a
    plain glove emoji via onerror if the id is missing or the image 404s,
    so a bad/unmapped id never leaves a broken-image icon on screen."""
    if not team_id:
        return '<div class="scorechip-glove-fallback">🧤</div>'
    url = f"https://www.mlbstatic.com/team-logos/{team_id}.svg"
    return (
        f'<img class="scorechip-logo" src="{url}" alt="logo" '
        f'onerror="this.onerror=null;this.outerHTML=\'<div class=&quot;scorechip-glove-fallback&quot;>🧤</div>\';" />'
    )

# -----------------------------------------------------------------------------
# STREAMLIT PAGE CONFIG & CUSTOM MLB DARK THEME STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Advanced MLB Algorithmic Analytics Engine",
    page_icon="⚾",
    layout="wide"
)

st.markdown("""
<style>
    /* Global Dark Navy Theme */
    .stApp {
        background-color: #0A0F1D !important;
        color: #F8FAFC !important;
    }

    /* Streamlit's own top header/toolbar bar (hamburger menu, "Deploy" button,
       etc.) sits outside .stApp and defaults to a plain white background --
       this was the white strip showing above the app content. */
    header[data-testid="stHeader"] {
        background-color: #0A0F1D !important;
    }
    div[data-testid="stToolbar"] {
        background-color: #0A0F1D !important;
    }
    div[data-testid="stDecoration"] {
        background-image: none !important;
        background-color: #0A0F1D !important;
    }
    html, body {
        background-color: #0A0F1D !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B;
    }
    section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] span {
        color: #E2E8F0 !important;
    }

    /* Typography */
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 0px;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 25px;
    }

    /* Custom MLB Styled Card Containers (Navy + Maroon/Red Accent) */
    .card-container {
        background: linear-gradient(135deg, #0F172A 0%, #17233D 100%);
        border-radius: 14px;
        padding: 24px;
        border: 1px solid #27354F;
        border-left: 5px solid #990000;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -3px rgba(0, 0, 0, 0.7), 0 4px 6px -4px rgba(0, 0, 0, 0.5);
    }
    .card-title {
        color: #60A5FA !important;
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .card-subtitle {
        color: #FFFFFF !important;
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 15px;
    }
    .card-list {
        color: #FFFFFF !important;
        font-size: 0.98rem;
        line-height: 1.8;
        padding-left: 20px;
        margin: 0;
    }
    .card-list li {
        color: #FFFFFF !important;
        margin-bottom: 10px;
    }
    .prob-badge {
        color: #94A3B8 !important;
        font-weight: 600;
        font-size: 0.88rem;
    }
    .book-tag {
        color: #60A5FA !important;
        font-size: 0.82rem;
        font-style: italic;
    }
    .no-market-note {
        color: #FBBF24 !important;
        font-size: 0.88rem;
    }

    /* Streamlit Metric Font Color Customization */
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
    }
    [data-testid="stMetricLabel"] {
        color: #FFFFFF !important;
    }

    /* Force Number Input Labels (Pitcher Name Strikeout Milestone) to White */
    div[data-testid="stNumberInput"] label p,
    [data-testid="stWidgetLabel"] p {
        color: #FFFFFF !important;
    }

    /* Custom Button & Widget Styling */
    div.stButton > button {
        background: linear-gradient(135deg, #990000 0%, #7A0000 100%);
        color: #FFFFFF;
        border: 1px solid #B91C1C;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #B91C1C 0%, #990000 100%);
        border-color: #EF4444;
        color: #FFFFFF;
    }

    /* Compact Broadcast-Style Live Scoreboard Chips */
    .scorechip-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 14px;
        margin-bottom: 10px;
    }
    .scorechip {
        background: #05070D;
        border: 1px solid #1E293B;
        border-radius: 14px;
        padding: 16px 20px 14px;
        min-width: 260px;
        flex: 1 1 260px;
        box-shadow: 0 8px 18px -6px rgba(0, 0, 0, 0.7);
        text-align: center;
    }
    .scorechip-title {
        color: #F8FAFC !important;
        font-size: 1.15rem;
        font-weight: 800;
        letter-spacing: 0.2px;
        margin-bottom: 8px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .scorechip-status-row {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 7px;
        margin-bottom: 10px;
    }
    .scorechip-live-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #F43F5E;
        animation: scorechip-live-pulse 1.4s ease-in-out infinite;
    }
    .scorechip-live-text {
        color: #F43F5E !important;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.5px;
    }
    @keyframes scorechip-live-pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.25; }
    }
    .scorechip-inning-arrow {
        color: #F43F5E !important;
        font-size: 0.7rem;
    }
    .scorechip-inning-text {
        color: #E2E8F0 !important;
        font-size: 0.9rem;
        font-weight: 700;
    }
    .scorechip-info-icon {
        color: #64748B !important;
        font-size: 0.85rem;
        border: 1.5px solid #475569;
        border-radius: 50%;
        width: 16px;
        height: 16px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        cursor: default;
    }
    .scorechip-score-row {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 18px;
        margin-bottom: 12px;
    }
    .scorechip-team-col {
        display: flex;
        flex-direction: column;
        align-items: center;
        min-width: 64px;
    }
    .scorechip-logo {
        width: 34px;
        height: 34px;
        object-fit: contain;
        margin-bottom: 4px;
    }
    .scorechip-glove-fallback {
        font-size: 1.6rem;
        line-height: 1;
        margin-bottom: 4px;
        filter: grayscale(0.15);
    }
    .scorechip-score-num {
        color: #F8FAFC !important;
        font-size: 2.4rem;
        font-weight: 800;
        line-height: 1;
    }
    .scorechip-team-abbr {
        color: #94A3B8 !important;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-top: 3px;
    }
    .scorechip-count-row {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 18px;
        margin-top: 4px;
    }
    .scorechip-pitcher-k-row {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        margin-top: 6px;
        font-size: 0.72rem;
        font-weight: 700;
        color: #94A3B8 !important;
    }
    .scorechip-pitcher-dot {
        display: inline-block;
        width: 6px;
        height: 6px;
        border-radius: 50%;
        margin-right: 4px;
        vertical-align: middle;
    }
    .scorechip-pitcher-k-sep {
        color: #334155 !important;
    }
    .scorechip-count-group {
        display: flex;
        align-items: center;
        gap: 5px;
    }
    .scorechip-label {
        color: #64748B !important;
        font-size: 0.7rem;
        font-weight: 800;
        margin-right: 1px;
    }
    .scorechip-shape {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
    }
    .scorechip-filled-ball { background: #22C55E; }
    .scorechip-empty-ball { background: transparent; border: 1.5px solid #475569; }
    .scorechip-filled-strike { background: #FBBF24; }
    .scorechip-empty-strike { background: transparent; border: 1.5px solid #475569; }
    .scorechip-filled-out { background: #EF4444; }
    .scorechip-empty-out { background: transparent; border: 1.5px solid #475569; }
    .scorechip-pick {
        color: #94A3B8 !important;
        font-size: 0.72rem;
        margin-top: 12px;
        padding-top: 10px;
        border-top: 1px solid #1E293B;
        white-space: nowrap;
    }

    /* Base-runner mini-diamond, used as the center icon between the two
       scores (rendered as inline SVG -- see render_scorechip_html) */
    .scorechip-diamond-svg {
        display: block;
        margin: 0 2px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TIME / SCHEDULE HELPERS
# -----------------------------------------------------------------------------
def convert_utc_to_central(utc_iso_str):
    """Converts UTC ISO timestamp from MLB API to Central Time (CT)."""
    if not utc_iso_str:
        return "TBD"
    try:
        dt_utc = datetime.fromisoformat(utc_iso_str.replace('Z', '+00:00'))
        if CENTRAL_TZ:
            dt_ct = dt_utc.astimezone(CENTRAL_TZ)
        else:
            dt_ct = dt_utc.astimezone(timezone(timedelta(hours=-5)))
        return dt_ct.strftime("%I:%M %p CT")
    except Exception:
        return "TBD"

def convert_utc_to_central_date(utc_iso_str):
    """Same conversion as convert_utc_to_central, but returns just the
    calendar date (YYYY-MM-DD) in Central Time. Used to build a date-aware
    market_map key -- without this, two teams playing a multi-game series
    (very common in MLB) would collide under the same '{away}@{home}' key,
    and whichever game's odds happened to be processed last would silently
    overwrite the others, regardless of which date was actually selected."""
    if not utc_iso_str:
        return None
    try:
        dt_utc = datetime.fromisoformat(utc_iso_str.replace('Z', '+00:00'))
        if CENTRAL_TZ:
            dt_ct = dt_utc.astimezone(CENTRAL_TZ)
        else:
            dt_ct = dt_utc.astimezone(timezone(timedelta(hours=-5)))
        return dt_ct.strftime("%Y-%m-%d")
    except Exception:
        return None

@st.cache_data(ttl=3600)
def get_real_pitcher_stats(pitcher_id, pitcher_name, team_name=""):
    """Fetches REAL season stats (ERA/WHIP/K9/IP, and throwing hand) for a
    starting pitcher from the official MLB Stats API. No random numbers: if
    the live call fails, falls back to a neutral league-average default,
    clearly flagged via is_real=False, rather than inventing pitcher-
    specific noise. This is a model INPUT, not a bet; it never determines
    what odds or lines are shown to the user.

    Retries up to 3 times (5 seconds apart) before giving up -- a transient
    MLB API hiccup (empty response, brief network blip) would otherwise
    trigger the fallback once and then stay stuck showing placeholder stats
    for the full hour-long cache, since the whole point of caching is to
    NOT re-hit the API again until the cache expires. Retrying here, before
    the result ever gets cached, means only a genuinely persistent outage
    (not a one-off blip) actually results in a cached fallback."""
    default_stats = {
        'era': 4.20, 'whip': 1.30, 'k9': 8.2, 'ip_avg': 5.0, 'k_avg': 4.5,
        'throws': 'R', 'is_real': False
    }

    if not pitcher_id or "TBD" in str(pitcher_name):
        return default_stats

    url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}?hydrate=stats(group=[pitching],type=[season])"
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            res = requests.get(url, timeout=6).json()
            person = res.get('people', [{}])[0]
            stats_list = person.get('stats', [])
            throws = person.get('pitchHand', {}).get('code', 'R')

            era, whip, k9, ip_avg, k_avg = 4.20, 1.30, 8.2, 5.0, 4.5
            found_real_splits = bool(stats_list and stats_list[0].get('splits'))
            if found_real_splits:
                stat = stats_list[0]['splits'][-1]['stat']
                era = float(stat.get('era', 4.20))
                whip = float(stat.get('whip', 1.30))
                k9 = float(stat.get('strikeoutsPer9Inn', 8.2))
                games = int(stat.get('gamesStarted', stat.get('gamesPlayed', 1)))
                innings = float(stat.get('inningsPitched', 5.0))
                ip_avg = max(3.0, min(7.0, round(innings / max(1, games), 1)))
                # Small-sample pitchers (a rookie call-up, a single long relief
                # outing, very few starts logged) can produce a wildly inflated
                # K/9 rate on their own -- e.g. 1 IP with 3 Ks is a 27.0 K/9. This
                # was previously only floored, not capped, and a resulting
                # extreme k_avg (like 35) could crash the strikeout-milestone
                # widget downstream, which validates its own default value
                # against a max of 15. Capped here at the same realistic ceiling
                # so it can never propagate an unrealistic number anywhere else
                # that uses this as a model input.
                k_avg = round((k9 / 9.0) * ip_avg, 1)

            if found_real_splits or attempt == max_attempts - 1:
                # Either got real data, or this was the last attempt -- return
                # now either way (on the final attempt, this returns the
                # placeholder defaults with is_real=False, same as before).
                return {
                    'era': era,
                    'whip': whip,
                    'k9': k9,
                    'ip_avg': ip_avg,
                    'k_avg': max(2.0, min(14.0, k_avg)),
                    'throws': throws,
                    # Must match found_real_splits, NOT just "no exception was
                    # thrown" -- a structurally-empty-but-technically-successful
                    # API response (stats_list empty or missing splits)
                    # previously fell through to the hardcoded defaults above
                    # while still claiming is_real=True, silently presenting
                    # fabricated placeholder numbers as genuine fetched stats
                    # with no warning shown anywhere.
                    'is_real': found_real_splits
                }
            time.sleep(5)  # empty response, but attempts remain -- wait and retry
        except Exception:
            if attempt == max_attempts - 1:
                return default_stats
            time.sleep(5)  # network hiccup, but attempts remain -- wait and retry
    return default_stats  # unreachable in practice, safety net only

@st.cache_data(ttl=1800)
def get_league_standings(season):
    """Fetches every MLB team's REAL win-loss record in ONE call (shared
    across all 30 teams for the whole slate, cached for 30 minutes). No
    estimation: if this call fails, returns {} and every team-quality
    lookup below falls back to a neutral .500 default rather than a
    guess."""
    try:
        url = f"https://statsapi.mlb.com/api/v1/standings?leagueId=103,104&season={season}&standingsTypes=regularSeason"
        res = requests.get(url, timeout=8).json()
        out = {}
        for record in res.get('records', []):
            for tr in record.get('teamRecords', []):
                tid = tr.get('team', {}).get('id')
                if tid:
                    out[tid] = {
                        'wins': int(tr.get('wins', 0)),
                        'losses': int(tr.get('losses', 0)),
                        'win_pct': float(tr.get('winningPercentage', 0.5)),
                    }
        return out
    except Exception:
        return {}

@st.cache_data(ttl=3600)
def get_team_run_environment(team_id, season):
    """REAL team-level runs-scored and runs-allowed rates (season totals
    divided by games played), and REAL team pitching-staff ERA -- all
    pulled directly from the MLB Stats API's team hitting/pitching stat
    groups. Used for Pythagorean win expectation and for run/total
    projections. Falls back to neutral league-average-ish values (flagged
    is_real=False) only if the live call fails."""
    default = {'runs_pg': 4.3, 'runs_allowed_pg': 4.3, 'staff_era': 4.20, 'games': 0, 'is_real': False}
    try:
        hit_url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/stats?stats=season&group=hitting&season={season}"
        hit_res = requests.get(hit_url, timeout=6).json()
        hit_splits = hit_res.get('stats', [{}])[0].get('splits', [])

        pitch_url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/stats?stats=season&group=pitching&season={season}"
        pitch_res = requests.get(pitch_url, timeout=6).json()
        pitch_splits = pitch_res.get('stats', [{}])[0].get('splits', [])

        if not hit_splits or not pitch_splits:
            return default

        hit_stat = hit_splits[0]['stat']
        pitch_stat = pitch_splits[0]['stat']
        games = max(1, int(hit_stat.get('gamesPlayed', 1)))
        runs_scored = float(hit_stat.get('runs', games * 4.3))
        runs_allowed = float(pitch_stat.get('runs', games * 4.3))
        staff_era = float(pitch_stat.get('era', 4.20))

        return {
            'runs_pg': runs_scored / games,
            'runs_allowed_pg': runs_allowed / games,
            'staff_era': staff_era,
            'games': games,
            'is_real': True
        }
    except Exception:
        return default

@st.cache_data(ttl=3600)
def get_team_fielding(team_id, season):
    """REAL team fielding percentage and error count from the MLB Stats
    API's fielding stat group. This is a simpler proxy than advanced
    Defensive Runs Saved / Outs Above Average (which live in Statcast, not
    the free API), but it's real, live data -- never a random number."""
    default = {'fielding_pct': 0.984, 'errors': 90, 'is_real': False}
    try:
        url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/stats?stats=season&group=fielding&season={season}"
        res = requests.get(url, timeout=6).json()
        splits = res.get('stats', [{}])[0].get('splits', [])
        if splits:
            stat = splits[0]['stat']
            return {
                'fielding_pct': float(stat.get('fielding', 0.984)),
                'errors': int(stat.get('errors', 90)),
                'is_real': True
            }
    except Exception:
        pass
    return default

def compute_team_quality(team_id, season, standings_map):
    """Blends REAL winning percentage (from standings) with REAL
    Pythagorean win expectation (from real runs scored/allowed), then
    regresses toward .500 early in the season when the sample size is
    still small -- a hot first two weeks shouldn't swing the model as hard
    as a full season would. No random numbers anywhere in this
    calculation."""
    record = standings_map.get(team_id)
    run_env = get_team_run_environment(team_id, season)

    if record is None and not run_env['is_real']:
        return {
            'quality': 0.5, 'win_pct': 0.5, 'pythag_pct': None, 'games': 0,
            'runs_pg': 4.3, 'runs_allowed_pg': 4.3, 'staff_era': 4.20, 'is_real': False
        }

    win_pct = record['win_pct'] if record else 0.5
    games = (record['wins'] + record['losses']) if record else run_env['games']

    pythag_pct = None
    if run_env['is_real'] and run_env['runs_pg'] > 0 and run_env['runs_allowed_pg'] > 0:
        rs, ra = run_env['runs_pg'], run_env['runs_allowed_pg']
        exponent = 1.83  # widely-used refined Pythagorean exponent (vs. the raw exponent of 2)
        pythag_pct = (rs ** exponent) / ((rs ** exponent) + (ra ** exponent))

    raw_quality = ((win_pct * 0.5) + (pythag_pct * 0.5)) if pythag_pct is not None else win_pct

    # Early-season regression toward .500: shrinks small samples (e.g. a
    # 12-3 April start) back toward a neutral prior instead of taking a
    # 15-game record at full face value.
    shrink_weight = min(1.0, games / 50.0)
    quality = 0.5 + (raw_quality - 0.5) * shrink_weight

    return {
        'quality': quality, 'win_pct': win_pct, 'pythag_pct': pythag_pct, 'games': games,
        'runs_pg': run_env['runs_pg'], 'runs_allowed_pg': run_env['runs_allowed_pg'],
        'staff_era': run_env['staff_era'], 'is_real': (record is not None) or run_env['is_real']
    }

def log5_win_prob(quality_a, quality_b):
    """Bill James' Log5 method -- combines two teams' true-quality estimates
    into a head-to-head win probability for team A. A well-established,
    tested sabermetric baseline, used here instead of an arbitrary flat
    home-win constant with ad hoc terms stacked on top."""
    denom = quality_a + quality_b - 2 * quality_a * quality_b
    if abs(denom) < 1e-9:
        return 0.5
    return (quality_a - quality_a * quality_b) / denom

@st.cache_data(ttl=3600)
def get_pitcher_rest_days(pitcher_id, game_date_str, season):
    """REAL days of rest since the starter's last appearance, computed from
    his actual MLB game log. Returns None (not a guess) if unavailable, so
    the caller can skip the rest-day adjustment entirely rather than
    fabricate one."""
    if not pitcher_id or not game_date_str:
        return None
    try:
        url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats?stats=gameLog&group=pitching&season={season}"
        res = requests.get(url, timeout=6).json()
        splits = res.get('stats', [{}])[0].get('splits', [])
        if not splits:
            return None
        game_date = datetime.strptime(game_date_str, "%Y-%m-%d")
        past_dates = []
        for s in splits:
            d = s.get('date')
            if d:
                try:
                    dt = datetime.strptime(d, "%Y-%m-%d")
                    if dt < game_date:
                        past_dates.append(dt)
                except Exception:
                    continue
        if not past_dates:
            return None
        return (game_date - max(past_dates)).days
    except Exception:
        return None

@st.cache_data(ttl=3600)
def get_team_platoon_ops(team_id, season, vs_hand):
    """Best-effort fetch of the team's REAL OPS specifically vs LHP or vs
    RHP this season, using MLB Stats API situational split codes. Returns
    None (skip the adjustment) rather than a guess if this split isn't
    available or parseable for a given team/season -- it is never used to
    fabricate a number."""
    sit_code = 'vl' if vs_hand == 'L' else 'vr'
    try:
        url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/stats?stats=season&group=hitting&season={season}&sitCodes={sit_code}"
        res = requests.get(url, timeout=6).json()
        splits = res.get('stats', [{}])[0].get('splits', [])
        if splits:
            ops = splits[0]['stat'].get('ops')
            if ops not in (None, '', '.---'):
                return float(ops)
    except Exception:
        pass
    return None

@st.cache_data(ttl=3600)
def get_league_ops_leaders(season, limit=250):
    """Fetches the league's REAL OPS leaderboard ONCE per hour (shared
    across every game/team that day), used to identify each team's actual
    best qualified hitter -- and, for the confirmed-lineup analysis below,
    to look up any of that day's real starters for free if they happen to
    already be in this top-N list, without needing a separate live call
    per player."""
    try:
        url = f"https://statsapi.mlb.com/api/v1/stats/leaders?leaderCategories=onBasePlusSlugging&season={season}&sportId=1&limit={limit}"
        res = requests.get(url, timeout=8).json()
        leaders = res.get('leagueLeaders', [{}])[0].get('leaders', [])
        by_team = {}
        by_player = {}
        for l in leaders:
            tid = l.get('team', {}).get('id')
            pid = l.get('person', {}).get('id')
            ops_val = float(l.get('value')) if l.get('value') not in (None, '') else None
            name = l.get('person', {}).get('fullName', 'Team Hitter')
            if tid and tid not in by_team:
                by_team[tid] = {'name': name, 'person_id': pid, 'ops': ops_val}
            if pid:
                by_player[pid] = {'name': name, 'ops': ops_val}
        return {'by_team': by_team, 'by_player': by_player}
    except Exception:
        return {'by_team': {}, 'by_player': {}}

@st.cache_data(ttl=3600)
def get_player_hitting_stats(person_id, season):
    """REAL season batting average, OPS, and home-run-per-game rate for a
    specific player, in one live call to the MLB Stats API. Returns None if
    unavailable, rather than a guess."""
    if not person_id:
        return None
    try:
        url = f"https://statsapi.mlb.com/api/v1/people/{person_id}?hydrate=stats(group=[hitting],type=[season])"
        res = requests.get(url, timeout=6).json()
        stats_list = res.get('people', [{}])[0].get('stats', [])
        if stats_list and stats_list[0].get('splits'):
            stat = stats_list[0]['splits'][-1]['stat']
            avg = stat.get('avg')
            ops = stat.get('ops')
            games = stat.get('gamesPlayed')
            home_runs = stat.get('homeRuns')
            hr_per_game = None
            if games and home_runs is not None and int(games) > 0:
                hr_per_game = float(home_runs) / int(games)
            if avg not in (None, '', '.---'):
                return {
                    'avg': float(avg),
                    'ops': float(ops) if ops not in (None, '', '.---') else None,
                    'hr_per_game': hr_per_game
                }
    except Exception:
        pass
    return None

@st.cache_data(ttl=3600)
def get_player_batting_average(person_id, season):
    """REAL season batting average for a specific player. Thin wrapper
    around get_player_hitting_stats, kept for existing callers. Returns
    None if unavailable, rather than a guess."""
    stats = get_player_hitting_stats(person_id, season)
    return stats['avg'] if stats else None

def get_lineup_hitters(lineup, ops_leaders_by_player, season, max_players=9):
    """Fetches REAL batting average/OPS for each player in a CONFIRMED
    starting lineup (already parsed for free from the schedule response).
    Checks the cached league OPS-leaders lookup first (no extra call) --
    only players not already in that top-250 list trigger an individual
    live call, and those are cached too, so revisiting the same game/day
    costs nothing extra. Bounded to at most `max_players` (a real lineup
    has 9), so worst case this is ~9 calls for the ONE game being viewed --
    never applied across a full slate. Skips (does not fabricate) any
    player whose real stats can't be found."""
    results = []
    for player in lineup[:max_players]:
        pid = player.get('id')
        name = player.get('name', 'Unknown')
        if not pid:
            continue
        leader_hit = ops_leaders_by_player.get(pid)
        if leader_hit and leader_hit.get('ops') is not None:
            ba = get_player_batting_average(pid, season)  # cached; cheap if already fetched elsewhere
            results.append({'name': name, 'ba': ba if ba is not None else 0.250, 'ops': leader_hit['ops'], 'is_real': ba is not None})
        else:
            stats = get_player_hitting_stats(pid, season)
            if stats:
                results.append({'name': name, 'ba': stats['avg'], 'ops': stats.get('ops'), 'is_real': True})
    return results

def get_key_hitter_for_team(team_id, team_name, season, ops_leaders_map):
    """Picks the team's REAL OPS leader (from the live league leaderboard)
    as the 'key hitter', then fetches that specific player's REAL season
    batting average, OPS, and home-run rate. Falls back to the team's active
    roster (still with real per-player stats) if the team has no qualified
    OPS leader yet (e.g. very early season). Only returns is_real=False, with
    a flat neutral default, if no real figure could be found anywhere --
    never an invented batting average or HR rate."""
    by_team = ops_leaders_map.get('by_team', {}) if isinstance(ops_leaders_map, dict) and 'by_team' in ops_leaders_map else ops_leaders_map
    entry = by_team.get(team_id)
    if entry and entry.get('person_id'):
        stats = get_player_hitting_stats(entry['person_id'], season)
        if stats and stats.get('avg') is not None:
            return {
                'name': entry['name'], 'ba': stats['avg'], 'ops': entry.get('ops'),
                'hr_per_game': stats.get('hr_per_game'), 'is_real': True
            }

    try:
        url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=active"
        res = requests.get(url, timeout=6).json()
        roster = res.get('roster', [])
        hitters = [p for p in roster if p.get('position', {}).get('abbreviation') not in ['P', 'TWP']]
        if hitters:
            player = hitters[0]
            name = player.get('person', {}).get('fullName', 'Active Hitter')
            pid = player.get('person', {}).get('id')
            stats = get_player_hitting_stats(pid, season)
            if stats and stats.get('avg') is not None:
                return {
                    'name': name, 'ba': stats['avg'], 'ops': stats.get('ops'),
                    'hr_per_game': stats.get('hr_per_game'), 'is_real': True
                }
    except Exception:
        pass

    return {'name': "Team Hitter", 'ba': 0.250, 'ops': None, 'hr_per_game': None, 'is_real': False}

# Static, periodically-updated reference of MLB park run-scoring factors
# (100 = neutral, >100 favors hitters, <100 favors pitchers). Park factors
# shift somewhat year to year and even differ across published sources for
# the same season, so treat these as directional multi-year tendencies --
# NOT this exact season's precise number. There is no free live API for
# this; refresh periodically from a source like FanGraphs or Baseball
# Savant park factors if you want current-year precision.
PARK_FACTORS = {
    "Coors Field": 1.12, "Great American Ball Park": 1.07, "Fenway Park": 1.05,
    "Oriole Park at Camden Yards": 1.05, "Yankee Stadium": 1.03, "Dodger Stadium": 1.03,
    "Wrigley Field": 1.02, "Globe Life Field": 1.02, "Chase Field": 1.01,
    "Nationals Park": 1.01, "Citizens Bank Park": 1.02, "American Family Field": 1.03,
    "Rogers Centre": 1.01, "Angel Stadium": 1.01, "Truist Park": 1.00, "Target Field": 1.00,
    "Guaranteed Rate Field": 1.00, "Progressive Field": 0.99, "Kauffman Stadium": 0.99,
    "Minute Maid Park": 0.99, "Comerica Park": 0.98, "Busch Stadium": 0.97,
    "Oakland Coliseum": 0.97, "loanDepot park": 0.96, "Petco Park": 0.95,
    "PNC Park": 0.94, "T-Mobile Park": 0.94, "Oracle Park": 0.92,
}
DEFAULT_PARK_FACTOR = 1.00

def get_park_factor(venue_name):
    return PARK_FACTORS.get(venue_name, DEFAULT_PARK_FACTOR)

def compute_weather_run_factor(weather):
    """Small multiplicative adjustment to expected runs, from MLB's own
    per-game weather data (wind speed/direction, temperature) -- this data
    was already being fetched (hydrate=weather in the schedule call) but
    was never actually applied anywhere; the model previously only used a
    static, day-independent park factor.

    Returns (factor, note). factor is 1.0 (no adjustment) whenever weather
    data is missing, indoors, or unparseable -- consistent with the rest of
    the app's real-data-only philosophy: never fabricate an adjustment when
    real conditions for this specific game aren't actually available.

    Effect sizes are intentionally modest and capped -- this is meant to be
    a real but secondary nudge on top of the pitcher/team-quality-driven
    projection, not something that can dominate it."""
    if not weather or not isinstance(weather, dict):
        return 1.0, "No weather data available for this game."

    condition = str(weather.get('condition', '') or '').lower()
    wind_str = str(weather.get('wind', '') or '').strip()
    temp_str = str(weather.get('temp', '') or '').strip()

    if 'roof closed' in condition or 'dome' in condition or 'indoor' in condition:
        return 1.0, "Indoor/dome game -- climate controlled, no weather adjustment applied."

    factor = 1.0
    notes = []

    wind_match = re.match(r'(\d+)\s*mph', wind_str, re.IGNORECASE)
    if wind_match:
        wind_mph = int(wind_match.group(1))
        wind_lower = wind_str.lower()
        if wind_mph >= 5:
            if 'out to' in wind_lower:
                # Blowing out helps fly balls/HRs carry -- boosts scoring.
                boost = min(0.15, (wind_mph - 5) * 0.012)
                factor += boost
                notes.append(f"Wind {wind_mph} mph blowing OUT ({boost*100:+.1f}% runs)")
            elif 'in from' in wind_lower:
                # Blowing in knocks fly balls down -- suppresses scoring.
                reduction = min(0.15, (wind_mph - 5) * 0.012)
                factor -= reduction
                notes.append(f"Wind {wind_mph} mph blowing IN ({-reduction*100:+.1f}% runs)")
            else:
                notes.append(f"Wind {wind_mph} mph crosswind/variable -- negligible run impact")
        else:
            notes.append(f"Wind {wind_mph} mph -- light, negligible impact")

    temp_match = re.match(r'(\d+)', temp_str)
    if temp_match:
        temp_f = int(temp_match.group(1))
        # Warmer air is less dense -- ball carries slightly further,
        # roughly the reverse in cold. ~0.15% total runs per degree away
        # from a 70F baseline, capped at +/-8% at the extremes.
        delta = temp_f - 70
        temp_adj = max(-0.08, min(0.08, delta * 0.0015))
        factor += temp_adj
        if abs(temp_adj) >= 0.01:
            direction = "warmer" if temp_adj > 0 else "colder"
            notes.append(f"{temp_f}°F ({direction} than 70°F baseline, {temp_adj*100:+.1f}% runs)")

    factor = max(0.85, min(1.15, factor))
    return factor, ("; ".join(notes) if notes else "No usable wind/temp data for this game.")

def estimate_expected_runs(pitcher_era, opposing_team_runs_pg, park_factor):
    """Blends the starting pitcher's REAL ERA with the opposing team's REAL
    season runs-per-game rate, then applies the park's run-scoring factor.
    Replaces an arbitrary batting-average-based formula with two real,
    live-fetched inputs."""
    base = (pitcher_era + opposing_team_runs_pg) / 2.0
    return max(1.5, base * park_factor)

@st.cache_data(ttl=10)
def fetch_mlb_schedule(date_str):
    """Fetch MLB schedule, live scores, lineups, ball-to-ball count, and game status from MLB API."""
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}&hydrate=probablePitcher,lineups,weather,team,linescore,status"
    try:
        res = requests.get(url, timeout=10)
        data = res.json()
        games = []
        if 'dates' in data and len(data['dates']) > 0:
            for g in data['dates'][0]['games']:
                utc_date = g.get('gameDate', '')
                time_ct = convert_utc_to_central(utc_date)

                # Doubleheaders: MLB returns each game of a doubleheader as its
                # own entry, same two teams, different gamePk and start time.
                # Without surfacing gameNumber/time, two genuinely different
                # games (one already Final, one still Live) look identical
                # and contradictory in the UI -- same matchup name, wildly
                # different score/state.
                game_number = g.get('gameNumber', 1)
                doubleheader_flag = g.get('doubleHeader', 'N')  # 'Y'/'S' = part of a doubleheader, 'N' = single game
                is_doubleheader = doubleheader_flag in ('Y', 'S')

                status_data = g.get('status', {})
                abstract_state = status_data.get('abstractGameState', 'Preview')
                detailed_state = status_data.get('detailedState', 'Scheduled')

                away_score = g['teams']['away'].get('score', 0) if abstract_state in ['Live', 'Final'] else 0
                home_score = g['teams']['home'].get('score', 0) if abstract_state in ['Live', 'Final'] else 0

                linescore = g.get('linescore', {})
                current_inning = linescore.get('currentInning', '')
                inning_state = linescore.get('inningState', '')
                balls = linescore.get('balls', 0)
                strikes = linescore.get('strikes', 0)
                outs = linescore.get('outs', 0)
                # Base-runner occupancy: MLB's linescore includes an
                # "offense" sub-object with a "first"/"second"/"third" key
                # ONLY when that base is occupied. Comes from the SAME
                # linescore data already being fetched -- zero extra API
                # calls. Defensive .get() so an unexpected shape just shows
                # empty bases rather than erroring.
                offense = linescore.get('offense', {})
                on_first = bool(offense.get('first'))
                on_second = bool(offense.get('second'))
                on_third = bool(offense.get('third'))

                away_pitcher_name = g['teams']['away'].get('probablePitcher', {}).get('fullName', 'TBD Pitcher')
                home_pitcher_name = g['teams']['home'].get('probablePitcher', {}).get('fullName', 'TBD Pitcher')
                pitchers_announced = ("TBD" not in away_pitcher_name) and ("TBD" not in home_pitcher_name)

                # Confirmed starting lineups, if MLB has posted them yet (usually
                # ~1-2 hours before first pitch). This comes from the SAME
                # schedule call above (hydrate=lineups) -- zero extra API cost.
                # Empty lists just mean lineups aren't posted yet, which is the
                # normal state for most of the day.
                lineups_data = g.get('lineups', {})
                away_lineup = [
                    {'id': p.get('id'), 'name': p.get('fullName', 'Unknown')}
                    for p in lineups_data.get('awayPlayers', []) if p.get('id')
                ]
                home_lineup = [
                    {'id': p.get('id'), 'name': p.get('fullName', 'Unknown')}
                    for p in lineups_data.get('homePlayers', []) if p.get('id')
                ]

                game_info = {
                    'game_pk': g.get('gamePk'),
                    'game_time_ct': time_ct,
                    'game_datetime_utc': utc_date,
                    'game_date': g.get('officialDate', date_str),
                    'game_number': game_number,
                    'is_doubleheader': is_doubleheader,
                    'away_team': g['teams']['away']['team']['name'],
                    'home_team': g['teams']['home']['team']['name'],
                    'away_team_id': g['teams']['away']['team']['id'],
                    'home_team_id': g['teams']['home']['team']['id'],
                    'away_pitcher': away_pitcher_name,
                    'home_pitcher': home_pitcher_name,
                    'away_pitcher_id': g['teams']['away'].get('probablePitcher', {}).get('id'),
                    'home_pitcher_id': g['teams']['home'].get('probablePitcher', {}).get('id'),
                    'away_lineup': away_lineup,
                    'home_lineup': home_lineup,
                    'venue': g.get('venue', {}).get('name', 'Standard Ballpark'),
                    'weather': g.get('weather', {}),
                    'abstract_state': abstract_state,
                    'detailed_state': detailed_state,
                    'away_score': away_score,
                    'home_score': home_score,
                    'current_inning': current_inning,
                    'inning_state': inning_state,
                    'balls': balls,
                    'strikes': strikes,
                    'outs': outs,
                    'on_first': on_first,
                    'on_second': on_second,
                    'on_third': on_third,
                    'pitchers_announced': pitchers_announced
                }
                games.append(game_info)
        return games
    except Exception as e:
        st.error(f"Error fetching MLB schedule: {e}")
        return []

# -----------------------------------------------------------------------------
# ODDS MATH HELPERS
# -----------------------------------------------------------------------------
def american_to_prob(american_odds):
    """Convert American odds to raw implied (vig-included) probability."""
    o = float(american_odds)
    if o < 0:
        return (-o) / ((-o) + 100)
    else:
        return 100 / (o + 100)

def american_to_decimal(american_odds):
    """Convert American odds to a decimal payout multiplier."""
    o = float(american_odds)
    if o > 0:
        return 1.0 + (o / 100.0)
    else:
        return 1.0 + (100.0 / abs(o))

def fmt_american(odds):
    """Format a raw American odds number/string with an explicit +/- sign."""
    if odds is None:
        return "N/A"
    try:
        o = int(round(float(odds)))
    except Exception:
        return str(odds)
    return f"+{o}" if o > 0 else f"{o}"

def prob_to_american(prob):
    """Convert a MODEL probability into 'fair odds' notation. This represents
    the model's own assessment only -- it is never presented as a real,
    bettable sportsbook price."""
    if prob >= 0.99: return "-10000"
    if prob <= 0.01: return "+10000"
    if prob >= 0.5:
        odds = int(round(- (prob / (1 - prob)) * 100))
        return f"{odds}"
    else:
        odds = int(round(((1 - prob) / prob) * 100))
        return f"+{odds}"

def prob_to_multiplier(prob):
    """Model fair-odds payout multiplier (not a real market price)."""
    if prob <= 0:
        return "100.00x"
    return f"{(1.0 / prob):.2f}x"

# -----------------------------------------------------------------------------
# REAL SPORTSBOOK ODDS -- The Odds API (no fabricated lines anywhere below)
# -----------------------------------------------------------------------------
def _find_market(bookmakers, preferred_key, market_key, strict=False):
    """Looks for `market_key` on the preferred bookmaker first; if the
    preferred book doesn't post that market, checks every other listed book
    -- UNLESS strict=True, in which case ONLY the preferred book counts and
    no fallback happens at all. strict=True is what makes selecting Kalshi
    actually mean 'Kalshi priced this, and only Kalshi' instead of silently
    substituting whatever other book happened to have the market.
    Returns (market_dict, book_title, was_fallback) or (None, None, False) if
    NO tracked book currently offers that market for this game -- in which
    case the caller must treat the bet as unavailable, not invent a line."""
    preferred = next((b for b in bookmakers if b.get('key') == preferred_key), None)
    if strict:
        if not preferred:
            return None, None, False
        mkt = next((m for m in preferred.get('markets', []) if m.get('key') == market_key), None)
        if mkt:
            return mkt, preferred.get('title', preferred.get('key', 'Unknown Book')), False
        return None, None, False
    ordered = ([preferred] if preferred else []) + [b for b in bookmakers if b is not preferred]
    for b in ordered:
        mkt = next((m for m in b.get('markets', []) if m.get('key') == market_key), None)
        if mkt:
            return mkt, b.get('title', b.get('key', 'Unknown Book')), (b is not preferred)
    return None, None, False

@st.cache_data(ttl=600)
def fetch_bulk_market_odds(api_key, preferred_key, strict=False):
    """Pulls REAL moneyline (h2h), run line (spreads), and game total (totals)
    odds for every scheduled MLB game from The Odds API. Returns
    (market_map, diagnostic) -- market_map is keyed by
    '{away_team}@{home_team}' (using the odds API's own team-name strings).
    A game or a specific market that no tracked book is currently posting is
    simply absent from the result -- nothing here is estimated or invented.
    `diagnostic` reports WHY the map might be empty (bad/missing key, quota
    exceeded, network failure, etc.) instead of silently returning nothing
    with no way to tell what went wrong.

    strict=True (used when Kalshi is the selected exchange) disables the
    normal cross-book fallback entirely -- a leg only appears if Kalshi
    itself is actually pricing it, never silently substituted from
    DraftKings/FanDuel/etc."""
    if not api_key:
        return {}, {'status': 'no_key', 'detail': 'No Odds API key is configured.'}

    url = ("https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"
           f"?apiKey={api_key}&regions=us,us_ex&markets=h2h,spreads,totals&oddsFormat=american")
    try:
        res = requests.get(url, timeout=8)
    except requests.exceptions.Timeout:
        return {}, {'status': 'timeout', 'detail': 'The Odds API request timed out after 8 seconds.'}
    except requests.exceptions.RequestException as e:
        return {}, {'status': 'network_error', 'detail': f'Network error reaching The Odds API: {e}'}

    if res.status_code != 200:
        # The Odds API's own documented meanings for these codes.
        _reason_map = {
            401: 'Unauthorized -- the API key is invalid, expired, or malformed.',
            403: 'Forbidden -- the API key does not have access to this endpoint/plan.',
            429: 'Too Many Requests -- the monthly/rate quota has been exceeded.',
            404: 'Not Found -- the requested sport/endpoint does not exist.',
        }
        detail = _reason_map.get(res.status_code, f'Unexpected HTTP {res.status_code} response.')
        return {}, {'status': 'http_error', 'status_code': res.status_code, 'detail': detail}

    try:
        data = res.json()
    except Exception:
        return {}, {'status': 'bad_json', 'detail': 'The Odds API returned a 200 response that was not valid JSON.'}

    if not data:
        return {}, {'status': 'empty', 'detail': 'The Odds API returned 200 OK but an empty game list -- no MLB games currently have odds posted by any tracked book.'}

    market_map = {}
    _pair_dates_seen = {}  # "{away}@{home}" -> set of dates this pair has appeared on so far
    for game in data:
        home_team = game.get('home_team')
        away_team = game.get('away_team')
        bookmakers = game.get('bookmakers', [])
        if not home_team or not away_team or not bookmakers:
            continue

        entry = {'event_id': game.get('id'), 'h2h': None, 'spreads': None, 'totals': None}
        game_date_ct = convert_utc_to_central_date(game.get('commence_time'))

        mkt, book, fb = _find_market(bookmakers, preferred_key, 'h2h', strict=strict)
        if mkt:
            outcomes = mkt.get('outcomes', [])
            home_o = next((o['price'] for o in outcomes if o.get('name') == home_team), None)
            away_o = next((o['price'] for o in outcomes if o.get('name') == away_team), None)
            if home_o is not None and away_o is not None:
                entry['h2h'] = {'home_odds': home_o, 'away_odds': away_o, 'source_book': book, 'fallback_used': fb}

        mkt, book, fb = _find_market(bookmakers, preferred_key, 'spreads', strict=strict)
        if mkt:
            outcomes = mkt.get('outcomes', [])
            home_side = next((o for o in outcomes if o.get('name') == home_team), None)
            away_side = next((o for o in outcomes if o.get('name') == away_team), None)
            if home_side and away_side and home_side.get('price') is not None and away_side.get('price') is not None:
                entry['spreads'] = {
                    'home_point': home_side.get('point'), 'home_odds': home_side.get('price'),
                    'away_point': away_side.get('point'), 'away_odds': away_side.get('price'),
                    'source_book': book, 'fallback_used': fb
                }

        mkt, book, fb = _find_market(bookmakers, preferred_key, 'totals', strict=strict)
        if mkt:
            outcomes = mkt.get('outcomes', [])
            over_side = next((o for o in outcomes if o.get('name') == 'Over'), None)
            under_side = next((o for o in outcomes if o.get('name') == 'Under'), None)
            if over_side and under_side and over_side.get('price') is not None and under_side.get('price') is not None:
                entry['totals'] = {
                    'point': over_side.get('point'),
                    'over_odds': over_side.get('price'), 'under_odds': under_side.get('price'),
                    'source_book': book, 'fallback_used': fb
                }

        pair_key = f"{away_team}@{home_team}"
        if game_date_ct:
            market_map[f"{game_date_ct}:{pair_key}"] = entry
            _pair_dates_seen.setdefault(pair_key, set()).add(game_date_ct)

        if len(_pair_dates_seen.get(pair_key, set())) <= 1:
            # This pair has only been seen on one date so far in this bulk
            # fetch -- safe to keep as an unambiguous fallback for the rare
            # case a lookup's date-qualified key doesn't resolve (e.g.
            # commence_time was missing).
            market_map[pair_key] = entry
        else:
            # A back-to-back series: the same two teams now span 2+ dates in
            # this same fetch. The team-only key can no longer safely point
            # to just one of them -- remove it rather than let a date-key
            # mismatch silently fall back to the WRONG day's odds (this is
            # exactly what caused the same total/moneyline to show up for
            # both today's and tomorrow's game of a series).
            market_map.pop(pair_key, None)

    if not market_map:
        return market_map, {'status': 'no_games_matched', 'detail': f'The Odds API returned {len(data)} game(s), but none had usable team names/bookmaker data to build a market entry from.'}
    return market_map, {'status': 'ok', 'detail': f'{len(market_map)} market entries loaded from {len(data)} game(s).'}

def normalize_player_name(name):
    """Normalizes a player name for matching across two different data
    sources: the MLB Stats API's 'fullName' (used to identify our pitcher/
    hitter) and the sportsbook's own 'description' string for a prop outcome
    on The Odds API. These frequently differ in small ways that break an
    exact-string match -- accented characters (e.g. 'Julio Urias' vs 'Julio
    Urías'), suffixes (Jr./Sr./II/III), and punctuation/case -- which was
    silently causing every prop lookup to miss and return no legs at all."""
    if not name:
        return ""
    ascii_name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    ascii_name = ascii_name.lower()
    ascii_name = re.sub(r'\b(jr|sr|ii|iii|iv)\b\.?', '', ascii_name)
    ascii_name = re.sub(r'[^a-z0-9\s]', '', ascii_name)
    return re.sub(r'\s+', ' ', ascii_name).strip()

@st.cache_data(ttl=600)
def fetch_event_player_props(api_key, event_id, preferred_key, strict=False):
    """Pulls REAL pitcher-strikeout, batter-hit, and batter-home-run prop
    lines for ONE game via The Odds API's per-event endpoint. This costs an
    extra API credit per call, so it is only invoked for a specific game
    (deep-dive tab, or parlay legs if the user opts in) -- never
    automatically for the whole slate. Returns {} if the book(s) haven't
    posted those props yet.

    strict=True (Kalshi selected) disables cross-book fallback -- a prop
    only shows up if Kalshi itself priced it."""
    if not api_key or not event_id:
        return {}

    url = ("https://api.the-odds-api.com/v4/sports/baseball_mlb/events/"
           f"{event_id}/odds?apiKey={api_key}&regions=us&markets=pitcher_strikeouts,batter_hits,batter_home_runs&oddsFormat=american")
    try:
        res = requests.get(url, timeout=8)
        if res.status_code != 200:
            return {}
        data = res.json()
    except Exception:
        return {}

    bookmakers = data.get('bookmakers', [])
    if not bookmakers:
        return {}

    props = {}
    for market_key in ('pitcher_strikeouts', 'batter_hits', 'batter_home_runs'):
        mkt, book, fb = _find_market(bookmakers, preferred_key, market_key, strict=strict)
        if not mkt:
            continue
        by_player = {}
        for o in mkt.get('outcomes', []):
            player = o.get('description')
            if not player:
                continue
            # Keyed by normalized name (see normalize_player_name) since the
            # book's own spelling of a player's name won't always exactly
            # match the MLB Stats API's fullName used elsewhere in the app.
            norm_key = normalize_player_name(player)
            by_player.setdefault(norm_key, {})[o.get('name')] = {'price': o.get('price'), 'point': o.get('point')}
        if by_player:
            props[market_key] = {'lines': by_player, 'source_book': book, 'fallback_used': fb}

    return props

@st.cache_data(ttl=600)
def fetch_event_alternate_spreads(api_key, event_id, preferred_key, strict=False):
    """Pulls REAL alternate run-margin lines (e.g. -1.5, -2.5, -3.5, ...) for
    ONE game via The Odds API's 'alternate_spreads' market -- this is the
    market that actually matches how Kalshi structures a 'team wins by X+
    runs' contract as a laddered set of thresholds, rather than a single
    -1.5/+1.5 run line. Costs an extra API credit per call, same as player
    props. Returns {} if the book hasn't posted alternate lines for this
    game yet.

    strict=True (Kalshi selected) disables cross-book fallback -- a
    threshold only shows up if Kalshi itself priced it."""
    if not api_key or not event_id:
        return {}

    url = ("https://api.the-odds-api.com/v4/sports/baseball_mlb/events/"
           f"{event_id}/odds?apiKey={api_key}&regions=us,us_ex&markets=alternate_spreads&oddsFormat=american")
    try:
        res = requests.get(url, timeout=8)
        if res.status_code != 200:
            return {}
        data = res.json()
    except Exception:
        return {}

    bookmakers = data.get('bookmakers', [])
    if not bookmakers:
        return {}

    mkt, book, fb = _find_market(bookmakers, preferred_key, 'alternate_spreads', strict=strict)
    if not mkt:
        return {}

    home_team = data.get('home_team')
    away_team = data.get('away_team')
    # Grouped by point value (as an absolute margin) -> {'home': {...}, 'away': {...}}
    # since alternate_spreads lists each team's side of every threshold as
    # its own outcome, same shape as the standard spreads market but with
    # many point values instead of just one.
    by_margin = {}
    for o in mkt.get('outcomes', []):
        name = o.get('name')
        point = o.get('point')
        price = o.get('price')
        if point is None or price is None or name not in (home_team, away_team):
            continue
        margin = abs(point)
        side = 'home' if name == home_team else 'away'
        by_margin.setdefault(margin, {})[side] = {'price': price, 'point': point}

    if not by_margin:
        return {}
    return {'by_margin': by_margin, 'source_book': book, 'fallback_used': fb, 'home_team': home_team, 'away_team': away_team}

# -----------------------------------------------------------------------------
# ADVANCED ALGORITHMIC PREDICTIVE MODEL ENGINE
# -----------------------------------------------------------------------------
def model_moneyline_game(game, market_map, min_edge, min_prob, strict_lineup_mode, season, standings_map, ops_leaders_map):
    """Models win probability using: REAL team quality (winning percentage +
    Pythagorean run-differential expectation, both live from the MLB Stats
    API, regressed for early-season sample size) combined via the Log5
    method as a baseline; then layered with REAL adjustments for the
    specific starting pitcher (ERA/WHIP/K9 vs. league average), REAL team
    pitching-staff ERA (bullpen/depth proxy), REAL team fielding
    percentage, a REAL rest-days check on the starter, a best-effort REAL
    platoon/handedness split, and a sourced (not live-fetched, since no
    free API exposes it) home-field constant. No random-seeded terms
    remain anywhere in this function -- any input that can't be fetched
    live simply contributes zero rather than a guess.
    The MARKET side of this function only ever reflects a REAL line pulled
    from market_map -- if no book has priced this game, has_ml_market is
    False and no market figures are fabricated."""
    away_stats = get_real_pitcher_stats(game['away_pitcher_id'], game['away_pitcher'], game['away_team'])
    home_stats = get_real_pitcher_stats(game['home_pitcher_id'], game['home_pitcher'], game['home_team'])

    away_quality = compute_team_quality(game['away_team_id'], season, standings_map)
    home_quality = compute_team_quality(game['home_team_id'], season, standings_map)

    away_hitter = get_key_hitter_for_team(game['away_team_id'], game['away_team'], season, ops_leaders_map)
    home_hitter = get_key_hitter_for_team(game['home_team_id'], game['home_team'], season, ops_leaders_map)

    away_def = get_team_fielding(game['away_team_id'], season)
    home_def = get_team_fielding(game['home_team_id'], season)

    park_factor = get_park_factor(game.get('venue', ''))
    weather_factor, weather_note = compute_weather_run_factor(game.get('weather'))
    combined_run_factor = park_factor * weather_factor

    # ---- 1) Baseline: REAL team quality combined via Log5 ----
    home_win_prob = log5_win_prob(home_quality['quality'], away_quality['quality'])

    # ---- 2) Home-field constant ----
    # A modest, sourced (not live-fetched) constant based on recent-season
    # MLB home win rates (~52-54% across multiple published analyses). This
    # is a slow-moving league-wide average -- refresh it periodically if
    # you want it tuned to the current year.
    HOME_FIELD_EDGE = 0.025
    home_win_prob += HOME_FIELD_EDGE

    # ---- 3) Starting pitcher adjustment (REAL ERA/WHIP/K9 vs. league avg) ----
    # League-average reference lines below are a fixed modern-era estimate
    # (not live-fetched), used only as a yardstick for how good/bad THIS
    # start is relative to average -- the ERA/WHIP/K9 being compared are
    # always the pitcher's real, live season numbers.
    LEAGUE_AVG_ERA, LEAGUE_AVG_WHIP, LEAGUE_AVG_K9 = 4.05, 1.28, 8.6

    def starter_edge(stats):
        return ((LEAGUE_AVG_ERA - stats['era']) * 0.028
                + (LEAGUE_AVG_WHIP - stats['whip']) * 0.05
                + (stats['k9'] - LEAGUE_AVG_K9) * 0.004)

    home_starter_edge = starter_edge(home_stats)
    away_starter_edge = starter_edge(away_stats)
    home_win_prob += (home_starter_edge - away_starter_edge)

    # ---- 4) Pitching-staff depth (REAL team ERA, includes bullpen innings) ----
    staff_edge = (away_quality['staff_era'] - home_quality['staff_era']) * 0.02
    home_win_prob += staff_edge

    # ---- 5) REAL team fielding percentage ----
    fielding_edge = (home_def['fielding_pct'] - away_def['fielding_pct']) * 3.0
    home_win_prob += fielding_edge

    # ---- 6) Rest days (REAL game log; skipped entirely if unknown) ----
    home_rest = get_pitcher_rest_days(game['home_pitcher_id'], game.get('game_date', ''), season)
    away_rest = get_pitcher_rest_days(game['away_pitcher_id'], game.get('game_date', ''), season)

    def rest_penalty(rest_days):
        if rest_days is None:
            return 0.0
        return -0.012 if rest_days < 4 else 0.0

    home_win_prob += rest_penalty(home_rest) - rest_penalty(away_rest)

    # ---- 7) Best-effort REAL platoon/handedness split ----
    # Skipped entirely (edge = 0) if the live split call doesn't return
    # data for either team -- never guessed.
    home_platoon_ops = get_team_platoon_ops(game['home_team_id'], season, away_stats['throws'])
    away_platoon_ops = get_team_platoon_ops(game['away_team_id'], season, home_stats['throws'])
    if home_platoon_ops is not None and away_platoon_ops is not None:
        platoon_edge = (home_platoon_ops - away_platoon_ops) * 0.05
    else:
        platoon_edge = 0.0
    home_win_prob += platoon_edge

    # ---- 8) Close-game damping ----
    # When the two teams project as nearly even on REAL run-scoring rates,
    # damp the model's confidence slightly rather than let a small edge
    # compound into an overconfident number.
    away_exp_runs = estimate_expected_runs(home_stats['era'], away_quality['runs_pg'], combined_run_factor)
    home_exp_runs = estimate_expected_runs(away_stats['era'], home_quality['runs_pg'], combined_run_factor)
    projected_run_diff = abs(home_exp_runs - away_exp_runs)
    if projected_run_diff < 0.6:
        home_win_prob = 0.5 + (home_win_prob - 0.5) * 0.80

    # ---- 9) Safety bounds only -- NOT an artificial confidence cap ----
    # These exist purely to avoid literal 0%/100% numerical edge cases.
    # Unlike the old model, there is no hard ceiling that blocks the model
    # from reflecting a genuinely lopsided real matchup.
    home_win_prob = max(0.03, min(0.97, home_win_prob))
    away_win_prob = 1.0 - home_win_prob

    is_ready = True
    warning_reason = ""
    if strict_lineup_mode and game['abstract_state'] not in ['Live', 'Final']:
        if not game['pitchers_announced']:
            is_ready = False
            warning_reason = "Pitchers Not Announced"

    if home_win_prob >= away_win_prob:
        pick = game['home_team']
        model_prob = home_win_prob
        underdog = game['away_team']
    else:
        pick = game['away_team']
        model_prob = away_win_prob
        underdog = game['home_team']

    # ---- REAL MARKET ONLY: no fallback fabrication ----
    # Date-qualified key first -- prevents a multi-game series between the
    # same two teams from matching the wrong date's odds. Falls back to the
    # team-only key only if no date-qualified entry exists (e.g. the odds
    # API's commence_time was missing for some reason).
    dated_key = f"{game.get('game_date')}:{game['away_team']}@{game['home_team']}"
    key = f"{game['away_team']}@{game['home_team']}"
    market_entry = market_map.get(dated_key) or market_map.get(key)
    has_ml_market = bool(market_entry and market_entry.get('h2h'))

    market_odds = None
    market_odds_str = "No Market Data"
    implied_prob = None
    edge = None
    is_qualified = False
    source_book = None

    if has_ml_market:
        h2h = market_entry['h2h']
        market_home_prob = american_to_prob(h2h['home_odds'])
        market_away_prob = american_to_prob(h2h['away_odds'])
        source_book = h2h['source_book']
        if pick == game['home_team']:
            market_odds = h2h['home_odds']
            implied_prob = market_home_prob
        else:
            market_odds = h2h['away_odds']
            implied_prob = market_away_prob
        market_odds_str = fmt_american(market_odds)
        edge = (model_prob - implied_prob) * 100.0
        is_qualified = is_ready and (edge >= min_edge) and (model_prob >= max(min_prob, 0.58))

    return {
        'game': f"{game['away_team']} @ {game['home_team']}",
        'game_pk': game.get('game_pk'),
        'game_date': game.get('game_date'),
        'game_number': game.get('game_number', 1),
        'is_doubleheader': game.get('is_doubleheader', False),
        'time_ct': game['game_time_ct'],
        'game_datetime_utc': game.get('game_datetime_utc'),
        'home_team': game['home_team'],
        'away_team': game['away_team'],
        'home_team_id': game.get('home_team_id'),
        'away_team_id': game.get('away_team_id'),
        'home_pitcher': game['home_pitcher'],
        'away_pitcher': game['away_pitcher'],
        'home_pitcher_id': game.get('home_pitcher_id'),
        'away_pitcher_id': game.get('away_pitcher_id'),
        'home_stats': home_stats,
        'away_stats': away_stats,
        'home_hitter': home_hitter,
        'away_hitter': away_hitter,
        'home_def': home_def,
        'away_def': away_def,
        'home_quality': home_quality,
        'away_quality': away_quality,
        'home_rest_days': home_rest,
        'away_rest_days': away_rest,
        'park_factor': park_factor,
        'weather_factor': weather_factor,
        'weather_note': weather_note,
        'projected_run_diff': projected_run_diff,
        'away_exp_runs': away_exp_runs,
        'home_exp_runs': home_exp_runs,
        'pick': pick,
        'underdog': underdog,
        'model_prob': model_prob,
        'market_odds': market_odds,
        'market_odds_str': market_odds_str,
        'implied_prob': implied_prob,
        'edge': edge,
        'fair_odds': prob_to_american(model_prob),
        'fair_multiplier': prob_to_multiplier(model_prob),
        'is_qualified': is_qualified,
        'is_ready': is_ready,
        'warning_reason': warning_reason,
        'has_ml_market': has_ml_market,
        'source_book': source_book,
        'market_entry': market_entry,
        'abstract_state': game['abstract_state'],
        'detailed_state': game['detailed_state'],
        'away_score': game['away_score'],
        'home_score': game['home_score'],
        'current_inning': game['current_inning'],
        'inning_state': game['inning_state'],
        'balls': game['balls'],
        'strikes': game['strikes'],
        'outs': game['outs'],
        'on_first': game.get('on_first', False),
        'on_second': game.get('on_second', False),
        'on_third': game.get('on_third', False)
    }

def calculate_k_prop_prob(k_target, pitcher_k_avg):
    """Model probability of a pitcher recording >= k_target strikeouts."""
    prob = 1.0 - poisson.cdf(k_target - 1, pitcher_k_avg)
    return max(0.05, min(0.95, prob))

def calculate_batter_hit_prob(hitter_ba, line=0.5, pas=4.0):
    """Model probability of a batter clearing a given hits line (default 0.5)."""
    expected_hits = hitter_ba * pas
    prob = 1.0 - poisson.cdf(int(np.floor(line)), expected_hits)
    return max(0.05, min(0.95, prob))

def calculate_batter_hr_prob(hr_per_game, line=0.5):
    """Model probability of a batter clearing a given home-run line (usually
    0.5, i.e. 1+ HR), from the batter's REAL season home-run-per-game rate."""
    if hr_per_game is None:
        hr_per_game = 0.12  # league-average-ish HR/game fallback, only used if real rate unavailable
    prob = 1.0 - poisson.cdf(int(np.floor(line)), hr_per_game)
    return max(0.02, min(0.85, prob))

def calculate_margin_cover_prob(exp_runs_for, exp_runs_against, margin):
    """Model probability that a team wins by MORE than `margin` runs (e.g.
    margin=1.5 -> wins by 2+), modeled as the difference of two independent
    Poisson-distributed run totals (a Skellam distribution) using the SAME
    real expected-runs-per-game inputs already computed for the totals
    model -- not an arbitrary multiplier, an actual distribution over the
    final run differential."""
    if exp_runs_for is None or exp_runs_against is None:
        return None
    threshold = int(np.floor(margin))  # margin=1.5 -> need diff >= 2, i.e. > 1
    prob = skellam.sf(threshold, exp_runs_for, exp_runs_against)
    return max(0.02, min(0.90, float(prob)))

# -----------------------------------------------------------------------------
# REAL-LINE BET BUILDERS (each returns None / [] if no book has posted a line)
# -----------------------------------------------------------------------------
def build_run_line_leg(result):
    """Run-line candidate for the model's moneyline pick, ONLY if a real
    spread is currently posted for this game."""
    entry = result.get('market_entry')
    if not entry or not entry.get('spreads'):
        return None
    sp = entry['spreads']
    if result['pick'] == result['home_team']:
        point, odds = sp.get('home_point'), sp.get('home_odds')
    else:
        point, odds = sp.get('away_point'), sp.get('away_odds')
    if point is None or odds is None:
        return None
    # Model's estimate of covering the run line: a heuristic derived from the
    # moneyline model, damped for the extra run margin required to cover.
    model_prob = max(0.30, min(0.90, result['model_prob'] * 0.82))
    return {
        'desc': f"{result['pick']} {point:+g} Run Line",
        'model_prob': model_prob,
        'market_prob': american_to_prob(odds),
        'odds_str': fmt_american(odds),
        'decimal_odds': american_to_decimal(odds),
        'source_book': sp['source_book'],
        'has_market': True,
        'game': result['game'],
        'type': 'rl'
    }

def build_model_run_line_projection(result):
    """Model-only run-line estimate for when no live spread is posted. MLB
    run lines are conventionally +/-1.5, so the model estimates the chance
    of covering that standard line using its own moneyline-derived
    heuristic. No real odds are attached, since no book is pricing it."""
    model_prob = max(0.30, min(0.90, result['model_prob'] * 0.82))
    return {
        'desc': f"{result['pick']} -1.5 Run Line (model projection -- standard MLB line, no live price posted)",
        'model_prob': model_prob,
        'market_prob': None, 'odds_str': None, 'decimal_odds': None, 'source_book': None,
        'has_market': False,
        'game': result['game'],
        'type': 'rl'
    }

def kalshi_threshold_label(point):
    """Converts a standard sportsbook Over/Under point into the whole-number
    threshold Kalshi's own YES/NO contracts use, e.g. 0.5 -> 1, 1.5 -> 2,
    6.5 -> 7. Whole-number points (less common, but possible) pass through
    unchanged."""
    if point is None:
        return None
    return int(point) if float(point) == int(point) else int(point) + 1

def build_alt_spread_legs(result, away_exp_runs, home_exp_runs, alt_spreads_data, game_label):
    """Builds 'Team wins by X+ runs' legs from REAL alternate-spread
    thresholds (e.g. 1.5, 2.5, 3.5+) -- one leg per posted margin for the
    model's picked team specifically. This is the market structure that
    actually matches how Kalshi prices a margin-of-victory contract (a
    ladder of separate thresholds), as opposed to a single fixed -1.5/+1.5
    run line. Each threshold's probability comes from a real Skellam
    distribution over the two teams' actual expected runs -- not one flat
    damping factor applied to every margin alike."""
    if not alt_spreads_data or not alt_spreads_data.get('by_margin'):
        return []
    picked_home = result['pick'] == result['home_team']
    side_key = 'home' if picked_home else 'away'
    exp_for = home_exp_runs if picked_home else away_exp_runs
    exp_against = away_exp_runs if picked_home else home_exp_runs

    legs = []
    for margin, sides in alt_spreads_data['by_margin'].items():
        side_data = sides.get(side_key)
        if not side_data or side_data.get('price') is None:
            continue
        model_prob = calculate_margin_cover_prob(exp_for, exp_against, margin)
        if model_prob is None:
            continue
        legs.append({
            'desc': f"{result['pick']} Win By {margin:g}+ Runs (YES)",
            'model_prob': model_prob,
            'market_prob': american_to_prob(side_data['price']),
            'odds_str': fmt_american(side_data['price']),
            'decimal_odds': american_to_decimal(side_data['price']),
            'source_book': alt_spreads_data['source_book'],
            'has_market': True,
            'game': game_label,
            'type': 'rl'
        })
    return legs

def build_total_legs(result, away_exp_runs, home_exp_runs, is_kalshi=False):
    """Over/Under legs for the REAL game-total line posted by the book. The
    line itself always comes from the market -- the model only supplies the
    probability of clearing that exact real line. is_kalshi=True labels the
    legs in Kalshi's own YES/NO framing instead of sportsbook Over/Under
    language."""
    entry = result.get('market_entry')
    if not entry or not entry.get('totals'):
        return []
    tot = entry['totals']
    point = tot.get('point')
    if point is None:
        return []
    total_exp = away_exp_runs + home_exp_runs
    floor_pt = int(np.floor(point))
    model_over = 1.0 - poisson.cdf(floor_pt, total_exp)
    model_under = poisson.cdf(floor_pt, total_exp)

    legs = []
    if tot.get('over_odds') is not None:
        desc = f"{result['game']} Total Runs Over {point} (YES)" if is_kalshi else f"{result['game']} Total Runs Over {point}"
        legs.append({
            'desc': desc,
            'model_prob': max(0.05, min(0.95, model_over)),
            'market_prob': american_to_prob(tot['over_odds']),
            'odds_str': fmt_american(tot['over_odds']),
            'decimal_odds': american_to_decimal(tot['over_odds']),
            'source_book': tot['source_book'],
            'has_market': True,
            'game': result['game'],
            'type': 'total'
        })
    if tot.get('under_odds') is not None:
        # Kalshi's real contract is always phrased as ONE question ("will
        # the total exceed {point}?") -- betting the Under side means
        # answering NO to that same question, not a separate "Under...(NO)"
        # double negative. Both legs share the identical 'Over {point}'
        # base statement; only the YES/NO answer differs.
        desc = f"{result['game']} Total Runs Over {point} (NO)" if is_kalshi else f"{result['game']} Total Runs Under {point}"
        legs.append({
            'desc': desc,
            'model_prob': max(0.05, min(0.95, model_under)),
            'market_prob': american_to_prob(tot['under_odds']),
            'odds_str': fmt_american(tot['under_odds']),
            'decimal_odds': american_to_decimal(tot['under_odds']),
            'source_book': tot['source_book'],
            'has_market': True,
            'game': result['game'],
            'type': 'total'
        })
    return legs

def build_model_total_projection(result, away_exp_runs, home_exp_runs):
    """Model-only total-runs projection for when no live total is posted.
    The model derives its own projected total line from expected runs for
    both teams; no real odds are attached, since no book is pricing it."""
    total_exp = away_exp_runs + home_exp_runs
    model_point = round(total_exp * 2) / 2.0
    floor_pt = int(np.floor(model_point))
    model_over = max(0.05, min(0.95, 1.0 - poisson.cdf(floor_pt, total_exp)))
    model_under = max(0.05, min(0.95, poisson.cdf(floor_pt, total_exp)))
    game_label = result['game']
    return [
        {'desc': f"{game_label} Total Runs Over {model_point} (model projection, no live total posted)",
         'model_prob': model_over, 'market_prob': None, 'odds_str': None, 'decimal_odds': None,
         'source_book': None, 'has_market': False, 'game': game_label, 'type': 'total'},
        {'desc': f"{game_label} Total Runs Under {model_point} (model projection, no live total posted)",
         'model_prob': model_under, 'market_prob': None, 'odds_str': None, 'decimal_odds': None,
         'source_book': None, 'has_market': False, 'game': game_label, 'type': 'total'},
    ]

def build_k_prop_legs(pitcher_name, team_name, k_avg, event_props, game_label, is_kalshi=False):
    """Pitcher strikeout Over/Under legs -- ONLY for lines the book actually
    lists for this exact pitcher. is_kalshi=True labels legs in Kalshi's own
    whole-number '+N' YES/NO framing instead of sportsbook Over/Under
    language."""
    if not event_props or 'pitcher_strikeouts' not in event_props:
        return []
    block = event_props['pitcher_strikeouts']
    lines = block['lines'].get(normalize_player_name(pitcher_name))
    if not lines:
        return []
    over = lines.get('Over')
    under = lines.get('Under')
    point = (over or {}).get('point', (under or {}).get('point'))
    if point is None:
        return []
    model_over = calculate_k_prop_prob(int(np.floor(point)) + 1, k_avg)
    threshold = kalshi_threshold_label(point)
    legs = []
    if over and over.get('price') is not None:
        desc = f"{pitcher_name} ({team_name}) +{threshold} Strikeouts (YES)" if is_kalshi else f"{pitcher_name} ({team_name}) Over {point} Strikeouts"
        legs.append({
            'desc': desc,
            'model_prob': model_over, 'market_prob': american_to_prob(over['price']),
            'odds_str': fmt_american(over['price']), 'decimal_odds': american_to_decimal(over['price']),
            'source_book': block['source_book'], 'has_market': True, 'game': game_label, 'type': 'prop'
        })
    if under and under.get('price') is not None:
        desc = f"{pitcher_name} ({team_name}) +{threshold} Strikeouts (NO)" if is_kalshi else f"{pitcher_name} ({team_name}) Under {point} Strikeouts"
        legs.append({
            'desc': desc,
            'model_prob': 1.0 - model_over, 'market_prob': american_to_prob(under['price']),
            'odds_str': fmt_american(under['price']), 'decimal_odds': american_to_decimal(under['price']),
            'source_book': block['source_book'], 'has_market': True, 'game': game_label, 'type': 'prop'
        })
    return legs

def build_model_k_prop_projection(pitcher_name, team_name, k_avg, game_label):
    """Model-only strikeout projection at the pitcher's own natural average
    threshold (derived purely from his real season K/9 and innings-per-start
    via the MLB Stats API), used when no live prop line is posted or props
    aren't enabled. No real odds are attached."""
    target = int(np.floor(k_avg)) + 1
    point = target - 0.5
    model_over = calculate_k_prop_prob(target, k_avg)
    return [
        {'desc': f"{pitcher_name} ({team_name}) Over {point} Strikeouts (model projection, no live prop line posted)",
         'model_prob': model_over, 'market_prob': None, 'odds_str': None, 'decimal_odds': None,
         'source_book': None, 'has_market': False, 'game': game_label, 'type': 'prop'},
        {'desc': f"{pitcher_name} ({team_name}) Under {point} Strikeouts (model projection, no live prop line posted)",
         'model_prob': 1.0 - model_over, 'market_prob': None, 'odds_str': None, 'decimal_odds': None,
         'source_book': None, 'has_market': False, 'game': game_label, 'type': 'prop'},
    ]

def build_hit_prop_legs(hitter_name, team_name, hitter_ba, event_props, game_label, is_kalshi=False):
    """Batter hits Over/Under legs -- ONLY for lines the book actually lists
    for this exact hitter. is_kalshi=True labels legs in Kalshi's own
    whole-number '+N' YES/NO framing instead of sportsbook Over/Under
    language."""
    if not event_props or 'batter_hits' not in event_props:
        return []
    block = event_props['batter_hits']
    lines = block['lines'].get(normalize_player_name(hitter_name))
    if not lines:
        return []
    over = lines.get('Over')
    under = lines.get('Under')
    point = (over or {}).get('point', (under or {}).get('point'))
    if point is None:
        return []
    model_over = calculate_batter_hit_prob(hitter_ba, line=point)
    threshold = kalshi_threshold_label(point)
    legs = []
    if over and over.get('price') is not None:
        desc = f"{hitter_name} ({team_name}) +{threshold} Hits (YES)" if is_kalshi else f"{hitter_name} ({team_name}) Over {point} Hits"
        legs.append({
            'desc': desc,
            'model_prob': model_over, 'market_prob': american_to_prob(over['price']),
            'odds_str': fmt_american(over['price']), 'decimal_odds': american_to_decimal(over['price']),
            'source_book': block['source_book'], 'has_market': True, 'game': game_label, 'type': 'prop'
        })
    if under and under.get('price') is not None:
        desc = f"{hitter_name} ({team_name}) +{threshold} Hits (NO)" if is_kalshi else f"{hitter_name} ({team_name}) Under {point} Hits"
        legs.append({
            'desc': desc,
            'model_prob': 1.0 - model_over, 'market_prob': american_to_prob(under['price']),
            'odds_str': fmt_american(under['price']), 'decimal_odds': american_to_decimal(under['price']),
            'source_book': block['source_book'], 'has_market': True, 'game': game_label, 'type': 'prop'
        })
    return legs

def build_hr_prop_legs(hitter_name, team_name, hr_per_game, event_props, game_label, is_kalshi=False):
    """Batter home-run Over/Under legs -- ONLY for lines the book actually
    lists for this exact hitter (real batter_home_runs market). is_kalshi=True
    labels legs in Kalshi's own '+N HR' YES/NO framing instead of sportsbook
    Over/Under language."""
    if not event_props or 'batter_home_runs' not in event_props:
        return []
    block = event_props['batter_home_runs']
    lines = block['lines'].get(normalize_player_name(hitter_name))
    if not lines:
        return []
    over = lines.get('Over')
    under = lines.get('Under')
    point = (over or {}).get('point', (under or {}).get('point'))
    if point is None:
        return []
    model_over = calculate_batter_hr_prob(hr_per_game, line=point)
    threshold = kalshi_threshold_label(point)
    legs = []
    if over and over.get('price') is not None:
        desc = f"{hitter_name} ({team_name}) +{threshold} HR (YES)" if is_kalshi else f"{hitter_name} ({team_name}) Over {point} Home Runs"
        legs.append({
            'desc': desc,
            'model_prob': model_over, 'market_prob': american_to_prob(over['price']),
            'odds_str': fmt_american(over['price']), 'decimal_odds': american_to_decimal(over['price']),
            'source_book': block['source_book'], 'has_market': True, 'game': game_label, 'type': 'prop'
        })
    if under and under.get('price') is not None:
        desc = f"{hitter_name} ({team_name}) +{threshold} HR (NO)" if is_kalshi else f"{hitter_name} ({team_name}) Under {point} Home Runs"
        legs.append({
            'desc': desc,
            'model_prob': 1.0 - model_over, 'market_prob': american_to_prob(under['price']),
            'odds_str': fmt_american(under['price']), 'decimal_odds': american_to_decimal(under['price']),
            'source_book': block['source_book'], 'has_market': True, 'game': game_label, 'type': 'prop'
        })
    return legs

def build_model_hit_prop_projection(hitter_name, team_name, hitter_ba, game_label):
    """Model-only hits projection at the standard 0.5-hits threshold
    (derived purely from the hitter's own batting average via a Poisson
    model), used when no live prop line is posted or props aren't enabled.
    No real odds are attached."""
    model_over = calculate_batter_hit_prob(hitter_ba, line=0.5)
    return [
        {'desc': f"{hitter_name} ({team_name}) Over 0.5 Hits (model projection, no live prop line posted)",
         'model_prob': model_over, 'market_prob': None, 'odds_str': None, 'decimal_odds': None,
         'source_book': None, 'has_market': False, 'game': game_label, 'type': 'prop'},
        {'desc': f"{hitter_name} ({team_name}) Under 0.5 Hits (model projection, no live prop line posted)",
         'model_prob': 1.0 - model_over, 'market_prob': None, 'odds_str': None, 'decimal_odds': None,
         'source_book': None, 'has_market': False, 'game': game_label, 'type': 'prop'},
    ]

# -----------------------------------------------------------------------------
# MODEL TRACK RECORD -- persistent prediction log + real-outcome settlement
# -----------------------------------------------------------------------------
# The rest of this app recomputes everything fresh, every run, from live data
# -- it never remembers what it predicted yesterday or whether that
# prediction was right. This section is a separate, honest add-on: every leg
# the model surfaces gets logged once to a local SQLite file, and once that
# game goes Final, it's settled against the REAL final score / real box-score
# strikeout-and-hit totals (pulled from the MLB Stats API, not invented). It
# rolls on a 120-day window -- anything older than that is deleted to make
# room for new days, so the file never grows without bound.
TRACK_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlb_track_record.db")
TRACK_RETENTION_DAYS = 120

# -----------------------------------------------------------------------------
# TRACK-RECORD DB PERSISTENCE ACROSS RESTARTS (GitHub-backed backup/restore)
# -----------------------------------------------------------------------------
# Streamlit Community Cloud's filesystem is ephemeral -- mlb_track_record.db
# would otherwise reset to empty every time the app redeploys or wakes up
# from sleep. Rather than rewriting the whole tracking layer onto a hosted
# database (a large, risky change touching dozens of call sites below), this
# stores the SQLite file itself as a single file in your private GitHub repo
# and syncs it down/up around it -- every query below is completely
# unchanged. Configure two secrets to turn this on:
#   GITHUB_TOKEN       -- a GitHub Personal Access Token with 'repo' scope
#   GITHUB_TRACK_REPO  -- "yourusername/your-repo-name" (the same repo you
#                         deployed the app from works fine)
# If either is missing, this silently no-ops and the app behaves exactly as
# before (local-only, resets on restart) -- never a hard failure.
import base64

def _github_track_config():
    try:
        token = st.secrets.get("GITHUB_TOKEN")
        repo = st.secrets.get("GITHUB_TRACK_REPO")
    except Exception:
        return None, None
    if not token or not repo:
        return None, None
    return token, repo

def _github_track_url(repo):
    return f"https://api.github.com/repos/{repo}/contents/mlb_track_record.db"

def restore_track_db_from_github():
    """Pulls the persisted DB down from GitHub before anything opens a local
    connection. Guarded by 'file already exists' so on a warm rerun (the
    normal case -- Streamlit reruns the whole script on every interaction)
    this is a no-op; it only does real work once, right after a fresh
    container start."""
    if os.path.exists(TRACK_DB_PATH):
        return
    token, repo = _github_track_config()
    if not token:
        return
    try:
        resp = requests.get(
            _github_track_url(repo),
            headers={"Authorization": f"token {token}", "Accept": "application/vnd.github+json"},
            timeout=15,
        )
        if resp.status_code == 200:
            with open(TRACK_DB_PATH, "wb") as f:
                f.write(base64.b64decode(resp.json()["content"]))
    except Exception:
        pass  # any failure here just means we start with a fresh local DB

def backup_track_db_to_github():
    """Pushes the current local DB back up to GitHub. Only ever called from
    inside the already-throttled (once-per-30-min) maintenance pass below --
    never on every rerun -- to stay well within GitHub's API rate limits."""
    token, repo = _github_track_config()
    if not token or not os.path.exists(TRACK_DB_PATH):
        return
    try:
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}
        get_resp = requests.get(_github_track_url(repo), headers=headers, timeout=15)
        sha = get_resp.json().get("sha") if get_resp.status_code == 200 else None
        with open(TRACK_DB_PATH, "rb") as f:
            content_b64 = base64.b64encode(f.read()).decode()
        payload = {"message": "Update MLB track-record DB", "content": content_b64}
        if sha:
            payload["sha"] = sha
        requests.put(_github_track_url(repo), headers=headers, json=payload, timeout=30)
    except Exception:
        pass  # a missed backup just means the next restart loses this cycle

restore_track_db_from_github()

def get_track_db():
    """One connection per Streamlit session is plenty for a local sqlite file;
    check_same_thread=False since Streamlit can touch it from the fragment
    thread as well as the main script thread.

    WAL mode + a busy_timeout matter a lot here: Streamlit reruns the whole
    script on almost every interaction (button clicks, widget changes, the
    15s live-scores fragment), so it's normal for more than one connection to
    the same file to exist at nearly the same moment. Without WAL + a
    timeout, SQLite's default behavior is to fail IMMEDIATELY with
    'database is locked' the instant it sees any contention at all, rather
    than just waiting the handful of milliseconds it actually takes for the
    other connection to finish. WAL lets readers and writers coexist much
    more gracefully, and the busy_timeout makes any remaining contention
    something SQLite waits out instead of raising on."""
    conn = sqlite3.connect(TRACK_DB_PATH, check_same_thread=False, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout = 30000;")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS legs (
            leg_key TEXT PRIMARY KEY,
            game_pk INTEGER,
            game_date TEXT,
            logged_at TEXT,
            leg_type TEXT,
            description TEXT,
            model_prob REAL,
            market_prob REAL,
            odds_str TEXT,
            source_book TEXT,
            resolved INTEGER DEFAULT 0,
            result TEXT,
            resolved_at TEXT
        )
    """)
    # Every moneyline pick the model makes for every game -- NOT filtered by
    # confidence or by whether it was strong enough to enter the parlay pool.
    # This is the dataset you'd actually want to judge/improve the moneyline
    # model against, since it includes the close/uncertain picks too, not
    # just the ones that already looked confident.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ml_predictions (
            pred_key TEXT PRIMARY KEY,
            game_pk INTEGER,
            game_date TEXT,
            logged_at TEXT,
            away_team TEXT,
            home_team TEXT,
            pick TEXT,
            model_prob REAL,
            has_market INTEGER,
            market_odds_str TEXT,
            implied_prob REAL,
            edge REAL,
            resolved INTEGER DEFAULT 0,
            result TEXT,
            resolved_at TEXT
        )
    """)
    # Safe migration: databases created before away_score/home_score existed
    # won't have these columns yet. SQLite has no "ADD COLUMN IF NOT EXISTS",
    # so check first and only add what's actually missing.
    _existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(ml_predictions)").fetchall()}
    if 'away_score' not in _existing_cols:
        conn.execute("ALTER TABLE ml_predictions ADD COLUMN away_score INTEGER")
    if 'home_score' not in _existing_cols:
        conn.execute("ALTER TABLE ml_predictions ADD COLUMN home_score INTEGER")
    # Every announced starting pitcher's real strikeout outcome for every
    # game -- logged whether or not a strikeout prop bet ever existed on
    # them. This is what lets you actually see whether the K/9-based
    # projection runs hot or cold, systematically, across a real sample.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pitcher_starts (
            start_key TEXT PRIMARY KEY,
            game_pk INTEGER,
            game_date TEXT,
            logged_at TEXT,
            pitcher_name TEXT,
            team TEXT,
            opponent TEXT,
            season_k_avg REAL,
            resolved INTEGER DEFAULT 0,
            actual_k INTEGER,
            resolved_at TEXT
        )
    """)
    # Small key-value table for one-off migration/repair flags -- e.g.
    # marking that the historical re-verification (see
    # reverify_resolved_ml_predictions) has already run once, so it doesn't
    # re-check the same already-confirmed-correct rows forever.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS app_metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    return conn

def build_leg_key(game_pk, leg_type, stable_label):
    """A stable identifier for one specific bet on one specific real game --
    stable meaning it does NOT include the price, since the same underlying
    bet logged an hour apart could have drifted odds, and we want that to
    still count as the same logged prediction, not a duplicate row."""
    return f"{game_pk}:{leg_type}:{stable_label}"

def log_leg_for_tracking(conn, game_pk, game_date, leg_type, stable_label, description, model_prob, market_prob, odds_str, source_book):
    """Logs one leg exactly once. INSERT OR IGNORE against the leg_key primary
    key means re-running the app (which happens constantly in Streamlit) never
    creates duplicate rows for the same bet -- the first time a leg is seen is
    the only time it gets written.

    This is a supplementary feature, not the app's core purpose -- so if the
    database is transiently unavailable (e.g. another process is mid-write
    for longer than the busy_timeout), we skip logging that one leg rather
    than letting the whole page crash. It'll simply get logged the next time
    the app runs and sees this same leg."""
    if not game_pk or not game_date:
        return
    leg_key = build_leg_key(game_pk, leg_type, stable_label)
    try:
        conn.execute(
            "INSERT OR IGNORE INTO legs "
            "(leg_key, game_pk, game_date, logged_at, leg_type, description, model_prob, market_prob, odds_str, source_book, resolved, result, resolved_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, NULL)",
            (leg_key, game_pk, game_date, datetime.now(timezone.utc).isoformat(), leg_type, description,
             model_prob, market_prob, odds_str, source_book)
        )
    except sqlite3.OperationalError:
        pass

def log_ml_prediction(conn, r):
    """Logs the model's moneyline pick for ONE game, unconditionally -- no
    confidence filter, no market-availability requirement beyond what's
    needed to grade it later. This is deliberately broader than the parlay
    leg log: it's meant to answer 'is the model's win-probability estimate
    trustworthy across everything it picks', not just the subset that
    already looked like a good bet."""
    if not r.get('game_pk') or not r.get('game_date'):
        return
    pred_key = f"{r['game_pk']}:ml_full"
    try:
        conn.execute(
            "INSERT OR IGNORE INTO ml_predictions "
            "(pred_key, game_pk, game_date, logged_at, away_team, home_team, pick, model_prob, has_market, market_odds_str, implied_prob, edge, resolved, result, resolved_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, NULL)",
            (pred_key, r['game_pk'], r['game_date'], datetime.now(timezone.utc).isoformat(),
             r['away_team'], r['home_team'], r['pick'], r['model_prob'],
             int(bool(r.get('has_ml_market'))), r.get('market_odds_str'), r.get('implied_prob'), r.get('edge'))
        )
    except sqlite3.OperationalError:
        pass

LOCK_WINDOW_SECONDS = 3600  # 1 hour before first pitch

def in_prediction_lock_window(r):
    """True once this game is within 1 hour of first pitch, or already
    live/final. Before this point, predictions are left to recalculate
    freely on every rerun (fresh injury news, lineup changes, updated
    pitcher/team stats, etc. can all still move the number, which is
    useful this far out) -- only once the window is entered does
    apply_locked_ml_prediction start freezing it."""
    if r.get('abstract_state') in ('Live', 'Final'):
        return True
    dt_str = r.get('game_datetime_utc')
    if not dt_str:
        # Can't determine start time -- fail toward locking (safer to freeze
        # too early than to never freeze a game that's actually started).
        return True
    try:
        dt_utc = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        seconds_until_start = (dt_utc - datetime.now(timezone.utc)).total_seconds()
    except Exception:
        return True
    return seconds_until_start <= LOCK_WINDOW_SECONDS

def apply_locked_ml_prediction(conn, r, min_edge, min_prob):
    """Freezes the model's moneyline pick/probability at whatever it was the
    FIRST time this game was ever logged today (log_ml_prediction is a
    no-op after that first insert, since it's INSERT OR IGNORE keyed on
    game_pk). Without this, once a game goes live, the starting pitcher's
    season ERA/WHIP/K9 -- which the MLB Stats API updates in real time as
    HE pitches THIS very game -- keeps feeding back into 'the prediction'
    for that same game, so the number drifts as a reaction to the game
    already in progress rather than staying a stable pre-game forecast.
    Market data (implied_prob/market_odds_str/edge) is intentionally left
    live -- a real sportsbook/Kalshi price moving is useful information;
    the model's own forecast drifting because of the outcome it's supposed
    to be predicting is not."""
    if not r.get('game_pk'):
        return r
    pred_key = f"{r['game_pk']}:ml_full"
    try:
        row = conn.execute(
            "SELECT pick, model_prob FROM ml_predictions WHERE pred_key = ?", (pred_key,)
        ).fetchone()
    except sqlite3.OperationalError:
        return r
    if not row:
        return r
    locked_pick, locked_model_prob = row

    r['pick'] = locked_pick
    r['model_prob'] = locked_model_prob
    r['underdog'] = r['home_team'] if locked_pick == r['away_team'] else r['away_team']
    r['fair_odds'] = prob_to_american(locked_model_prob)
    r['fair_multiplier'] = prob_to_multiplier(locked_model_prob)

    # Re-derive market odds/implied prob for whichever side is now the
    # LOCKED pick (not whatever side happened to be the freshly-computed
    # pick this rerun) -- on the rare day the model's side actually flips
    # intraday, this keeps the displayed market price matched to the real
    # locked pick instead of silently showing the other team's odds.
    market_entry = r.get('market_entry')
    if r.get('has_ml_market') and market_entry and market_entry.get('h2h'):
        h2h = market_entry['h2h']
        market_odds = h2h['home_odds'] if locked_pick == r['home_team'] else h2h['away_odds']
        implied_prob = american_to_prob(market_odds)
        r['market_odds'] = market_odds
        r['market_odds_str'] = fmt_american(market_odds)
        r['implied_prob'] = implied_prob
        r['edge'] = (locked_model_prob - implied_prob) * 100.0
        r['is_qualified'] = r['is_ready'] and (r['edge'] >= min_edge) and (locked_model_prob >= max(min_prob, 0.58))
    else:
        r['edge'] = None
        r['is_qualified'] = False
    return r

def log_pitcher_start(conn, game_pk, game_date, pitcher_name, team, opponent, season_k_avg):
    """Logs ONE starting pitcher's pre-game strikeout projection, regardless
    of whether any prop bet exists on them -- this is what makes it possible
    to later check 'does the K/9-based projection run hot or cold' across a
    real, complete sample of starts rather than just the ones that happened
    to have a posted prop line."""
    if not game_pk or not game_date or not pitcher_name or "TBD" in pitcher_name:
        return
    norm_name = normalize_player_name(pitcher_name)
    start_key = f"{game_pk}:{norm_name}"
    try:
        conn.execute(
            "INSERT OR IGNORE INTO pitcher_starts "
            "(start_key, game_pk, game_date, logged_at, pitcher_name, team, opponent, season_k_avg, resolved, actual_k, resolved_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, NULL)",
            (start_key, game_pk, game_date, datetime.now(timezone.utc).isoformat(),
             pitcher_name, team, opponent, season_k_avg)
        )
    except sqlite3.OperationalError:
        pass

@st.cache_data(ttl=900)
def fetch_final_boxscore(game_pk):
    """Real final box score for one game -- final score plus every pitcher's
    real strikeout total and every hitter's real hit total, straight from the
    MLB Stats API's live-feed endpoint (v1.1/game/{id}/feed/live). Used ONLY
    to settle already-logged predictions against what actually happened.
    Returns None if the game isn't final yet or the request fails, so
    callers just skip settling it until next time.

    This single endpoint carries BOTH the real game status
    (gameData.status.abstractGameState) and the full box score
    (liveData.boxscore), so status is verified as genuinely 'Final' before
    trusting any score -- a game caught mid-play could otherwise get graded
    on a partial snapshot and never rechecked, since settlement only looks
    at still-unresolved rows. Consolidating into one endpoint (rather than a
    separate schedule-lookup call first) removes a second network
    round-trip and a less-certain query pattern as a point of failure --
    this is the standard, unambiguously gamePk-keyed endpoint for a single
    game's full data."""
    if not game_pk:
        return None
    try:
        res = requests.get(f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live", timeout=10)
        if res.status_code != 200:
            return None
        data = res.json()
    except Exception:
        return None

    if data.get('gameData', {}).get('status', {}).get('abstractGameState') != 'Final':
        return None

    boxscore = data.get('liveData', {}).get('boxscore', {})
    result = {'away_runs': None, 'home_runs': None, 'away_team': None, 'home_team': None, 'pitcher_k': {}, 'hitter_hits': {}, 'hitter_hr': {}}
    for side in ('away', 'home'):
        team_block = boxscore.get('teams', {}).get(side, {})
        team_stats = team_block.get('teamStats', {})
        runs = team_stats.get('batting', {}).get('runs')
        if runs is not None:
            result[f'{side}_runs'] = runs
        result[f'{side}_team'] = team_block.get('team', {}).get('name')
        for player in team_block.get('players', {}).values():
            person = player.get('person', {})
            full_name = person.get('fullName')
            if not full_name:
                continue
            norm_name = normalize_player_name(full_name)
            pitching = player.get('stats', {}).get('pitching', {})
            if pitching and pitching.get('strikeOuts') is not None:
                result['pitcher_k'][norm_name] = pitching.get('strikeOuts')
            batting = player.get('stats', {}).get('batting', {})
            if batting and batting.get('hits') is not None:
                result['hitter_hits'][norm_name] = batting.get('hits')
            if batting and batting.get('homeRuns') is not None:
                result['hitter_hr'][norm_name] = batting.get('homeRuns')
    if result['away_runs'] is None and result['home_runs'] is None:
        return None
    return result

def _parse_over_under_and_point(description):
    """Pulls the (side, point) back out of a leg's own description text --
    avoids needing a second stored column just to know which side of a line
    a logged prop/total leg was on. Handles three formats:
    1. Kalshi-style totals: 'Total Runs Over {point} (YES/NO)' -- both legs
       share the identical 'Over {point}' base statement (one real contract,
       one question); the trailing YES/NO -- not the word 'Over' itself --
       is what determines the actual side. YES keeps 'Over'; NO flips to
       'Under' at the SAME point (betting NO on 'exceeds X' means the total
       stays under X).
    2. Kalshi-style props: '+N ... (YES/NO)' -- 'at least N' (YES) or
       'fewer than N' (NO), reconstructed as the equivalent Over/Under
       (N-0.5).
    3. Traditional sportsbook: '(Over|Under) <point>', no YES/NO suffix."""
    m = re.search(r'\bOver\s+([+-]?\d+(?:\.\d+)?).*?\((YES|NO)\)\s*$', description)
    if m:
        point = float(m.group(1))
        side = 'Over' if m.group(2) == 'YES' else 'Under'
        return side, point
    m = re.search(r'\+(\d+)\s+[A-Za-z ]+?\((YES|NO)\)', description)
    if m:
        threshold = int(m.group(1))
        side = 'Over' if m.group(2) == 'YES' else 'Under'
        return side, threshold - 0.5
    m = re.search(r'\b(Over|Under)\s+([+-]?\d+(?:\.\d+)?)', description)
    if m:
        return m.group(1), float(m.group(2))
    return None, None

def _parse_picked_team(leg_type, description):
    """Pulls the picked team's name back out of an ml/rl leg's own
    description -- handles both the new Kalshi-style alternate-margin format
    ('<Team> Win By X+ Runs (YES)') and the traditional run-line format
    ('<Team> +1.5 Run Line')."""
    if leg_type == 'ml':
        return description.split(' Moneyline')[0].strip()
    if leg_type == 'rl':
        m = re.match(r'^(.*?)\s+Win By\s+[\d.]+\+\s+Runs', description)
        if m:
            return m.group(1).strip()
        m = re.match(r'^(.*?)\s+[+-]?\d+(?:\.\d+)?\s+Run Line', description)
        return m.group(1).strip() if m else None
    return None

def _parse_margin_threshold(description):
    """Pulls the real run-margin required to WIN an 'rl'-type leg. The
    Kalshi-style alternate-margin format states its own threshold directly
    (e.g. 'Win By 2.5+ Runs' needs a real 3+ run margin to win, following
    standard half-point-spread convention: covering -2.5 means winning by
    more than 2.5, i.e. by 3 or more). Falls back to 2 (the real winning
    margin for a standard -1.5/+1.5 run line) for the traditional format,
    which doesn't state a threshold in its own text."""
    m = re.match(r'^.*?\s+Win By\s+([\d.]+)\+\s+Runs', description)
    if m:
        stated_margin = float(m.group(1))
        return int(np.floor(stated_margin)) + 1
    return 2

def settle_one_leg(row, boxscore):
    """Grades a single logged leg against a real final result. Returns
    'win' / 'loss' / 'push', or None if it can't be determined (e.g. a stat
    that never got matched) -- an undetermined leg is left pending rather
    than guessed at."""
    leg_type = row['leg_type']
    desc = row['description']
    away_runs, home_runs = boxscore['away_runs'], boxscore['home_runs']
    if away_runs is None or home_runs is None:
        return None

    if leg_type in ('ml', 'rl'):
        picked_team = _parse_picked_team(leg_type, desc)
        if picked_team == boxscore['home_team']:
            picked_home = True
        elif picked_team == boxscore['away_team']:
            picked_home = False
        else:
            return None  # picked team name didn't match either side -- skip, don't guess

        if leg_type == 'ml':
            if home_runs == away_runs:
                return 'push'
            won = (home_runs > away_runs) if picked_home else (away_runs > home_runs)
            return 'win' if won else 'loss'

        margin = (home_runs - away_runs) if picked_home else (away_runs - home_runs)
        needed_margin = _parse_margin_threshold(desc)
        return 'win' if margin >= needed_margin else 'loss'  # half-point spread/margin threshold -- no push possible

    if leg_type == 'total':
        side, point = _parse_over_under_and_point(desc)
        if side is None:
            return None
        actual_total = away_runs + home_runs
        if actual_total == point:
            return 'push'
        if side == 'Over':
            return 'win' if actual_total > point else 'loss'
        return 'win' if actual_total < point else 'loss'

    if leg_type == 'prop':
        side, point = _parse_over_under_and_point(desc)
        if side is None:
            return None
        # description is "{name} ({team}) Over/Under {point} Strikeouts/Hits/Home Runs"
        # or the Kalshi-style "{name} ({team}) +N Strikeouts/Hits/HR (YES/NO)"
        # -- pull the name back out to look up the real stat.
        name_part = desc.split(' (')[0]
        norm_name = normalize_player_name(name_part)
        if 'Strikeouts' in desc:
            actual = boxscore['pitcher_k'].get(norm_name)
        elif 'Home Runs' in desc or re.search(r'\bHR\b', desc):
            actual = boxscore['hitter_hr'].get(norm_name)
        elif 'Hits' in desc:
            actual = boxscore['hitter_hits'].get(norm_name)
        else:
            actual = None
        if actual is None:
            return None
        if actual == point:
            return 'push'
        if side == 'Over':
            return 'win' if actual > point else 'loss'
        return 'win' if actual < point else 'loss'

    return None

def settle_pending_legs(conn):
    """Finds every logged leg that hasn't been graded yet, checks whether
    that game is actually Final, and if so grades it against the real
    box score. Silently skips anything it can't confidently determine --
    an ungraded leg just stays pending and gets tried again next time,
    rather than ever being guessed at."""
    pending = conn.execute(
        "SELECT DISTINCT game_pk FROM legs WHERE resolved = 0 AND game_pk IS NOT NULL"
    ).fetchall()

    for (game_pk,) in pending:
        try:
            boxscore = fetch_final_boxscore(game_pk)
        except Exception:
            continue
        if not boxscore:
            continue  # not final yet (or request failed) -- try again later

        rows = conn.execute(
            "SELECT leg_key, leg_type, description FROM legs WHERE game_pk = ? AND resolved = 0",
            (game_pk,)
        ).fetchall()

        for leg_key, leg_type, description in rows:
            outcome = settle_one_leg({'leg_type': leg_type, 'description': description}, boxscore)
            if outcome is not None:
                conn.execute(
                    "UPDATE legs SET resolved = 1, result = ?, resolved_at = ? WHERE leg_key = ?",
                    (outcome, datetime.now(timezone.utc).isoformat(), leg_key)
                )
    conn.commit()

def settle_pending_predictions(conn):
    """Same idea as settle_pending_legs, but for the two unconditional logs:
    every moneyline pick (graded win/loss/push against the real final score)
    and every starting pitcher's real strikeout total (filled in from the
    same real box score) -- regardless of whether either one was ever tied
    to an actual bet."""
    pending_games = conn.execute(
        "SELECT game_pk FROM ml_predictions WHERE resolved = 0 AND game_pk IS NOT NULL "
        "UNION "
        "SELECT game_pk FROM pitcher_starts WHERE resolved = 0 AND game_pk IS NOT NULL"
    ).fetchall()

    for (game_pk,) in pending_games:
        try:
            boxscore = fetch_final_boxscore(game_pk)
        except Exception:
            continue
        if not boxscore:
            continue

        away_runs, home_runs = boxscore['away_runs'], boxscore['home_runs']
        if away_runs is not None and home_runs is not None:
            ml_rows = conn.execute(
                "SELECT pred_key, pick FROM ml_predictions WHERE game_pk = ? AND resolved = 0",
                (game_pk,)
            ).fetchall()
            for pred_key, pick in ml_rows:
                if pick == boxscore['home_team']:
                    picked_home = True
                elif pick == boxscore['away_team']:
                    picked_home = False
                else:
                    continue  # team name didn't match either side -- skip, don't guess
                if home_runs == away_runs:
                    outcome = 'push'
                else:
                    won = (home_runs > away_runs) if picked_home else (away_runs > home_runs)
                    outcome = 'win' if won else 'loss'
                conn.execute(
                    "UPDATE ml_predictions SET resolved = 1, result = ?, resolved_at = ?, away_score = ?, home_score = ? WHERE pred_key = ?",
                    (outcome, datetime.now(timezone.utc).isoformat(), away_runs, home_runs, pred_key)
                )

        p_rows = conn.execute(
            "SELECT start_key, pitcher_name FROM pitcher_starts WHERE game_pk = ? AND resolved = 0",
            (game_pk,)
        ).fetchall()
        for start_key, pitcher_name in p_rows:
            actual_k = boxscore['pitcher_k'].get(normalize_player_name(pitcher_name))
            if actual_k is None:
                continue  # box score didn't have this pitcher yet/at all -- try again later
            conn.execute(
                "UPDATE pitcher_starts SET resolved = 1, actual_k = ?, resolved_at = ? WHERE start_key = ?",
                (actual_k, datetime.now(timezone.utc).isoformat(), start_key)
            )
    conn.commit()

def reverify_resolved_ml_predictions(conn):
    """One-time data repair: fetch_final_boxscore previously had no check for
    whether a game was ACTUALLY final before trusting its score, meaning a
    prediction could get permanently marked 'resolved' based on a mid-game
    snapshot -- and since settlement only ever re-checks still-pending rows,
    any such mistake made before the fix would otherwise stay wrong forever.

    This re-checks every ALREADY-resolved win/loss prediction against the
    box score fetcher AS IT NOW WORKS (which refuses to return anything for
    a game that isn't genuinely Final), and corrects the stored result if it
    no longer matches. Runs once, ever, per database (tracked via
    app_metadata), not on every session -- there's no reason to keep
    re-checking rows that have already been confirmed correct once."""
    already_done = conn.execute(
        "SELECT value FROM app_metadata WHERE key = 'ml_predictions_reverified_v1'"
    ).fetchone()
    if already_done:
        return 0

    rows = conn.execute(
        "SELECT pred_key, game_pk, pick, result, home_team, away_team FROM ml_predictions "
        "WHERE resolved = 1 AND result IN ('win', 'loss', 'push') AND game_pk IS NOT NULL"
    ).fetchall()

    corrected = 0
    for pred_key, game_pk, pick, stored_result, home_team, away_team in rows:
        try:
            boxscore = fetch_final_boxscore(game_pk)
        except Exception:
            continue
        if not boxscore:
            # Can't confirm this one is genuinely Final right now (or the
            # request failed) -- leave the existing stored result alone
            # rather than blanking out a row we can't currently verify.
            continue

        away_runs, home_runs = boxscore['away_runs'], boxscore['home_runs']
        if away_runs is None or home_runs is None:
            continue
        if pick == boxscore['home_team']:
            picked_home = True
        elif pick == boxscore['away_team']:
            picked_home = False
        else:
            continue

        if home_runs == away_runs:
            real_result = 'push'
        else:
            won = (home_runs > away_runs) if picked_home else (away_runs > home_runs)
            real_result = 'win' if won else 'loss'

        if real_result != stored_result:
            conn.execute(
                "UPDATE ml_predictions SET result = ?, away_score = ?, home_score = ?, resolved_at = ? WHERE pred_key = ?",
                (real_result, away_runs, home_runs, datetime.now(timezone.utc).isoformat(), pred_key)
            )
            corrected += 1

    conn.execute(
        "INSERT OR REPLACE INTO app_metadata (key, value) VALUES ('ml_predictions_reverified_v1', ?)",
        (datetime.now(timezone.utc).isoformat(),)
    )
    conn.commit()
    return corrected

def enforce_retention_window(conn, retention_days=TRACK_RETENTION_DAYS):
    """Rolling window: once logged data would span more than `retention_days`,
    delete the oldest day(s) to make room. Keeps the file bounded forever
    instead of growing every single day. Applies to all three logs."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).strftime("%Y-%m-%d")
    conn.execute("DELETE FROM legs WHERE game_date < ?", (cutoff,))
    conn.execute("DELETE FROM ml_predictions WHERE game_date < ?", (cutoff,))
    conn.execute("DELETE FROM pitcher_starts WHERE game_date < ?", (cutoff,))
    conn.commit()

def run_track_record_maintenance():
    """Settlement + retention are real network/DB work -- only worth doing
    every so often, not on every single Streamlit rerun (which happens on
    every widget interaction). Session-state timestamp throttles it to once
    per 30 minutes per session.

    Wrapped defensively: if the DB is transiently locked by another
    connection despite the busy_timeout, this just skips the maintenance
    pass silently rather than crashing the page -- it'll simply try again
    next time (note the timestamp is only updated on success, so a failed
    attempt doesn't get skipped for the full 30 minutes)."""
    now = datetime.now(timezone.utc)
    last_run = st.session_state.get('track_record_last_run')
    if last_run and (now - last_run).total_seconds() < 1800:
        return
    try:
        conn = get_track_db()
        try:
            # settle_pending_legs and settle_pending_predictions each fetch a
            # real box score PER pending game, sequentially -- and unlike
            # today's slate, this runs against the ENTIRE 120-day rolling
            # history of anything not yet resolved, which can be a real
            # backlog on a fresh session. Pre-warming fetch_final_boxscore
            # here, concurrently, for every unique pending game across all
            # three tables means both settlement passes below almost always
            # hit an already-populated cache instead of a live request each.
            _pending_pks = set()
            for _row in conn.execute("SELECT DISTINCT game_pk FROM legs WHERE resolved = 0 AND game_pk IS NOT NULL").fetchall():
                _pending_pks.add(_row[0])
            for _row in conn.execute("SELECT DISTINCT game_pk FROM ml_predictions WHERE resolved = 0 AND game_pk IS NOT NULL").fetchall():
                _pending_pks.add(_row[0])
            for _row in conn.execute("SELECT DISTINCT game_pk FROM pitcher_starts WHERE resolved = 0 AND game_pk IS NOT NULL").fetchall():
                _pending_pks.add(_row[0])

            # The one-time historical repair (see reverify_resolved_ml_predictions)
            # re-checks every ALREADY-resolved prediction too -- only relevant the
            # very first time this runs on a given database, so its game_pks are
            # only added to this same parallel batch while that repair is still
            # pending, not on every subsequent session.
            _repair_pending = not conn.execute(
                "SELECT 1 FROM app_metadata WHERE key = 'ml_predictions_reverified_v1'"
            ).fetchone()
            if _repair_pending:
                for _row in conn.execute(
                    "SELECT DISTINCT game_pk FROM ml_predictions WHERE resolved = 1 AND game_pk IS NOT NULL"
                ).fetchall():
                    _pending_pks.add(_row[0])

            if _pending_pks:
                _bw_count = max(16, min(64, len(_pending_pks) * 3))
                with ThreadPoolExecutor(max_workers=_bw_count) as _bw_executor:
                    _bw_futures = [_bw_executor.submit(_safe_call, fetch_final_boxscore, pk) for pk in _pending_pks]
                    for _bf in as_completed(_bw_futures):
                        pass  # results discarded -- this call's only job is to populate the shared cache

            settle_pending_legs(conn)
            settle_pending_predictions(conn)
            if _repair_pending:
                _num_corrected = reverify_resolved_ml_predictions(conn)
                if _num_corrected > 0:
                    st.session_state['track_record_repair_message'] = (
                        f"🔧 One-time data repair: found and corrected {_num_corrected} previously "
                        f"mis-graded prediction(s) that had been settled from a mid-game snapshot "
                        f"instead of the true final score."
                    )
            enforce_retention_window(conn)
        finally:
            conn.close()
    except sqlite3.OperationalError:
        return
    st.session_state['track_record_last_run'] = now
    backup_track_db_to_github()


st.markdown("<h1 class='main-title'>⚾ Advanced MLB Algorithmic Analytics Engine</h1>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Every bet shown is tied to a real, currently-posted sportsbook line -- with Model Probability and Market Probability shown side by side.</div>", unsafe_allow_html=True)

st.sidebar.markdown("### 🎯 Market Source")
selected_bookmaker = st.sidebar.selectbox(
    "Preferred Bookmaker / Exchange",
    options=["Kalshi", "DraftKings", "FanDuel", "BetMGM"],
    index=0
)
bookmaker_key_map = {
    "DraftKings": "draftkings",
    "FanDuel": "fanduel",
    "BetMGM": "betmgm",
    "Kalshi": "kalshi"
}
chosen_bm_key = bookmaker_key_map[selected_bookmaker]
# When Kalshi is the selected exchange, every fetch below goes strict --
# a leg only shows up if Kalshi itself actually priced it, never silently
# substituted from DraftKings/FanDuel/etc. just because Kalshi didn't have
# that particular market yet.
strict_kalshi = (chosen_bm_key == 'kalshi')

def get_saved_odds_api_key():
    """Looks for a locally-saved key so you don't have to paste it in every
    session. Checks, in order:
      1. .streamlit/secrets.toml -> ODDS_API_KEY = "..."
      2. an environment variable named ODDS_API_KEY
    Returns "" if neither is set, in which case the sidebar will ask for one
    manually as a fallback."""
    try:
        if "ODDS_API_KEY" in st.secrets:
            return st.secrets["ODDS_API_KEY"]
    except Exception:
        pass
    return os.environ.get("ODDS_API_KEY", "")

saved_odds_api_key = get_saved_odds_api_key()

st.sidebar.markdown("---")
enable_odds_api = st.sidebar.toggle(
    "🔌 Enable Odds API (Live Market Data)",
    value=True,
    help="Turn OFF to stop every Odds API call app-wide instantly -- bulk moneyline/run "
         "line/total odds, player props, and Kalshi alternate-spread ladders all stop "
         "hitting the API. Useful when you're close to your monthly call limit. While "
         "off, you'll still see the model's own win-probability predictions -- just no "
         "live market odds or edge comparison, since there's no market data to compare against."
)

# Clear the Odds-API-specific caches the instant this switch flips (either
# direction), so turning it back ON always makes one genuinely fresh call
# right now rather than possibly serving a result cached from before the
# flip (fetch_bulk_market_odds/etc. cache for 10 minutes -- without this,
# switching on partway through that window would silently reuse the old
# cached data instead of actually calling the API). This does NOT touch
# unrelated caches (schedule, standings, pitcher stats, etc.) -- only the
# three functions that actually hit The Odds API.
if 'prev_enable_odds_api' not in st.session_state:
    st.session_state['prev_enable_odds_api'] = enable_odds_api
if st.session_state['prev_enable_odds_api'] != enable_odds_api:
    fetch_bulk_market_odds.clear()
    fetch_event_player_props.clear()
    fetch_event_alternate_spreads.clear()
    st.session_state['prev_enable_odds_api'] = enable_odds_api

if enable_odds_api:
    if st.sidebar.button("🔄 Refresh Odds Now", help="Clears the Odds API cache and makes a fresh call immediately, instead of waiting for the normal 10-minute cache to expire on its own."):
        fetch_bulk_market_odds.clear()
        fetch_event_player_props.clear()
        fetch_event_alternate_spreads.clear()
        st.rerun()

if not enable_odds_api:
    st.sidebar.warning("🔌 Odds API calls are OFF. No API credits are being used right now.")

if saved_odds_api_key:
    odds_api_key = saved_odds_api_key
    st.sidebar.success("✅ Odds API key loaded automatically (from secrets.toml / environment).")
else:
    odds_api_key = st.sidebar.text_input(
        "The-Odds-API Key", type="password",
        help="Required to show any bets. Without a key, this app will not invent odds -- it will only show model win-probabilities for context. Tip: save it in .streamlit/secrets.toml so you don't need to re-enter it each session."
    )

# Every fetch_bulk_market_odds / fetch_event_*_odds call site already treats
# an empty api_key as "skip the call, return {} / no_key status" -- so
# routing an empty string through here when the switch is off is a single
# clean kill-switch for the ENTIRE Odds API surface, without needing to
# thread an extra flag through every call site individually.
effective_odds_api_key = odds_api_key if enable_odds_api else ""

st.sidebar.markdown("---")
st.sidebar.markdown("### 🗓️ Model & Scope Controls")
selected_date = st.sidebar.date_input("Game Date", datetime.now())
date_str = selected_date.strftime("%Y-%m-%d")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ High-Confidence Bet Filters")
min_edge_threshold = st.sidebar.slider("Min Value Edge (%):", min_value=1.0, max_value=10.0, value=5.0, step=0.5)
min_win_prob_threshold = st.sidebar.slider("Min Model Win Prob (%):", min_value=58.0, max_value=90.0, value=65.0, step=1.0) / 100.0

strict_lineup_mode = st.sidebar.toggle(
    "Strict Pitcher Announcement Gate",
    value=True,
    help="Blocks predictions/picks for pre-game matches where starting pitchers are unannounced."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Kalshi Compatibility")
st.sidebar.caption(
    "Always on: run lines are built from Kalshi's real 'team wins by X+ runs' "
    "ladder of contracts (1.5, 2.5, 3.5+, etc.) rather than a traditional "
    "-1.5/+1.5 line, whenever Kalshi is the selected exchange."
)
kalshi_mode = True

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎲 Player Props (Optional)")
include_props = st.sidebar.toggle(
    "Show real pitcher K / batter hit / batter HR prop lines for the selected matchup",
    value=False,
    help="Uses 1 extra API credit for the game you're viewing in the Deep-Dive tab."
)
include_props_in_parlay = False
if include_props:
    include_props_in_parlay = st.sidebar.checkbox(
        "Also use real prop lines in the parlay builder",
        value=False,
        help="Uses 1 extra API credit PER GAME in today's slate -- can add up quickly on a full slate."
    )

games = fetch_mlb_schedule(date_str)
market_map, market_odds_diagnostic = fetch_bulk_market_odds(effective_odds_api_key, chosen_bm_key, strict=strict_kalshi)

if not enable_odds_api:
    st.info(
        "🔌 **Odds API is turned off** (sidebar toggle). Showing the model's own win-probability "
        "predictions only -- no live market odds, edge comparison, or parlay builder until it's "
        "switched back on."
    )
elif market_odds_diagnostic['status'] not in ('ok',):
    _diag_status = market_odds_diagnostic['status']
    _diag_detail = market_odds_diagnostic['detail']
    if _diag_status == 'http_error' and market_odds_diagnostic.get('status_code') in (401, 403):
        st.error(
            f"🔑 **The Odds API rejected the request (HTTP {market_odds_diagnostic['status_code']}):** {_diag_detail} "
            f"This means moneylines, totals, run lines, and props will all be unavailable until this is fixed -- "
            f"double-check the key in `secrets.toml` / environment, and confirm it's still active on "
            f"[the-odds-api.com](https://the-odds-api.com)."
        )
    elif _diag_status == 'http_error' and market_odds_diagnostic.get('status_code') == 429:
        st.error(
            f"📊 **The Odds API quota has been exceeded (HTTP 429):** {_diag_detail} "
            f"The free tier resets monthly -- check your usage at the-odds-api.com, or wait for the reset."
        )
    elif _diag_status == 'no_key':
        st.warning(f"⚠️ {_diag_detail} Add `odds_api_key` to `secrets.toml` or your environment to enable real odds.")
    else:
        st.warning(f"⚠️ **No market odds loaded:** {_diag_detail}")

if not games:
    st.info(f"No MLB games scheduled or found for date: {date_str}.")
    st.stop()

if not odds_api_key:
    st.error(
        "🔑 **No Odds API key entered.** This app will not invent sportsbook lines, so no bets or "
        "market probabilities can be shown right now. Model win-probabilities below are still viewable "
        "for informational context. Add a free key from the-odds-api.com in the sidebar to unlock real "
        "lines, market probabilities, and the parlay builder."
    )
elif not market_map and enable_odds_api:
    st.warning(
        "⚠️ Live odds could not be retrieved right now (API limit, no games currently priced by any "
        "tracked book, or a network issue). Recommendations below will only include games where a real "
        "market line comes back."
    )

season = date_str.split('-')[0]
standings_map = get_league_standings(season)
ops_leaders_map = get_league_ops_leaders(season)

with st.spinner(f"Loading real-time stats for {len(games)} game(s) on {date_str}... (fetching team/pitcher data in parallel)"):
    # Every game below independently triggers several REAL, live network calls
    # per team AND per starting pitcher (run environment, fielding, platoon
    # splits, key hitter, pitcher season stats, pitcher rest days) -- each
    # individually fast, but run one-at-a-time across a full ~15-game slate on
    # a cold cache, that's 150+ sequential blocking requests, which is what
    # was actually causing the page to sit "stuck" on load for minutes at a
    # time. Pre-warming the SAME @st.cache_data-backed functions here,
    # concurrently, for every unique team AND every unique starting pitcher in
    # today's slate means the per-game loop right below almost always hits an
    # already-populated cache instead of a live request.
    #
    # Every INDIVIDUAL call below is submitted as its own task, rather than
    # bundling a team's 5 calls (or a pitcher's 2) into one sequential chain
    # -- bundling them left calls inside the same chain waiting on each other
    # even though they're unrelated, which under-used the worker pool.
    # Flattened like this, all ~200+ calls compete directly for a worker slot.
    #
    # This entire block runs inside st.spinner() specifically because
    # Streamlit's normal "Running <function>(...)" indicator only tracks
    # calls made directly on the main script thread -- once work moves into
    # background worker threads (exactly what makes this fast), that
    # indicator no longer has anything to show, so without an explicit
    # spinner here the page would look blank while very much still working.
    _unique_team_ids = set()
    _unique_pitchers = set()  # (pitcher_id, pitcher_name, team_name) tuples
    for _g in games:
        if _g.get('away_team_id'):
            _unique_team_ids.add(_g['away_team_id'])
        if _g.get('home_team_id'):
            _unique_team_ids.add(_g['home_team_id'])
        if _g.get('away_pitcher_id'):
            _unique_pitchers.add((_g['away_pitcher_id'], _g.get('away_pitcher', ''), _g.get('away_team', '')))
        if _g.get('home_pitcher_id'):
            _unique_pitchers.add((_g['home_pitcher_id'], _g.get('home_pitcher', ''), _g.get('home_team', '')))

    def _safe_call(fn, *args):
        try:
            fn(*args)
        except Exception:
            pass  # a single call failing just means that one lookup falls back to a live request later, not a crash

    if _unique_team_ids or _unique_pitchers:
        # 3 workers per unique team/pitcher target (rounded, capped) gives the
        # pool enough headroom that essentially every call can run at once on
        # a typical ~15-game slate (usually well under 100 concurrent sockets
        # to a single well-provisioned public API), while still capping it so
        # an unusually large slate doesn't try to open hundreds of
        # connections at once.
        _target_count = len(_unique_team_ids) + len(_unique_pitchers)
        _worker_count = max(16, min(64, _target_count * 3))
        with ThreadPoolExecutor(max_workers=_worker_count) as _executor:
            _futures = []
            for tid in _unique_team_ids:
                _futures.append(_executor.submit(_safe_call, get_team_run_environment, tid, season))
                _futures.append(_executor.submit(_safe_call, get_team_fielding, tid, season))
                _futures.append(_executor.submit(_safe_call, get_team_platoon_ops, tid, season, 'L'))
                _futures.append(_executor.submit(_safe_call, get_team_platoon_ops, tid, season, 'R'))
                _by_team = ops_leaders_map.get('by_team', {}) if isinstance(ops_leaders_map, dict) else {}
                _entry = _by_team.get(tid)
                if _entry and _entry.get('person_id'):
                    # Must match exactly what get_key_hitter_for_team actually
                    # calls (get_player_hitting_stats), not a different cached
                    # function -- pre-warming the wrong function's cache key
                    # warms nothing.
                    _futures.append(_executor.submit(_safe_call, get_player_hitting_stats, _entry['person_id'], season))
            for pid, pname, tname in _unique_pitchers:
                _futures.append(_executor.submit(_safe_call, get_real_pitcher_stats, pid, pname, tname))
                _futures.append(_executor.submit(_safe_call, get_pitcher_rest_days, pid, date_str, season))
            for _f in as_completed(_futures):
                pass  # results are discarded -- this call's only job is to populate the shared cache

    ml_results = [model_moneyline_game(g, market_map, min_edge_threshold, min_win_prob_threshold, strict_lineup_mode, season, standings_map, ops_leaders_map) for g in games]

# Lock in each game's moneyline pick/probability once it enters its 1-hour-
# before-first-pitch window (see in_prediction_lock_window /
# apply_locked_ml_prediction) before anything below -- sorting, qualifying,
# parlay-pool building -- reads model_prob/edge/is_qualified. Outside that
# window, predictions are left to recalculate freely on every rerun.
_pred_conn = get_track_db()
for r in ml_results:
    if r['is_ready'] and r.get('game_pk') and in_prediction_lock_window(r):
        _ = log_ml_prediction(_pred_conn, r)
        apply_locked_ml_prediction(_pred_conn, r, min_edge_threshold, min_win_prob_threshold)
try:
    _pred_conn.commit()
except sqlite3.OperationalError:
    pass

ml_results_sorted = sorted(ml_results, key=lambda x: (x['edge'] if x['edge'] is not None else -999), reverse=True)
qualified_picks = [r for r in ml_results_sorted if r['is_qualified']]
final_games = [r for r in ml_results_sorted if r['abstract_state'] == 'Final']

# Log every announced starting pitcher for the WHOLE slate, unconditionally
# -- no confidence threshold, no parlay-pool filtering. This is the
# complete, unbiased dataset for checking whether the K/9-based projection
# runs hot or cold. (Moneyline picks are already logged/locked above.)
for r in ml_results_sorted:
    if not r['is_ready'] or not r.get('game_pk'):
        continue
    if "TBD" not in r['away_pitcher']:
        _ = log_pitcher_start(_pred_conn, r['game_pk'], r['game_date'], r['away_pitcher'], r['away_team'], r['home_team'], r['away_stats']['k_avg'])
    if "TBD" not in r['home_pitcher']:
        _ = log_pitcher_start(_pred_conn, r['game_pk'], r['game_date'], r['home_pitcher'], r['home_team'], r['away_team'], r['home_stats']['k_avg'])
    try:
        _pred_conn.commit()
    except sqlite3.OperationalError:
        pass
try:
    _pred_conn.close()
except sqlite3.OperationalError:
    pass

# -----------------------------------------------------------------------------
# SECTION 1: LIVE IN-PROGRESS GAMES & TRACKING
# -----------------------------------------------------------------------------
@st.cache_data(ttl=10)
def fetch_live_pitcher_strikeouts(game_pk, away_pitcher_id, home_pitcher_id):
    """Live in-game strikeout count AND still-pitching status for each
    STARTING pitcher, straight from MLB's free boxscore endpoint (no API
    key/credit involved -- separate from the rate-limited Odds API
    entirely). Cached at the same 10s interval as the live ticker itself,
    so this doesn't add extra calls beyond what the ticker already does.

    Returns (away_k, home_k, away_still_pitching, home_still_pitching).
    K counts can be None if that pitcher doesn't have a stat line yet (game
    just started) or has already been pulled (in which case it's his final
    strikeout total for the outing -- correct, since it just stops climbing
    once he's out). still_pitching is None if it can't be determined, else
    True/False based on whether he's the last name in that team's pitching
    order so far (MLB's boxscore lists pitchers in the order they
    appeared -- the last one in that list is whoever's actually on the
    mound right now)."""
    if not game_pk:
        return None, None, None, None
    try:
        url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
        resp = requests.get(url, timeout=8)
        data = resp.json()
    except Exception:
        return None, None, None, None

    def _find_k(team_key, pitcher_id):
        if not pitcher_id:
            return None
        players = data.get('teams', {}).get(team_key, {}).get('players', {})
        entry = players.get(f"ID{pitcher_id}")
        if not entry:
            return None
        return entry.get('stats', {}).get('pitching', {}).get('strikeOuts')

    def _still_pitching(team_key, pitcher_id):
        if not pitcher_id:
            return None
        pitchers_list = data.get('teams', {}).get(team_key, {}).get('pitchers', [])
        if not pitchers_list:
            return None
        return pitchers_list[-1] == pitcher_id

    return (
        _find_k('away', away_pitcher_id),
        _find_k('home', home_pitcher_id),
        _still_pitching('away', away_pitcher_id),
        _still_pitching('home', home_pitcher_id),
    )

def render_scorechip_html(r, away_k=None, home_k=None, away_still_pitching=None, home_still_pitching=None):
    """Builds one broadcast-style live-game scoreboard card: a title header,
    a LIVE + inning status row, glove-flanked score columns with the live
    base-runner diamond as the center icon, and B/S/O dot rows -- all real,
    live data from the same game result dict already used elsewhere
    (r['balls']/['strikes']/['outs']/['current_inning']/['inning_state']/
    ['on_first']/['on_second']/['on_third'] all come straight from the MLB
    Stats API linescore)."""
    away_city = team_city_display(r['away_team'])
    home_city = team_city_display(r['home_team'])
    away_abbr = team_abbr(r['away_team'])
    home_abbr = team_abbr(r['home_team'])

    inning_num = r.get('current_inning') or ''
    inning_state = r.get('inning_state') or ''
    inning_arrow = "▲" if inning_state == "Top" else ("▼" if inning_state == "Bottom" else "")
    inning_display = ordinal_inning(inning_num) if inning_num else r['detailed_state']

    # Doubleheaders: two real, separate games between the same two teams on
    # the same date, each with its own gamePk/score/state. Without this label,
    # they look like one contradictory game (e.g. "Live, Top 8" AND "Final"
    # for what looks like the same matchup) since only team names show.
    doubleheader_label = f" (Gm {r['game_number']})" if r.get('is_doubleheader') else ""
    title_label = f"{away_city} vs {home_city}{doubleheader_label}"

    balls = max(0, min(3, r.get('balls', 0) or 0))
    strikes = max(0, min(3, r.get('strikes', 0) or 0))
    outs = max(0, min(3, r.get('outs', 0) or 0))

    def shape_row(filled_cls, empty_cls, filled_count, total):
        cells = []
        for i in range(total):
            state_cls = filled_cls if i < filled_count else empty_cls
            cells.append(f'<span class="scorechip-shape {state_cls}"></span>')
        return "".join(cells)

    balls_row = shape_row("scorechip-filled-ball", "scorechip-empty-ball", balls, 4)
    strikes_row = shape_row("scorechip-filled-strike", "scorechip-empty-strike", strikes, 3)
    outs_row = shape_row("scorechip-filled-out", "scorechip-empty-out", outs, 3)

    first_fill = "#FBBF24" if r.get('on_first') else "#0F172A"
    second_fill = "#FBBF24" if r.get('on_second') else "#0F172A"
    third_fill = "#FBBF24" if r.get('on_third') else "#0F172A"

    # A simple infield diamond outline (home at bottom, 2nd at top, 1st/3rd
    # at the sides) drawn as plain SVG geometry -- not a borrowed image --
    # with the same live on/off base state layered on top, used here as the
    # center icon between the two scores (doubling as a real baserunner
    # indicator rather than a purely decorative graphic).
    bases_html = (
        f'<svg class="scorechip-diamond-svg" viewBox="0 0 60 54" width="40" height="36">'
        f'<path d="M30,50 L52,28 L30,6 L8,28 Z" fill="none" stroke="#334155" stroke-width="2"/>'
        f'<rect x="26" y="2" width="8" height="8" fill="{second_fill}" stroke="#475569" stroke-width="1.2" transform="rotate(45 30 6)"/>'
        f'<rect x="4" y="24" width="8" height="8" fill="{third_fill}" stroke="#475569" stroke-width="1.2" transform="rotate(45 8 28)"/>'
        f'<rect x="48" y="24" width="8" height="8" fill="{first_fill}" stroke="#475569" stroke-width="1.2" transform="rotate(45 52 28)"/>'
        f'</svg>'
    )

    away_k_display = f"{away_k} K" if away_k is not None else "—"
    home_k_display = f"{home_k} K" if home_k is not None else "—"
    away_sp_last = pitcher_last_name(r["away_pitcher"])
    home_sp_last = pitcher_last_name(r["home_pitcher"])

    def _status_dot(still_pitching):
        if still_pitching is None:
            return ''
        color = '#10B981' if still_pitching else '#64748B'  # green = still in, gray = pulled
        label = 'Still pitching' if still_pitching else 'Pulled from game'
        return f'<span class="scorechip-pitcher-dot" style="background:{color};" title="{label}"></span>'

    away_dot = _status_dot(away_still_pitching)
    home_dot = _status_dot(home_still_pitching)
    pitcher_k_row = (
        f'<div class="scorechip-pitcher-k-row">'
        f'<span title="{r["away_pitcher"]}">{away_dot}{away_sp_last}: {away_k_display}</span>'
        f'<span class="scorechip-pitcher-k-sep">|</span>'
        f'<span title="{r["home_pitcher"]}">{home_dot}{home_sp_last}: {home_k_display}</span>'
        f'</div>'
    )

    return (
        f'<div class="scorechip">'
        f'<div class="scorechip-title">{title_label}</div>'
        f'<div class="scorechip-status-row">'
        f'<span class="scorechip-live-dot"></span><span class="scorechip-live-text">LIVE</span>'
        f'<span class="scorechip-inning-arrow">{inning_arrow}</span>'
        f'<span class="scorechip-inning-text">{inning_display}</span>'
        f'<span class="scorechip-info-icon" title="{r["time_ct"]}">ⓘ</span>'
        f'</div>'
        f'<div class="scorechip-score-row">'
        f'<div class="scorechip-team-col">'
        f'{team_logo_html(r.get("away_team_id"))}'
        f'<div class="scorechip-score-num">{r["away_score"]}</div>'
        f'<div class="scorechip-team-abbr">{away_abbr}</div>'
        f'</div>'
        f'{bases_html}'
        f'<div class="scorechip-team-col">'
        f'{team_logo_html(r.get("home_team_id"))}'
        f'<div class="scorechip-score-num">{r["home_score"]}</div>'
        f'<div class="scorechip-team-abbr">{home_abbr}</div>'
        f'</div>'
        f'</div>'
        f'<div class="scorechip-count-row">'
        f'<div class="scorechip-count-group"><span class="scorechip-label">B</span>{balls_row}</div>'
        f'<div class="scorechip-count-group"><span class="scorechip-label">S</span>{strikes_row}</div>'
        f'<div class="scorechip-count-group"><span class="scorechip-label">O</span>{outs_row}</div>'
        f'</div>'
        f'{pitcher_k_row}'
        f'<div class="scorechip-pick">👉 {r["pick"]} ({r["model_prob"]*100:.1f}%) | {r["market_odds_str"]}</div>'
        f'</div>'
    )

@st.fragment(run_every=10)
def render_live_scores_section(current_date, current_market_map, min_edge, min_prob, strict_mode):
    st.markdown("### 🔴 Live In-Progress Games & Ball-to-Ball Tracking")

    current_season = current_date.split('-')[0]
    current_standings_map = get_league_standings(current_season)
    current_ops_leaders_map = get_league_ops_leaders(current_season)

    current_games = fetch_mlb_schedule(current_date)
    current_ml_results = [model_moneyline_game(g, current_market_map, min_edge, min_prob, strict_mode, current_season, current_standings_map, current_ops_leaders_map) for g in current_games]
    # Same lock as the main slate list -- without it, this ticker (which
    # reruns every 15 seconds on its own, no click needed) is exactly where
    # a live-drifting pick/probability would be most visible, since the
    # pitcher currently on the mound updates his own season stats mid-game.
    _live_conn = get_track_db()
    for r in current_ml_results:
        if r['is_ready'] and r.get('game_pk') and in_prediction_lock_window(r):
            _ = log_ml_prediction(_live_conn, r)
            apply_locked_ml_prediction(_live_conn, r, min_edge, min_prob)
    try:
        _live_conn.commit()
    except sqlite3.OperationalError:
        pass
    # abstractGameState flips to 'Live' a little early for some games --
    # during warmup / pre-game ceremonies / a delayed start before the first
    # pitch is actually thrown -- which is why a scorechip could show up
    # reading "Top 1" for a game that hasn't really started. detailedState
    # carries the finer-grained phase, so explicitly exclude the known
    # pre-first-pitch phases even when abstractGameState already says Live.
    def is_actually_in_progress(r):
        if r['abstract_state'] != 'Live':
            return False
        ds = (r.get('detailed_state') or '').strip()
        if ds in ('Scheduled', 'Pre-Game', 'Warmup'):
            return False
        if ds.startswith('Delayed Start') or ds.startswith('Pre-Game'):
            return False
        return True

    current_live_games = [r for r in current_ml_results if is_actually_in_progress(r)]

    if current_live_games:
        chips_html = "".join([
            render_scorechip_html(
                r,
                *fetch_live_pitcher_strikeouts(r['game_pk'], r.get('away_pitcher_id'), r.get('home_pitcher_id'))
            )
            for r in current_live_games
        ])
        st.markdown(f'<div class="scorechip-grid">{chips_html}</div>', unsafe_allow_html=True)
    else:
        st.info("ℹ️ No games are currently live right now for this date.")

render_live_scores_section(
    date_str,
    market_map,
    min_edge_threshold,
    min_win_prob_threshold,
    strict_lineup_mode
)

st.markdown("---")

# -----------------------------------------------------------------------------
# SECTION 2: FINAL GAMES SCORES TABLE
# -----------------------------------------------------------------------------
st.markdown("### 🏁 Final / Ended Games Scores")
if final_games:
    final_table_data = []
    for r in final_games:
        if r['away_score'] > r['home_score']:
            actual_winner = r['away_team']
        elif r['home_score'] > r['away_score']:
            actual_winner = r['home_team']
        else:
            actual_winner = "Tie"
        dh_label = f" (Gm {r['game_number']})" if r.get('is_doubleheader') else ""
        final_table_data.append({
            "Time (CT)": r['time_ct'],
            "Matchup": f"{r['game']}{dh_label}",
            "Final Score": f"{r['away_team']} {r['away_score']} - {r['home_score']} {r['home_team']}",
            "Status": r['detailed_state'],
            "Actual Winner": f"🏆 {actual_winner}",
            "Algorithmic Pick": f"👉 {r['pick']} ({r['model_prob']*100:.1f}%)"
        })
    # Explicit height sized to fit every row -- st.dataframe defaults to a
    # fixed viewport height with its own internal scrollbar once there are
    # more than ~10 rows, which is exactly what forced scrolling to see the
    # rest of the table. Sizing height to the actual row count means the
    # whole table renders at once instead.
    final_df = pd.DataFrame(final_table_data)
    table_height = (len(final_df) + 1) * 35 + 3
    st.dataframe(final_df, use_container_width=True, hide_index=True, height=table_height)
else:
    st.info("ℹ️ No final games recorded yet for this date.")

st.markdown("---")

# -----------------------------------------------------------------------------
# SECTION 3: QUALIFIED VALUE PICKS
# -----------------------------------------------------------------------------
st.markdown("### 🔥 Today's High-Confidence Qualified Value Picks")
if qualified_picks:
    top = qualified_picks[0]
    st.success(
        f"🎯 **Best Qualified Bet:** **{top['pick']}** at **{top['time_ct']}** "
        f"| Model Prob: **{top['model_prob']*100:.1f}%** | Market Odds: **{top['market_odds_str']}** "
        f"(Market Prob: **{top['implied_prob']*100:.1f}%**) via {top['source_book']}"
    )
else:
    st.warning(f"⚠️ **No games meet your strict criteria today** (Edge ≥ +{min_edge_threshold:.1f}%, Moneyline Win Chance ≥ 58%, a real market line available, and Pitcher Gate met).")

st.markdown("---")

# -----------------------------------------------------------------------------
# SECTION 4: FULL SLATE TABLE
# -----------------------------------------------------------------------------
# Finished games already have their own dedicated table above (Section 2:
# "Final / Ended Games Scores"), so they're excluded here to avoid showing
# the same game twice -- this table is meant for what's still upcoming or
# actively in progress.
non_final_results = [r for r in ml_results_sorted if r['abstract_state'] != 'Final']
st.markdown(
    f"### 🏆 Full Slate Predictions & Starting Pitchers ({selected_bookmaker}) — "
    f"{len(non_final_results)} of {len(games)} Games (Upcoming/Live)"
)

def short_name(full_name):
    """Short form of a team/pitcher name (e.g. 'Los Angeles Dodgers' -> 'Dodgers',
    'Boston Red Sox' -> 'Red Sox'), used to keep table columns narrow enough
    to avoid horizontal scrolling."""
    return team_short_name(full_name)

ml_table_data = []
ml_table_data_full = []
for idx, r in enumerate(non_final_results, 1):
    if not r['is_ready']:
        status_tag = "🛑 Withheld"
        status_tag_full = f"🛑 WITHHELD ({r['warning_reason']})"
    elif not r['has_ml_market']:
        status_tag = "⚪ No Mkt"
        status_tag_full = "⚪ NO MARKET DATA"
    else:
        status_tag = "✅ Qual" if r['is_qualified'] else "🛑 Pass"
        status_tag_full = "✅ QUALIFIED" if r['is_qualified'] else "🛑 PASS"

    if r['abstract_state'] == 'Live':
        score_display = f"🔴 {r['away_score']}-{r['home_score']}"
        score_display_full = f"🔴 {r['away_score']} - {r['home_score']} ({r['detailed_state']})"
    elif r['abstract_state'] == 'Final':
        score_display = f"🏁 {r['away_score']}-{r['home_score']}"
        score_display_full = f"🏁 {r['away_score']} - {r['home_score']} (Final)"
    else:
        score_display = r['time_ct']
        score_display_full = r['detailed_state']

    away_p_short = short_name(r['away_pitcher']) if "TBD" not in r['away_pitcher'] else "TBD"
    home_p_short = short_name(r['home_pitcher']) if "TBD" not in r['home_pitcher'] else "TBD"

    dh_tag = f" (Gm {r['game_number']})" if r.get('is_doubleheader') else ""

    ml_table_data.append({
        "#": idx,
        "Status": status_tag,
        "Matchup": f"{short_name(r['away_team'])} @ {short_name(r['home_team'])}{dh_tag}",
        "Pitchers": f"{away_p_short} vs {home_p_short}",
        "Score / Time": score_display,
        "Pick (Model %)": f"👉 {short_name(r['pick'])} ({r['model_prob']*100:.1f}%)",
        "Market (Book)": f"{r['market_odds_str']} ({r['source_book']})" if r['has_ml_market'] else "No Market",
        "Edge (pp)": f"{r['edge']:+.1f}" if r['edge'] is not None else "—",
    })

    ml_table_data_full.append({
        "#": idx,
        "Gate Status": status_tag_full,
        "Time (CT)": r['time_ct'],
        "Matchup": f"{r['game']}{dh_tag}",
        "Starting Pitchers": f"Away: {r['away_pitcher']} vs Home: {r['home_pitcher']}",
        "Score / State": score_display_full,
        "Algorithmic Pick": f"👉 {r['pick']}",
        "Model Win Prob (%)": f"{r['model_prob']*100:.1f}%",
        "Market Odds": r['market_odds_str'],
        "Market Implied Prob (%)": f"{r['implied_prob']*100:.1f}%" if r['has_ml_market'] else "—",
        "Edge (pp)": f"{r['edge']:+.1f}" if r['edge'] is not None else "—",
        "Book": r['source_book'] if r['has_ml_market'] else "—",
        "Proj. Run Diff": f"{r['projected_run_diff']:.2f}",
    })

st.dataframe(pd.DataFrame(ml_table_data_full), use_container_width=True, hide_index=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# SECTION 5: DEEP DIVE SELECTOR
# -----------------------------------------------------------------------------
st.markdown("### 🎯 Live Game & Advanced Variables Deep-Dive Selector")
game_options = [
    f"{r['game']} | Starters: {r['away_pitcher']} vs {r['home_pitcher']} ({r['time_ct']})"
    for r in ml_results_sorted
]

selected_game_label = st.selectbox(
    "Choose Matchup:",
    options=game_options,
    key="game_selector_dropdown"
)

selected_idx = game_options.index(selected_game_label)
selected_game_item = ml_results_sorted[selected_idx]
selected_game = next(g for g in games if f"{g['away_team']} @ {g['home_team']}" == selected_game_item['game'])

st.markdown("---")

# -----------------------------------------------------------------------------
# SECTION 6: RECOMMENDED BETS, PROPS & METRICS (real lines only)
# -----------------------------------------------------------------------------
st.markdown(f"### 📊 Advanced Deep-Dive Analysis: {selected_game['away_team']} @ {selected_game['home_team']} ({selected_game['game_time_ct']})")

gate_status_text = "Passed" if selected_game_item['is_ready'] else f"Withheld ({selected_game_item['warning_reason']})"
aq = selected_game_item['away_quality']
hq = selected_game_item['home_quality']
away_quality_text = f"{aq['win_pct']*100:.1f}% win rate" + (f", {aq['pythag_pct']*100:.1f}% Pythagorean" if aq['pythag_pct'] is not None else "") + f" ({aq['games']} G)"
home_quality_text = f"{hq['win_pct']*100:.1f}% win rate" + (f", {hq['pythag_pct']*100:.1f}% Pythagorean" if hq['pythag_pct'] is not None else "") + f" ({hq['games']} G)"
st.info(
    f"📊 **{selected_game['away_team']}:** {away_quality_text} &nbsp;|&nbsp; **{selected_game['home_team']}:** {home_quality_text}\n\n"
    f"🏟️ **Park Factor:** {selected_game_item['park_factor']:.2f} (100 = neutral; approximate multi-year reference, not this exact season's precise number) | "
    f"⚖️ **Projected Run Differential:** {selected_game_item['projected_run_diff']:.2f} runs | "
    f"🛑 **Gate Status:** {gate_status_text}"
)

st.markdown("#### 💡 Recommended Bets & Model Analysis for This Matchup")
st.caption(
    "Win/lose, strikeout, and hit probabilities are always computed by the model itself from real, "
    "live-fetched factors (team winning percentage and Pythagorean run expectation, starting pitcher "
    "ERA/WHIP/K9, team pitching-staff ERA, team fielding percentage, rest days, and platoon splits where "
    "available) -- never copied from a sportsbook number and never a random placeholder. Where the "
    "selected book has a real line posted, it's shown alongside for comparison, tagged with the book. "
    "Where no book has posted a line yet, the model's own estimate is still shown, clearly marked as a "
    "projection with no live price attached -- so it's never presented as something you could actually "
    "place."
)

event_props = {}
if include_props and selected_game_item.get('market_entry') and selected_game_item['market_entry'].get('event_id'):
    event_props = fetch_event_player_props(effective_odds_api_key, selected_game_item['market_entry']['event_id'], chosen_bm_key)

cand_bets = []

# Applies to Run Line, Game Total, Pitcher Strikeout Prop, and Batter Hit Prop
# only -- Moneyline keeps its own separate threshold/logic below, untouched.
MIN_PROP_MODEL_PROB = 0.70

def add_cand(bet_type, desc, leg):
    if leg.get('has_market') and leg.get('market_prob') is not None:
        edge_pp = (leg['model_prob'] - leg['market_prob']) * 100
        detail = (
            f"Sportsbook Odds: <strong>{leg['odds_str']}</strong> ({leg['decimal_odds']:.2f}x) "
            f"<span class='book-tag'>via {leg['source_book']}</span>"
            f"<br><span class='prob-badge'>Model Probability: {leg['model_prob']*100:.1f}%"
            f" &nbsp;|&nbsp; Market Probability: {leg['market_prob']*100:.1f}%"
            f" &nbsp;|&nbsp; Edge: {edge_pp:+.1f}pp</span>"
        )
    else:
        detail = (
            f"<span class='no-market-note'>⚪ No live sportsbook line currently posted for this exact bet.</span>"
            f"<br><span class='prob-badge'>Model Probability: {leg['model_prob']*100:.1f}%"
            f" (from the model's own factors)</span>"
        )
    cand_bets.append({'type': bet_type, 'html': f"<li><strong>{bet_type}:</strong> {desc} — {detail}</li>"})

away_exp = selected_game_item['away_exp_runs']
home_exp = selected_game_item['home_exp_runs']

# ---- Confirmed starting lineup analysis ----
# This ONLY runs for the single game selected in this Deep-Dive tab -- it is
# never applied across the full slate, so worst-case added API load is
# bounded to ~9 calls per team for the ONE game being viewed (often 0, since
# most everyday starters are already covered by the cached OPS-leaders
# lookup above). Most of the day, MLB hasn't posted lineups yet (they
# usually appear ~1-2 hours before first pitch) -- in that case this
# section is skipped entirely at zero extra cost, and everything falls
# back to the team-level real data already used elsewhere.
LEAGUE_AVG_OPS = 0.740  # approximate modern-era reference, not live-fetched

away_lineup_raw = selected_game.get('away_lineup', [])
home_lineup_raw = selected_game.get('home_lineup', [])
lineups_posted = bool(away_lineup_raw) and bool(home_lineup_raw)

away_lineup_hitters = []
home_lineup_hitters = []
if lineups_posted:
    ops_by_player = ops_leaders_map.get('by_player', {})
    away_lineup_hitters = get_lineup_hitters(away_lineup_raw, ops_by_player, season)
    home_lineup_hitters = get_lineup_hitters(home_lineup_raw, ops_by_player, season)

    away_found_ops = [h['ops'] for h in away_lineup_hitters if h.get('ops') is not None]
    home_found_ops = [h['ops'] for h in home_lineup_hitters if h.get('ops') is not None]

    # Modest, capped nudge to the season-long run projection based on how
    # TODAY's real confirmed lineup compares to league-average OPS -- e.g.
    # correctly reflects a weakened lineup if a regular is out, which the
    # team's full-season average alone wouldn't catch.
    if away_found_ops:
        away_lineup_avg_ops = sum(away_found_ops) / len(away_found_ops)
        ops_adjust = max(-0.15, min(0.15, (away_lineup_avg_ops - LEAGUE_AVG_OPS) / LEAGUE_AVG_OPS))
        away_exp = max(1.5, away_exp * (1.0 + ops_adjust * 0.5))
    if home_found_ops:
        home_lineup_avg_ops = sum(home_found_ops) / len(home_found_ops)
        ops_adjust = max(-0.15, min(0.15, (home_lineup_avg_ops - LEAGUE_AVG_OPS) / LEAGUE_AVG_OPS))
        home_exp = max(1.5, home_exp * (1.0 + ops_adjust * 0.5))

st.markdown("#### 🎯 Game Total (Over/Under) Analysis")
model_total_exp = away_exp + home_exp
_model_total_legs = build_model_total_projection(selected_game_item, away_exp, home_exp)
_over_m = next(l for l in _model_total_legs if ' Over ' in l['desc'])
_under_m = next(l for l in _model_total_legs if ' Under ' in l['desc'])
_model_point = round(model_total_exp * 2) / 2.0
_weather_factor = selected_game_item.get('weather_factor', 1.0)
_weather_note = selected_game_item.get('weather_note', '')

# Always shown -- this is the model's own general read on the game, entirely
# independent of whether any book has posted a line at all.
total_display_text = (
    f"⚾ **Model's Projected Total:** {model_total_exp:.2f} combined runs (model's own line: {_model_point})\n\n"
    f"**Over {_model_point}:** {_over_m['model_prob']*100:.1f}% &nbsp;|&nbsp; "
    f"**Under {_model_point}:** {_under_m['model_prob']*100:.1f}%\n\n"
    f"🌬️ *Weather adjustment: {_weather_note} (net factor: {_weather_factor:.3f}x on expected runs)*"
)

# Layered on top, only when a real line exists -- how the model's view
# compares against the actual live market at the market's own point, which
# can differ from the model's own natural line above.
real_total_legs_display = build_total_legs(selected_game_item, away_exp, home_exp)
if real_total_legs_display:
    line_point = selected_game_item['market_entry']['totals']['point']
    over_leg = next((l for l in real_total_legs_display if ' Over ' in l['desc']), None)
    under_leg = next((l for l in real_total_legs_display if ' Under ' in l['desc']), None)
    rows = []
    if over_leg:
        edge = (over_leg['model_prob'] - over_leg['market_prob']) * 100
        rows.append(
            f"**Over {line_point}** ({over_leg['odds_str']} via {over_leg['source_book']}): "
            f"Model {over_leg['model_prob']*100:.1f}% &nbsp;|&nbsp; Market {over_leg['market_prob']*100:.1f}% "
            f"&nbsp;|&nbsp; Edge {edge:+.1f}pp"
        )
    if under_leg:
        edge = (under_leg['model_prob'] - under_leg['market_prob']) * 100
        rows.append(
            f"**Under {line_point}** ({under_leg['odds_str']} via {under_leg['source_book']}): "
            f"Model {under_leg['model_prob']*100:.1f}% &nbsp;|&nbsp; Market {under_leg['market_prob']*100:.1f}% "
            f"&nbsp;|&nbsp; Edge {edge:+.1f}pp"
        )
    total_display_text += "\n\n---\n\n**📈 Live Market Comparison**\n\n" + "\n\n".join(rows)
else:
    total_display_text += "\n\n*No live total line posted for this game yet -- nothing to compare against.*"

st.info(total_display_text)

# Moneyline -- model_prob always comes from model_moneyline_game's own factor
# model; market fields are only populated when has_ml_market is True.
ml_model_prob = selected_game_item['model_prob']
if selected_game_item['has_ml_market']:
    ml_leg = {
        'model_prob': ml_model_prob, 'market_prob': selected_game_item['implied_prob'],
        'odds_str': selected_game_item['market_odds_str'],
        'decimal_odds': american_to_decimal(selected_game_item['market_odds']),
        'source_book': selected_game_item['source_book'], 'has_market': True,
    }
    if 0.58 <= ml_model_prob < 0.92:
        add_cand("Moneyline", f"<strong>{selected_game_item['pick']}</strong> to win", ml_leg)
else:
    # No live market price to compare against -- this is purely informational
    # (what the model thinks), not a claimed value edge, so it's shown
    # regardless of confidence level rather than gated behind a threshold
    # meant for real, comparable market prices.
    ml_leg = {
        'model_prob': ml_model_prob, 'market_prob': None, 'odds_str': None,
        'decimal_odds': None, 'source_book': None, 'has_market': False,
    }
    add_cand("Moneyline", f"<strong>{selected_game_item['pick']}</strong> to win", ml_leg)

rl_real_leg = build_run_line_leg(selected_game_item)
if rl_real_leg:
    if MIN_PROP_MODEL_PROB <= rl_real_leg['model_prob'] < 0.92:
        add_cand("Run Line", rl_real_leg['desc'], rl_real_leg)
else:
    rl_model_leg = build_model_run_line_projection(selected_game_item)
    if MIN_PROP_MODEL_PROB <= rl_model_leg['model_prob'] < 0.92:
        add_cand("Run Line", rl_model_leg['desc'], rl_model_leg)

real_total_legs = build_total_legs(selected_game_item, away_exp, home_exp)
if real_total_legs:
    for leg in real_total_legs:
        if MIN_PROP_MODEL_PROB <= leg['model_prob'] < 0.92:
            add_cand("Game Total", leg['desc'], leg)
else:
    # No live total posted (e.g. the odds API isn't returning bulk odds right
    # now) -- the model's own Over/Under estimate at its own projected line
    # is shown instead. Note this line is picked to match the model's own
    # expected total, so it's naturally close to 50/50 and will often be
    # filtered out entirely by the confidence threshold below -- that's
    # expected, not a bug.
    for leg in build_model_total_projection(selected_game_item, away_exp, home_exp):
        if MIN_PROP_MODEL_PROB <= leg['model_prob'] < 0.92:
            add_cand("Game Total", leg['desc'], leg)

for p_name, p_stats, t_name in [
    (selected_game['away_pitcher'], selected_game_item['away_stats'], selected_game['away_team']),
    (selected_game['home_pitcher'], selected_game_item['home_stats'], selected_game['home_team'])
]:
    if "TBD" not in p_name:
        real_k_legs = []
        if include_props and event_props:
            real_k_legs = build_k_prop_legs(p_name, t_name, p_stats['k_avg'], event_props, selected_game_item['game'])
        if real_k_legs:
            for leg in real_k_legs:
                if MIN_PROP_MODEL_PROB <= leg['model_prob'] < 0.92:
                    add_cand("Pitcher Strikeout Prop", leg['desc'], leg)
        else:
            for leg in build_model_k_prop_projection(p_name, t_name, p_stats['k_avg'], selected_game_item['game']):
                if MIN_PROP_MODEL_PROB <= leg['model_prob'] < 0.92:
                    add_cand("Pitcher Strikeout Prop", leg['desc'], leg)

# Hit-prop candidates: if today's real confirmed lineup is available, use
# ALL of it (real BA per player) instead of just the team's one OPS-leader
# proxy -- more, and more specifically targeted, real candidates. Falls
# back to the single key-hitter proxy when lineups aren't posted yet.
if lineups_posted and away_lineup_hitters:
    away_hit_candidates = [(h['name'], selected_game['away_team'], h['ba']) for h in away_lineup_hitters if h.get('ba') is not None]
else:
    away_hit_candidates = [(selected_game_item['away_hitter']['name'], selected_game['away_team'], selected_game_item['away_hitter']['ba'])]

if lineups_posted and home_lineup_hitters:
    home_hit_candidates = [(h['name'], selected_game['home_team'], h['ba']) for h in home_lineup_hitters if h.get('ba') is not None]
else:
    home_hit_candidates = [(selected_game_item['home_hitter']['name'], selected_game['home_team'], selected_game_item['home_hitter']['ba'])]

for name, t_name, ba in away_hit_candidates + home_hit_candidates:
    real_h_legs = []
    if include_props and event_props:
        real_h_legs = build_hit_prop_legs(name, t_name, ba, event_props, selected_game_item['game'])
    if real_h_legs:
        for leg in real_h_legs:
            if MIN_PROP_MODEL_PROB <= leg['model_prob'] < 0.92:
                add_cand("Batter Hit Prop", leg['desc'], leg)
    else:
        for leg in build_model_hit_prop_projection(name, t_name, ba, selected_game_item['game']):
            if MIN_PROP_MODEL_PROB <= leg['model_prob'] < 0.92:
                add_cand("Batter Hit Prop", leg['desc'], leg)

if include_props and not event_props and selected_game_item.get('market_entry'):
    st.caption("No pitcher-strikeout or batter-hit prop lines were returned for this matchup by the selected book right now -- the pitcher/hitter entries above are the model's own projections instead.")

if cand_bets:
    bets_html = "".join([c['html'] for c in cand_bets[:10]])
    st.markdown(f'<div class="card-container"><div class="card-title">🛡️ Recommended Bets & Model Analysis</div><ol class="card-list">{bets_html}</ol></div>', unsafe_allow_html=True)
else:
    st.info("No bets for this matchup currently clear your confidence thresholds. Try another matchup or loosen your filters.")

if lineups_posted:
    with st.expander(f"🧾 Confirmed Starting Lineups (real, live -- {len(away_lineup_hitters)} + {len(home_lineup_hitters)} players)"):
        lc1, lc2 = st.columns(2)
        with lc1:
            st.markdown(f"**{selected_game['away_team']}**")
            for h in away_lineup_hitters:
                ops_txt = f", OPS {h['ops']:.3f}" if h.get('ops') is not None else ""
                st.write(f"• {h['name']} — BA {h['ba']:.3f}{ops_txt}")
        with lc2:
            st.markdown(f"**{selected_game['home_team']}**")
            for h in home_lineup_hitters:
                ops_txt = f", OPS {h['ops']:.3f}" if h.get('ops') is not None else ""
                st.write(f"• {h['name']} — BA {h['ba']:.3f}{ops_txt}")
else:
    st.caption(
        "ℹ️ Starting lineups for this game aren't posted yet by MLB (usually available ~1-2 hours before "
        "first pitch). Using each team's real OPS leader as the offensive proxy instead -- shown in the "
        "profiles below."
    )

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"#### 🧢 {selected_game['away_team']} Profile")
    p_stats_away = selected_game_item['away_stats']
    hitter_away = selected_game_item['away_hitter']
    def_away = selected_game_item['away_def']
    q_away = selected_game_item['away_quality']

    st.markdown(f"**Starting Pitcher:** {selected_game['away_pitcher']} ({'LHP' if p_stats_away['throws']=='L' else 'RHP'})")
    if not p_stats_away.get('is_real', True):
        st.warning(
            f"⚠️ Could not fetch real season stats for {selected_game['away_pitcher']} right now -- "
            f"the numbers below are a generic league-average placeholder (ERA 4.20, ~4.5 K/game), "
            f"NOT this pitcher's actual real stats. Treat any prediction using these as unreliable "
            f"until this clears (usually a temporary MLB API hiccup)."
        )
    st.write(f"• ERA: **{p_stats_away['era']:.2f}** | Team Staff ERA (real, bullpen/depth proxy): **{q_away['staff_era']:.2f}**")
    st.write(f"• **Team Fielding (real):** FLD%: `{def_away['fielding_pct']:.3f}` | Errors: `{def_away['errors']}`")
    if selected_game_item.get('away_rest_days') is not None:
        st.caption(f"Rest: {selected_game_item['away_rest_days']} days since last start (real, from game log)")

    k_line_away = st.number_input(f"{selected_game['away_pitcher']} Strikeout Milestone (model-only, custom line)", min_value=1, max_value=15, value=max(1, min(15, int(np.floor(p_stats_away['k_avg'])))), step=1, key=f"k_away_{selected_game_item['game_pk']}")
    k_prob_away = calculate_k_prop_prob(k_line_away, p_stats_away['k_avg'])
    st.metric(f"Model Over Prob ({k_line_away}+ Strikeouts)", f"{k_prob_away*100:.1f}%")
    if event_props and 'pitcher_strikeouts' in event_props:
        real_line = event_props['pitcher_strikeouts']['lines'].get(normalize_player_name(selected_game['away_pitcher']))
        if real_line and real_line.get('Over', {}).get('point') is not None:
            pt = real_line['Over']['point']
            st.caption(f"📈 Real sportsbook line: Over/Under {pt} strikeouts, Over at {fmt_american(real_line['Over']['price'])} (Market Prob: {american_to_prob(real_line['Over']['price'])*100:.1f}%) via {event_props['pitcher_strikeouts']['source_book']}")

    st.markdown(f"**Key Hitter (real OPS leader):** {hitter_away['name']} (BA: **{hitter_away['ba']:.3f}**" + (f", OPS: **{hitter_away['ops']:.3f}**" if hitter_away.get('ops') else "") + ")")
    hit_prob_away = calculate_batter_hit_prob(hitter_away['ba'])
    st.metric(f"Model Hit Prob ({hitter_away['name']} 1+ Hits)", f"{hit_prob_away*100:.1f}%")
    if event_props and 'batter_hits' in event_props:
        real_line = event_props['batter_hits']['lines'].get(normalize_player_name(hitter_away['name']))
        if real_line and real_line.get('Over', {}).get('point') is not None:
            pt = real_line['Over']['point']
            st.caption(f"📈 Real sportsbook line: Over/Under {pt} hits, Over at {fmt_american(real_line['Over']['price'])} (Market Prob: {american_to_prob(real_line['Over']['price'])*100:.1f}%) via {event_props['batter_hits']['source_book']}")

with col2:
    st.markdown(f"#### 🧢 {selected_game['home_team']} Profile")
    p_stats_home = selected_game_item['home_stats']
    hitter_home = selected_game_item['home_hitter']
    def_home = selected_game_item['home_def']
    q_home = selected_game_item['home_quality']

    st.markdown(f"**Starting Pitcher:** {selected_game['home_pitcher']} ({'LHP' if p_stats_home['throws']=='L' else 'RHP'})")
    if not p_stats_home.get('is_real', True):
        st.warning(
            f"⚠️ Could not fetch real season stats for {selected_game['home_pitcher']} right now -- "
            f"the numbers below are a generic league-average placeholder (ERA 4.20, ~4.5 K/game), "
            f"NOT this pitcher's actual real stats. Treat any prediction using these as unreliable "
            f"until this clears (usually a temporary MLB API hiccup)."
        )
    st.write(f"• ERA: **{p_stats_home['era']:.2f}** | Team Staff ERA (real, bullpen/depth proxy): **{q_home['staff_era']:.2f}**")
    st.write(f"• **Team Fielding (real):** FLD%: `{def_home['fielding_pct']:.3f}` | Errors: `{def_home['errors']}`")
    if selected_game_item.get('home_rest_days') is not None:
        st.caption(f"Rest: {selected_game_item['home_rest_days']} days since last start (real, from game log)")

    k_line_home = st.number_input(f"{selected_game['home_pitcher']} Strikeout Milestone (model-only, custom line)", min_value=1, max_value=15, value=max(1, min(15, int(np.floor(p_stats_home['k_avg'])))), step=1, key=f"k_home_{selected_game_item['game_pk']}")
    k_prob_home = calculate_k_prop_prob(k_line_home, p_stats_home['k_avg'])
    st.metric(f"Model Over Prob ({k_line_home}+ Strikeouts)", f"{k_prob_home*100:.1f}%")
    if event_props and 'pitcher_strikeouts' in event_props:
        real_line = event_props['pitcher_strikeouts']['lines'].get(normalize_player_name(selected_game['home_pitcher']))
        if real_line and real_line.get('Over', {}).get('point') is not None:
            pt = real_line['Over']['point']
            st.caption(f"📈 Real sportsbook line: Over/Under {pt} strikeouts, Over at {fmt_american(real_line['Over']['price'])} (Market Prob: {american_to_prob(real_line['Over']['price'])*100:.1f}%) via {event_props['pitcher_strikeouts']['source_book']}")

    st.markdown(f"**Key Hitter (real OPS leader):** {hitter_home['name']} (BA: **{hitter_home['ba']:.3f}**" + (f", OPS: **{hitter_home['ops']:.3f}**" if hitter_home.get('ops') else "") + ")")
    hit_prob_home = calculate_batter_hit_prob(hitter_home['ba'])
    st.metric(f"Model Hit Prob ({hitter_home['name']} 1+ Hits)", f"{hit_prob_home*100:.1f}%")
    if event_props and 'batter_hits' in event_props:
        real_line = event_props['batter_hits']['lines'].get(normalize_player_name(hitter_home['name']))
        if real_line and real_line.get('Over', {}).get('point') is not None:
            pt = real_line['Over']['point']
            st.caption(f"📈 Real sportsbook line: Over/Under {pt} hits, Over at {fmt_american(real_line['Over']['price'])} (Market Prob: {american_to_prob(real_line['Over']['price'])*100:.1f}%) via {event_props['batter_hits']['source_book']}")

st.markdown("---")

# -----------------------------------------------------------------------------
# SECTION 7: PARLAY BUILDER (real lines only, real combined payout)
# -----------------------------------------------------------------------------
st.markdown("### 🚀 Algorithmic Multi-Tier Parlay Builder")
st.caption(
    "Model probabilities here are computed the same way as above -- from the model's own factors. But unlike "
    "the Recommended Bets list, every leg below also requires a REAL price from the selected book, since a "
    "parlay payout can only be calculated from actual posted odds, not a model-only projection. No leg is "
    "invented, and the payout shown is the actual product of the posted odds -- not a theoretical "
    "1/probability number."
)

_ready_games_count = sum(1 for _r in ml_results_sorted if _r['is_ready'])
_total_games_count = len(ml_results_sorted)
if _total_games_count > 0 and _ready_games_count < _total_games_count:
    st.info(
        f"ℹ️ **{_ready_games_count} of {_total_games_count} games today have announced pitchers** and are "
        f"eligible for legs right now -- the rest are skipped until MLB confirms their starters. If parlays "
        f"feel repetitive, this is very often why: there may just not be much else eligible yet. See the "
        f"Player Props Diagnostic below for the full breakdown."
    )

if 'parlay_seed' not in st.session_state:
    st.session_state.parlay_seed = 100

if st.button("🔄 Refresh Parlays"):
    st.session_state.parlay_seed += 1
    st.rerun()

rng = random.Random(st.session_state.parlay_seed)

# Diagnostic counters for the player-props pipeline. Since prop legs depend on
# a chain of things all lining up (toggle on -> game has an event_id -> the
# book has posted a line for that specific pitcher/hitter -> the model clears
# 73%), it's easy for legs to silently disappear at any link. These counters
# surface exactly which link is failing instead of just showing "no props."
props_diag = {
    'eligible_games': 0,       # games where we even attempt an event-props fetch
    'games_with_props_data': 0,  # of those, how many returned ANY market data
    'k_legs_built': 0,         # strikeout legs built (book had a line + name matched)
    'k_legs_passed_73': 0,     # of those, how many cleared the 73% model-prob bar
    'hit_legs_built': 0,
    'hit_legs_passed_73': 0,
    'hr_legs_built': 0,
    'hr_legs_passed_73': 0,
    'total_games_in_slate': len(ml_results_sorted),
    'ready_games': sum(1 for _r in ml_results_sorted if _r['is_ready']),
}

# The alternate-margin (Kalshi) and player-prop lookups below are PER-EVENT
# calls (one extra network round trip per game, for each), and until now
# they only ran sequentially inside the leg-building loop -- unlike the
# team/pitcher stats pre-warmed earlier in the script, these were never
# parallelized, which is exactly the kind of "the page just sits there"
# slowness a fully-loaded slate with Kalshi mode + props both on would hit.
# Pre-warming them here, concurrently, means the sequential loop right below
# almost always hits an already-populated cache instead of a live request.
_event_ids_needing_props = []
_event_ids_needing_alt_spreads = []
for _r in ml_results_sorted:
    if not _r['is_ready'] or not _r.get('market_entry') or not _r['market_entry'].get('event_id'):
        continue
    _eid = _r['market_entry']['event_id']
    if include_props_in_parlay:
        _event_ids_needing_props.append(_eid)
    if kalshi_mode and strict_kalshi:
        _event_ids_needing_alt_spreads.append(_eid)

if _event_ids_needing_props or _event_ids_needing_alt_spreads:
    with st.spinner(f"Loading real-time prop/margin lines for {len(set(_event_ids_needing_props) | set(_event_ids_needing_alt_spreads))} game(s)..."):
        _prop_target_count = len(_event_ids_needing_props) + len(_event_ids_needing_alt_spreads)
        _prop_worker_count = max(16, min(64, _prop_target_count * 3)) if _prop_target_count else 16
        with ThreadPoolExecutor(max_workers=_prop_worker_count) as _executor:
            _futures = []
            for _eid in _event_ids_needing_props:
                _futures.append(_executor.submit(_safe_call, fetch_event_player_props, effective_odds_api_key, _eid, chosen_bm_key, strict_kalshi))
            for _eid in _event_ids_needing_alt_spreads:
                _futures.append(_executor.submit(_safe_call, fetch_event_alternate_spreads, effective_odds_api_key, _eid, chosen_bm_key, True))
            for _f in as_completed(_futures):
                pass  # results discarded -- this call's only job is to populate the shared cache

master_legs = []
track_conn = get_track_db()
for r in ml_results_sorted:
    if not r['is_ready']:
        continue
    if r['abstract_state'] == 'Final':
        # Can't place a new bet on a game that's already over -- exclude it
        # from the parlay-leg pool entirely rather than relying on is_ready,
        # which is deliberately True for Final games (it only gates on
        # pitcher-announcement, since obviously the starters are known by
        # the time a game has ended).
        continue

    if r['has_ml_market'] and r['model_prob'] >= 0.58:
        master_legs.append({
            'desc': f"{r['pick']} Moneyline ({r['market_odds_str']})",
            'model_prob': r['model_prob'],
            'market_prob': r['implied_prob'],
            'odds_str': r['market_odds_str'],
            'decimal_odds': american_to_decimal(r['market_odds']),
            'source_book': r['source_book'],
            'game': r['game'],
            'type': 'ml'
        })
        _ = log_leg_for_tracking(
            track_conn, r['game_pk'], r['game_date'], 'ml', r['pick'],
            f"{r['pick']} Moneyline ({r['market_odds_str']})",
            r['model_prob'], r['implied_prob'], r['market_odds_str'], r['source_book']
        )

    if kalshi_mode and strict_kalshi:
        # Kalshi's actual product for margin-of-victory: a ladder of separate
        # "team wins by X+ runs" contracts, not one fixed -1.5/+1.5 run line.
        # Fetched per-event (costs 1 extra API credit per game, same as
        # props) and priced strictly from Kalshi -- no fallback to other
        # books, matching the same strict sourcing used for props below.
        if r.get('market_entry') and r['market_entry'].get('event_id'):
            alt_spreads = fetch_event_alternate_spreads(
                effective_odds_api_key, r['market_entry']['event_id'], chosen_bm_key, strict=True
            )
            for leg in build_alt_spread_legs(r, r['away_exp_runs'], r['home_exp_runs'], alt_spreads, r['game']):
                if leg['model_prob'] >= 0.73:
                    master_legs.append(leg)
                    _ = log_leg_for_tracking(
                        track_conn, r['game_pk'], r['game_date'], 'rl', leg['desc'],
                        leg['desc'], leg['model_prob'], leg['market_prob'], leg['odds_str'], leg['source_book']
                    )
    elif not kalshi_mode:
        rl_leg = build_run_line_leg(r)
        if rl_leg and rl_leg['model_prob'] >= 0.73:
            master_legs.append(rl_leg)
            _ = log_leg_for_tracking(
                track_conn, r['game_pk'], r['game_date'], 'rl', rl_leg['desc'],
                rl_leg['desc'], rl_leg['model_prob'], rl_leg['market_prob'], rl_leg['odds_str'], rl_leg['source_book']
            )

    away_exp_r = r['away_exp_runs']
    home_exp_r = r['home_exp_runs']
    for leg in build_total_legs(r, away_exp_r, home_exp_r, is_kalshi=strict_kalshi):
        if leg['model_prob'] >= 0.73:
            master_legs.append(leg)
            _ = log_leg_for_tracking(
                track_conn, r['game_pk'], r['game_date'], 'total', leg['desc'],
                leg['desc'], leg['model_prob'], leg['market_prob'], leg['odds_str'], leg['source_book']
            )

    if include_props_in_parlay and r.get('market_entry') and r['market_entry'].get('event_id'):
        props_diag['eligible_games'] += 1
        ev_props = fetch_event_player_props(
            effective_odds_api_key, r['market_entry']['event_id'], chosen_bm_key, strict=strict_kalshi
        )
        if ev_props:
            props_diag['games_with_props_data'] += 1
            for p_name, p_stats, t_name in [(r['away_pitcher'], r['away_stats'], r['away_team']), (r['home_pitcher'], r['home_stats'], r['home_team'])]:
                if "TBD" not in p_name:
                    for leg in build_k_prop_legs(p_name, t_name, p_stats['k_avg'], ev_props, r['game'], is_kalshi=strict_kalshi):
                        props_diag['k_legs_built'] += 1
                        if leg['model_prob'] >= 0.73:
                            props_diag['k_legs_passed_73'] += 1
                            master_legs.append(leg)
                            _ = log_leg_for_tracking(
                                track_conn, r['game_pk'], r['game_date'], 'prop', leg['desc'],
                                leg['desc'], leg['model_prob'], leg['market_prob'], leg['odds_str'], leg['source_book']
                            )
            for h_dict, t_name in [(r['away_hitter'], r['away_team']), (r['home_hitter'], r['home_team'])]:
                for leg in build_hit_prop_legs(h_dict['name'], t_name, h_dict['ba'], ev_props, r['game'], is_kalshi=strict_kalshi):
                    props_diag['hit_legs_built'] += 1
                    if leg['model_prob'] >= 0.73:
                        props_diag['hit_legs_passed_73'] += 1
                        master_legs.append(leg)
                        _ = log_leg_for_tracking(
                            track_conn, r['game_pk'], r['game_date'], 'prop', leg['desc'],
                            leg['desc'], leg['model_prob'], leg['market_prob'], leg['odds_str'], leg['source_book']
                        )
                for leg in build_hr_prop_legs(h_dict['name'], t_name, h_dict.get('hr_per_game'), ev_props, r['game'], is_kalshi=strict_kalshi):
                    props_diag['hr_legs_built'] += 1
                    if leg['model_prob'] >= 0.73:
                        props_diag['hr_legs_passed_73'] += 1
                        master_legs.append(leg)
                        _ = log_leg_for_tracking(
                            track_conn, r['game_pk'], r['game_date'], 'prop', leg['desc'],
                            leg['desc'], leg['model_prob'], leg['market_prob'], leg['odds_str'], leg['source_book']
                        )

    try:
        track_conn.commit()
    except sqlite3.OperationalError:
        pass

try:
    track_conn.close()
except sqlite3.OperationalError:
    pass
with st.spinner("Syncing track record (settling recent results)..."):
    _ = run_track_record_maintenance()

with st.expander("🔍 Player Props Diagnostic (why strikeout/hit/HR legs may be missing)"):
    _ready = props_diag['ready_games']
    _total = props_diag['total_games_in_slate']
    if _total > 0 and _ready < _total:
        st.write(
            f"⚠️ **Only {_ready} of {_total} games today have announced starting pitchers "
            f"('Strict Pitcher Announcement Gate' is ON).** The other {_total - _ready} game(s) are "
            f"skipped entirely right now -- no moneyline, no totals, no props -- because MLB hasn't "
            f"confirmed their pitchers yet. This is very often WHY parlays feel repetitive or limited "
            f"to just a couple of games: there may simply be nothing else eligible yet. This naturally "
            f"opens up as pitchers get announced closer to game time (usually within 12-24 hours of "
            f"first pitch)."
        )
    else:
        st.write(f"✅ All {_total} game(s) today have announced pitchers -- nothing is being skipped for that reason.")

    if not include_props_in_parlay:
        st.write(
            "❌ **'Also use real prop lines in the parlay builder' is currently OFF.** "
            "Turn it on in the sidebar under Player Props -- nothing downstream of this runs "
            "at all while it's off."
        )
    else:
        st.write(f"✅ Prop-line toggle is ON. Games checked for props: **{props_diag['eligible_games']}**")
        if props_diag['eligible_games'] == 0:
            st.write(
                f"⚠️ Zero games were even eligible to check -- this means no game in today's slate "
                f"has a `market_entry` with an `event_id`. Real reason from the bulk odds fetch: "
                f"**{market_odds_diagnostic['status']}** -- {market_odds_diagnostic['detail']}"
            )
        else:
            st.write(f"Games where the book(s) returned ANY strikeout/hit/HR market data: **{props_diag['games_with_props_data']}** / {props_diag['eligible_games']}")
            if props_diag['games_with_props_data'] == 0:
                st.write(
                    "⚠️ **None of the tracked books have posted pitcher-strikeout, batter-hit, or "
                    "batter-home-run lines for any game today.** This is a real, live-data situation "
                    "(props often aren't posted until closer to first pitch) rather than an app bug -- "
                    "try again closer to game time."
                )
            else:
                st.write(f"Strikeout legs built (book had a line + name matched): **{props_diag['k_legs_built']}** -- of those, cleared the 73% model-prob bar: **{props_diag['k_legs_passed_73']}**")
                st.write(f"Hit legs built (book had a line + name matched): **{props_diag['hit_legs_built']}** -- of those, cleared the 73% model-prob bar: **{props_diag['hit_legs_passed_73']}**")
                st.write(f"HR legs built (book had a line + name matched): **{props_diag['hr_legs_built']}** -- of those, cleared the 73% model-prob bar: **{props_diag['hr_legs_passed_73']}**")
                if props_diag['k_legs_built'] == 0 and props_diag['hit_legs_built'] == 0 and props_diag['hr_legs_built'] == 0:
                    st.write(
                        "⚠️ Books returned prop data, but none of it matched a pitcher/hitter name in "
                        "today's slate. If this persists, there may be a new name-formatting edge case "
                        "the normalizer doesn't handle yet -- happy to look at a specific pitcher/hitter name."
                    )
                elif props_diag['k_legs_passed_73'] == 0 and props_diag['hit_legs_passed_73'] == 0 and props_diag['hr_legs_passed_73'] == 0:
                    st.write(
                        "ℹ️ Legs exist, but none of today's pitchers/hitters clear the 73% model-confidence "
                        "bar -- this is expected on some days (73% is a strict bar, especially for HR props "
                        "which are naturally lower-probability) rather than a bug."
                    )

ml_pool = [leg for leg in master_legs if leg['type'] in ('ml', 'rl')]
prop_pool = [leg for leg in master_legs if leg['type'] in ('prop', 'total')]

def build_constrained_parlay(ml_pool, prop_pool, min_payout, max_payout, max_legs, rng, min_combined_prob=None):
    all_legs = ml_pool + prop_pool
    if not all_legs:
        return []

    candidates = []
    target_payout = (min_payout + max_payout) / 2.0

    def prob_floor_for_size(n):
        # min_combined_prob was being used as one FLAT floor for every leg
        # count -- e.g. 0.30 for a 2-leg combo AND for a 6-leg combo. That's
        # not just strict, it's arithmetically almost impossible past 2 legs:
        # multiplying together several legs that are each individually near
        # their own minimum bar (58% for moneylines, 73% for props) sinks
        # below a flat 0.30 by 3 legs and keeps falling from there, so every
        # High-Value parlay search was structurally forced down to 2 legs no
        # matter how many were requested. Anchoring the floor at 2 legs (so
        # behavior at size=2 is unchanged) and scaling it geometrically for
        # larger sizes keeps the same *average per-leg* quality bar instead
        # of an ever-harder-to-clear absolute product.
        if min_combined_prob is None:
            return None
        return min_combined_prob ** (n / 2.0)

    # Try the largest leg counts first. Searching small-to-large meant that as
    # soon as any 2-leg combo landed in the payout range it was returned
    # immediately, even though more legs (each a smaller, less extreme price)
    # were available and would have been a more diversified way to hit the
    # same payout target.
    # Group legs by their actual bet type (ml / rl / total / prop) rather
    # than the coarse ml_pool/prop_pool split. Game totals are almost always
    # available (one simple market fetch per game) while strikeout/hit/HR
    # props need an extra per-game API call and can come back empty --
    # lumping them into one "prop_pool" bucket meant totals crowded out
    # props by sheer volume, so a round of random draws would seat mostly
    # totals and moneylines even when good prop legs existed. Bucketing by
    # real type and round-robin-ing across whichever types actually have
    # legs available (below) guarantees a parlay pulls from every bet type
    # present, not just whichever type happens to have the most legs.
    type_buckets = {}
    for leg in all_legs:
        type_buckets.setdefault(leg['type'], []).append(leg)
    bucket_types = list(type_buckets.keys())

    for size in range(max_legs, 1, -1):
        # Geometric-mean decimal odds needed per leg for `size` legs to land
        # near target_payout. Drawing legs fully at random doesn't account
        # for this: multiplying `size` random odds together compounds fast,
        # so as size grows the random product almost always blows straight
        # past max_payout before landing in the window -- 2-leg combos were
        # the only size where a random draw reliably fit the narrow range,
        # which is why every high-value parlay was collapsing to 2 legs
        # regardless of max_legs. Biasing each pool toward legs whose own
        # odds sit near this per-leg target (while still shuffling within
        # that band for variety) makes the compounded payout actually land
        # near the target as size increases.
        per_leg_target = target_payout ** (1.0 / size)

        for _ in range(150):
            sorted_all = sorted(all_legs, key=lambda l: abs(l['decimal_odds'] - per_leg_target))
            shuffled_all = sorted_all[:max(size * 3, 10)]
            rng.shuffle(shuffled_all)

            sorted_buckets = {}
            for t in bucket_types:
                s = sorted(type_buckets[t], key=lambda l: abs(l['decimal_odds'] - per_leg_target))
                top = s[:max(size * 2, 6)]
                rng.shuffle(top)
                sorted_buckets[t] = top

            combo = []
            used_games = set()

            # Round-robin: one pass through the bet types (freshly shuffled
            # order each attempt) grabbing at most one leg per type per
            # pass, so a 4-leg combo naturally ends up as one of each type
            # when all four are available, instead of type composition
            # being left to chance.
            type_order = list(bucket_types)
            rng.shuffle(type_order)
            bucket_idx = {t: 0 for t in type_order}
            made_progress = True
            while len(combo) < size and made_progress:
                made_progress = False
                for t in type_order:
                    if len(combo) >= size:
                        break
                    bucket = sorted_buckets[t]
                    while bucket_idx[t] < len(bucket):
                        leg = bucket[bucket_idx[t]]
                        bucket_idx[t] += 1
                        if leg['game'] not in used_games:
                            combo.append(leg)
                            used_games.add(leg['game'])
                            made_progress = True
                            break


            if len(combo) < size:
                # Widen back out to the FULL pool (not just the odds-band
                # slice) so a short slate can still reach `size` legs by
                # reusing games if it truly has to -- same fallback as
                # before, just now only triggered when the biased slice
                # genuinely doesn't have enough distinct games.
                for leg in sorted_all:
                    if leg not in combo and len(combo) < size:
                        combo.append(leg)

            if len(combo) >= 2:
                payout = 1.0
                combined_prob = 1.0
                for l in combo:
                    payout *= l['decimal_odds']
                    combined_prob *= l['model_prob']
                meets_payout = min_payout <= payout <= max_payout
                size_floor = prob_floor_for_size(len(combo))
                meets_prob = (size_floor is None) or (combined_prob >= size_floor)
                if meets_payout and meets_prob:
                    return combo
                candidates.append((abs(payout - target_payout), combo, combined_prob))

    if candidates:
        # If there's a combined-probability floor, only consider candidates
        # that actually clear it FOR THEIR OWN leg count (falling back to the
        # full candidate list only if literally nothing does, so we still
        # return something).
        if min_combined_prob is not None:
            prob_ok = [c for c in candidates if c[2] >= prob_floor_for_size(len(c[1]))]
            pool_source = prob_ok if prob_ok else candidates
        else:
            pool_source = candidates
        # Prefer more legs when a combo is reasonably close to the target
        # payout (within 15%), rather than always grabbing the single
        # closest match, which tended to be a 2-leg combo.
        close_enough = [c for c in pool_source if c[0] <= target_payout * 0.15]
        pool = close_enough if close_enough else pool_source
        pool.sort(key=lambda x: (-len(x[1]), x[0]))
        return pool[0][1]
    return all_legs[:min(max_legs, len(all_legs))]

# Safe Anchor Parlay: Real Payout 2.0x to 3.5x (Max 3 legs)
safe_legs = build_constrained_parlay(ml_pool, prop_pool, min_payout=2.0, max_payout=3.5, max_legs=3, rng=rng)

# High-Value Parlay: Real Payout 5.0x to 9.0x (Max 6 legs), combined model hit
# probability floored at 30% so it doesn't chain together so many legs that
# the realistic odds of it actually hitting are negligible.
high_value_legs = build_constrained_parlay(
    ml_pool, prop_pool, min_payout=5.0, max_payout=9.0, max_legs=6, rng=rng, min_combined_prob=0.30
)

tier_tab1, tier_tab2 = st.tabs(["🛡️ Safe Anchor Parlay (2x - 3.5x Payout)", "🚀 High-Value Builder (5x - 9x Payout | Max 6 Bets)"])

def render_parlay_card(title, legs):
    if not legs:
        st.warning("⚠️ Not enough legs with real, currently-posted sportsbook lines meet the criteria for this parlay right now. Try adjusting filters, enabling player props, or refreshing.")
        return

    combined_payout = 1.0
    combined_model_prob = 1.0
    combined_market_prob = 1.0
    for leg in legs:
        combined_payout *= leg['decimal_odds']
        combined_model_prob *= leg['model_prob']
        combined_market_prob *= leg['market_prob']

    legs_html = "".join([
        f'<li><strong style="color: #FFFFFF;">{leg["desc"]}</strong> — Odds: <strong>{leg["odds_str"]}</strong> '
        f'<span class="book-tag">via {leg["source_book"]}</span>'
        f'<br><span class="prob-badge">Model: {leg["model_prob"]*100:.1f}% | Market: {leg["market_prob"]*100:.1f}% '
        f'| Edge: {(leg["model_prob"]-leg["market_prob"])*100:+.1f}pp</span></li>'
        for leg in legs
    ])
    st.markdown(
        f'<div class="card-container"><div class="card-title">{title} — {combined_payout:.2f}x Real Payout</div>'
        f'<div class="card-subtitle">Model Combined Hit Prob: {combined_model_prob*100:.1f}% '
        f'| Market Combined Implied Prob: {combined_market_prob*100:.1f}% | Total Legs: {len(legs)}</div>'
        f'<ol class="card-list">{legs_html}</ol></div>',
        unsafe_allow_html=True
    )

with tier_tab1:
    render_parlay_card("Safe Anchor Multi-Leg Parlay (2.0x - 3.5x Payout)", safe_legs)

with tier_tab2:
    render_parlay_card("High-Value Multi-Leg Parlay (5.0x - 9.0x Payout | Max 6 Bets)", high_value_legs)

# -----------------------------------------------------------------------------
# SECTION 8: MODEL TRACK RECORD & CALIBRATION
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 📊 Model Track Record & Calibration")
st.caption(
    "Every leg the model has surfaced above gets logged once to a local file, then graded against the REAL "
    "final score / real box-score strikeout and hit totals once that game goes Final -- nothing here is "
    "simulated or back-tested on made-up data. Rolls on a 120-day window: once logged history would span "
    "more than 120 days, the oldest days are deleted to make room for new ones, so this never grows without "
    "bound. New legs take a little while to show up here as 'resolved' -- settlement runs in the background "
    "roughly every 30 minutes, and only after a game is actually Final."
)

if st.button("🔄 Force Settle Pending Now", help="Bypasses the normal 30-minute wait and immediately tries to grade every still-pending prediction against real results. Useful for checking a backlog right away rather than waiting for the next automatic pass."):
    with st.spinner("Settling all pending predictions against real results..."):
        try:
            _fs_conn = get_track_db()
            try:
                _before = _fs_conn.execute("SELECT COUNT(*) FROM ml_predictions WHERE resolved = 0").fetchone()[0]
                settle_pending_legs(_fs_conn)
                settle_pending_predictions(_fs_conn)
                enforce_retention_window(_fs_conn)
                _after = _fs_conn.execute("SELECT COUNT(*) FROM ml_predictions WHERE resolved = 0").fetchone()[0]
            finally:
                _fs_conn.close()
            _resolved_now = _before - _after
            if _resolved_now > 0:
                st.success(f"✅ Settled {_resolved_now} prediction(s). {_after} still pending (likely still in progress or not yet Final).")
            else:
                st.info(f"ℹ️ No change -- {_after} prediction(s) still pending. If this stays stuck across multiple tries for clearly-completed games, something specific is wrong with those entries (worth flagging).")
        except sqlite3.OperationalError:
            st.warning("⚠️ Database was busy -- try again in a moment.")

def render_all_ml_predictions(conn):
    summary = conn.execute(
        "SELECT COUNT(*), SUM(resolved), MIN(game_date), MAX(game_date) FROM ml_predictions"
    ).fetchone()
    total, resolved, oldest, newest = summary
    resolved = resolved or 0
    if not total:
        st.info("ℹ️ No moneyline predictions logged yet -- come back after the app has run through at least one slate.")
        return

    # Direct answer to "did the model project a confident winner that still
    # lost today?" -- pulls resolved picks for the currently selected slate
    # date where the model said >=70% but the pick lost, with the real final
    # score. This is the at-a-glance version of that question; the bucketed
    # calibration table further down is the version that actually tells you
    # whether it's a real pattern or just today's ordinary variance.
    st.markdown(f"**🚨 Confident Misses -- {date_str}**")
    miss_rows = conn.execute(
        "SELECT away_team, home_team, pick, model_prob, away_score, home_score, market_odds_str "
        "FROM ml_predictions WHERE game_date = ? AND resolved = 1 AND result = 'loss' AND model_prob >= 0.70 "
        "ORDER BY model_prob DESC",
        (date_str,)
    ).fetchall()
    if miss_rows:
        for away_team, home_team, pick, model_prob, away_score, home_score, odds_str in miss_rows:
            score_txt = f"{away_team} {away_score} - {home_score} {home_team}" if away_score is not None else "score unavailable"
            st.markdown(f"- **{pick}** was projected at **{model_prob*100:.1f}%** ({odds_str}) and lost -- final: {score_txt}")
        st.caption(
            "One day's misses on their own don't mean much -- even a genuinely accurate 75% pick is *expected* "
            "to lose 1 time in 4. The calibration table below (built from every resolved pick across the full "
            "rolling window, not just today) is what actually tells you whether a probability range like this "
            "one is systematically off, versus just today's ordinary variance."
        )
    else:
        st.info(f"No resolved picks at 70%+ confidence lost on {date_str} (or nothing's resolved for this date yet).")

    days_span = None
    if oldest and newest:
        days_span = (datetime.strptime(newest, "%Y-%m-%d") - datetime.strptime(oldest, "%Y-%m-%d")).days + 1

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Picks Logged", f"{total:,}")
    m2.metric("Resolved (Final)", f"{resolved:,}")
    m3.metric("Pending", f"{total - resolved:,}")
    m4.metric("Days of History", f"{days_span or 0} / {TRACK_RETENTION_DAYS}")
    st.caption(f"Oldest: **{oldest}** | Newest: **{newest}** -- every pick the model made, win or lose, confident or not.")

    rows = conn.execute(
        "SELECT model_prob, result FROM ml_predictions WHERE resolved = 1 AND result IN ('win','loss')"
    ).fetchall()
    if not rows:
        st.info("ℹ️ Nothing has been graded yet -- check back once today's (or a recent day's) games go Final.")
        return

    df = pd.DataFrame(rows, columns=['model_prob', 'result'])
    bucket_edges = [0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 1.01]
    bucket_labels = ['50-55%', '55-60%', '60-65%', '65-70%', '70-80%', '80-90%', '90%+']
    df['bucket'] = pd.cut(df['model_prob'], bins=bucket_edges, labels=bucket_labels, right=False)

    calib_rows = []
    for bucket in bucket_labels:
        sub = df[df['bucket'] == bucket]
        if len(sub) == 0:
            continue
        wins = (sub['result'] == 'win').sum()
        n = len(sub)
        calib_rows.append({
            'Model Prob Bucket': bucket,
            'Avg Model Prob': f"{sub['model_prob'].mean()*100:.1f}%",
            'Actual Hit Rate': f"{wins/n*100:.1f}%",
            'Sample Size': n,
            'Confidence': "⚠️ Low sample" if n < 30 else "✅ OK"
        })
    st.markdown("**Calibration: Model Win Probability vs. Real Outcome**")
    if calib_rows:
        st.dataframe(pd.DataFrame(calib_rows), hide_index=True, use_container_width=True)
        st.caption(
            "This is the complete picture -- every pick, not just the confident ones that made it into a parlay. "
            "If, say, '55-60%' picks are actually winning less than 55-60% of the time, that's a real, useful signal "
            "the model runs overconfident in close games -- something the parlay-only view above would never show you."
        )
    else:
        st.info("Not enough resolved picks yet to build a calibration table.")

def render_pitcher_strikeout_log(conn):
    summary = conn.execute(
        "SELECT COUNT(*), SUM(resolved), MIN(game_date), MAX(game_date) FROM pitcher_starts"
    ).fetchone()
    total, resolved, oldest, newest = summary
    resolved = resolved or 0
    if not total:
        st.info("ℹ️ No pitcher starts logged yet -- come back after the app has run through at least one slate.")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Starts Logged", f"{total:,}")
    m2.metric("Resolved (Final)", f"{resolved:,}")
    m3.metric("Pending", f"{total - resolved:,}")
    st.caption("Every announced starting pitcher, every game -- logged whether or not a strikeout prop bet ever existed on them.")

    rows = conn.execute(
        "SELECT game_date, pitcher_name, team, opponent, season_k_avg, actual_k "
        "FROM pitcher_starts WHERE resolved = 1 ORDER BY game_date DESC LIMIT 300"
    ).fetchall()
    if not rows:
        st.info("ℹ️ Nothing has been graded yet -- check back once today's (or a recent day's) starts go Final.")
        return

    df = pd.DataFrame(rows, columns=['Date', 'Pitcher', 'Team', 'Opponent', 'Projected K (season avg)', 'Actual K'])
    df['Diff (Actual - Projected)'] = (df['Actual K'] - df['Projected K (season avg)']).round(2)

    mae = df['Diff (Actual - Projected)'].abs().mean()
    bias = df['Diff (Actual - Projected)'].mean()
    c1, c2 = st.columns(2)
    c1.metric("Avg Projection Error (MAE)", f"{mae:.2f} K")
    c2.metric(
        "Systematic Bias", f"{bias:+.2f} K",
        help="Positive means pitchers are actually striking out MORE than projected on average (model runs cold); "
             "negative means the model runs hot."
    )

    st.markdown("**Recent Starts (most recent 300 resolved)**")
    df['Projected K (season avg)'] = df['Projected K (season avg)'].round(2)
    st.dataframe(df, hide_index=True, use_container_width=True)

def render_parlay_legs_calibration(_read_conn):
    _summary = _read_conn.execute(
        "SELECT COUNT(*), SUM(resolved), MIN(game_date), MAX(game_date) FROM legs"
    ).fetchone()
    _total_legs, _resolved_legs, _oldest_date, _newest_date = _summary
    _resolved_legs = _resolved_legs or 0

    if not _total_legs:
        st.info("ℹ️ No legs logged yet -- come back after the app has run through at least one slate.")
        return

    days_span = None
    if _oldest_date and _newest_date:
        days_span = (datetime.strptime(_newest_date, "%Y-%m-%d") - datetime.strptime(_oldest_date, "%Y-%m-%d")).days + 1

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Legs Logged", f"{_total_legs:,}")
    m2.metric("Resolved (Final)", f"{_resolved_legs:,}")
    m3.metric("Pending Settlement", f"{_total_legs - _resolved_legs:,}")
    m4.metric("Days of History", f"{days_span or 0} / {TRACK_RETENTION_DAYS}")
    st.caption(f"Oldest logged date: **{_oldest_date}** | Newest: **{_newest_date}** -- rolling {TRACK_RETENTION_DAYS}-day window. This tab only covers legs that were confident/available enough to enter the Parlay Builder -- see the other tabs for the complete, unfiltered picture.")

    resolved_rows = _read_conn.execute(
        "SELECT leg_type, model_prob, result FROM legs WHERE resolved = 1 AND result IN ('win','loss')"
    ).fetchall()

    if not resolved_rows:
        st.info("ℹ️ Nothing has been graded yet -- check back once today's (or a recent day's) games go Final.")
        return

    df = pd.DataFrame(resolved_rows, columns=['leg_type', 'model_prob', 'result'])
    bucket_edges = [0.50, 0.58, 0.65, 0.70, 0.73, 0.80, 0.90, 1.01]
    bucket_labels = ['50-58%', '58-65%', '65-70%', '70-73%', '73-80%', '80-90%', '90%+']
    df['bucket'] = pd.cut(df['model_prob'], bins=bucket_edges, labels=bucket_labels, right=False)

    calib_rows = []
    for bucket in bucket_labels:
        sub = df[df['bucket'] == bucket]
        if len(sub) == 0:
            continue
        wins = (sub['result'] == 'win').sum()
        n = len(sub)
        hit_rate = wins / n
        avg_model_prob = sub['model_prob'].mean()
        calib_rows.append({
            'Model Prob Bucket': bucket,
            'Avg Model Prob': f"{avg_model_prob*100:.1f}%",
            'Actual Hit Rate': f"{hit_rate*100:.1f}%",
            'Sample Size': n,
            'Confidence': "⚠️ Low sample" if n < 30 else "✅ OK"
        })

    st.markdown("**Calibration: Model Probability vs. Real Hit Rate**")
    if calib_rows:
        st.dataframe(pd.DataFrame(calib_rows), hide_index=True, use_container_width=True)
        st.caption(
            "Read this as: for legs where the model said, e.g., '73-80% confident', did they actually hit "
            "73-80% of the time? Buckets marked ⚠️ have too few graded legs yet for the hit rate to mean much -- "
            "treat those as noise until more games settle into them."
        )
    else:
        st.info("Not enough resolved legs yet to build a calibration table.")

    st.markdown("**By Bet Type**")
    type_rows = []
    for leg_type, label in [('ml', 'Moneyline'), ('rl', 'Run Line'), ('total', 'Total'), ('prop', 'Strikeout/Hit Prop')]:
        sub = df[df['leg_type'] == leg_type]
        if len(sub) == 0:
            continue
        wins = (sub['result'] == 'win').sum()
        n = len(sub)
        type_rows.append({
            'Bet Type': label,
            'Avg Model Prob': f"{sub['model_prob'].mean()*100:.1f}%",
            'Actual Hit Rate': f"{wins/n*100:.1f}%",
            'Sample Size': n
        })
    if type_rows:
        st.dataframe(pd.DataFrame(type_rows), hide_index=True, use_container_width=True)

def render_track_record_section():
    if st.session_state.get('track_record_repair_message'):
        st.success(st.session_state.pop('track_record_repair_message'))
    try:
        _read_conn = get_track_db()
    except sqlite3.OperationalError:
        st.warning("⚠️ Track record database is temporarily busy (another process is writing to it) -- try refreshing in a moment.")
        return
    try:
        tab_ml, tab_pitchers, tab_legs = st.tabs([
            "⚾ All Moneyline Predictions", "🎯 Pitcher Strikeout Log", "🎫 Parlay Legs"
        ])
        with tab_ml:
            render_all_ml_predictions(_read_conn)
        with tab_pitchers:
            render_pitcher_strikeout_log(_read_conn)
        with tab_legs:
            render_parlay_legs_calibration(_read_conn)
    except sqlite3.OperationalError:
        st.warning("⚠️ Track record database is temporarily busy (another process is writing to it) -- try refreshing in a moment.")
    finally:
        try:
            _read_conn.close()
        except Exception:
            pass

_ = render_track_record_section()