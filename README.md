# Free MLB Predictor

A Streamlit MLB moneyline model built from data sources that are obtainable
without paid statistical feeds.

## Data sources

- MLB Stats API: schedule, teams, starters, lineups, recent performance, rest and travel
- Retrosheet: 2021–2025 historical training, pitchers, bullpen, defense, parks and umpires
- Open-Meteo: hourly weather forecasts without an API key
- The Odds API: current prices and no-vig market comparison (free key required)

The model uses 120 leakage-safe features. Current odds are not used as model
inputs because the free plan does not provide comparable historical odds; they
are used after inference to calculate market edge.

## Model audit

The fitted artifact contains XGBoost, LightGBM, logistic regression, Elo and
Poisson voters with Platt calibration. The chronological untouched test set is
1,811 games. Brier score is 0.2456 and log loss is 0.6843. These represent a
modest predictive signal, not guaranteed betting profit.

## Streamlit secret

Only one third-party secret is required:

```toml
ODDS_API_KEY = "your-key"
```

Never commit the real key to GitHub.
