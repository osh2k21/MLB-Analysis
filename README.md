# MLB Predictor

This repository is a fail-closed rebuild of the original single-file Streamlit
application. It never replaces missing inputs with invented league-average
values. A game is marked **Prediction Withheld** until every required evidence
check passes and a fitted, calibrated ensemble is installed.

## Quick start

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python -m mlb_predictor.training.train --input data/training_games.csv
streamlit run app.py
```

The free MLB schedule feed works without a key. Odds and weather require keys.
Advanced Statcast, FanGraphs, injury, umpire, and park-factor inputs are
provider adapters: configure a structured JSON/CSV feed you are licensed to
use. There is deliberately no fake fallback and no brittle page scraping.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the data contract,
validation gates, feature catalog, training process, and known limitations.

