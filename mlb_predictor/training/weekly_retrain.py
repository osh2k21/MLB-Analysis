from __future__ import annotations

import argparse
from pathlib import Path

from ..config import load_settings
from ..database import PredictionRepository
from .export import export_settled
from .train import train_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Export settled predictions and retrain the ensemble")
    parser.add_argument("--training-output", type=Path, default=Path("data/settled_training.csv"))
    args = parser.parse_args()
    settings = load_settings()
    rows = export_settled(PredictionRepository(settings.database_path), args.training_output)
    if rows < 500:
        raise RuntimeError(f"only {rows} settled rows; 500 required before retraining")
    bundle, metrics = train_bundle(args.training_output, settings.model_path)
    print(f"Retrained through {bundle.trained_through}: Brier={metrics['brier_score']:.4f}, log loss={metrics['log_loss']:.4f}")


if __name__ == "__main__":
    main()

