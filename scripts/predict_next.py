"""Predice los próximos partidos y los añade al historial de predicciones publicadas.

Usa los partidos jugados de data/raw/ (incluida la temporada en curso) y el archivo de próximos
partidos data/raw/fixtures.csv. Solo predice partidos con fecha igual o posterior a hoy que no
tengan resultado todavía. Lo que ya está en el historial no se vuelve a calcular ni se modifica.

Uso (desde la raíz del repo):
    python scripts/download_data.py --include-current --fixtures
    python scripts/predict_next.py
    python scripts/predict_next.py --as-of 2026-11-05 --history otra/ruta.csv   # para pruebas
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sports_predictor.football.data import load_matches  # noqa: E402
from sports_predictor.football.predict import (  # noqa: E402
    MATCH_KEY,
    append_predictions,
    fit_goal_model,
    load_fixtures,
    predict_fixtures,
    read_history,
)

HISTORY = ROOT / "site_data" / "predictions.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--as-of", default=None, help="Fecha de publicación AAAA-MM-DD (por defecto, hoy en UTC)")
    parser.add_argument("--history", default=str(HISTORY), help="Archivo de historial")
    args = parser.parse_args()
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()

    matches = load_matches(ROOT / "data" / "raw", include_holdout=True)
    matches = matches[matches["Date"] < as_of]  # con --as-of se simula una fecha pasada
    fixtures = load_fixtures(ROOT / "data" / "raw" / "fixtures.csv")
    played = set(map(tuple, matches[MATCH_KEY].itertuples(index=False)))
    pending = np.array([key not in played for key in map(tuple, fixtures[MATCH_KEY].itertuples(index=False))], dtype=bool)
    fixtures = fixtures[pending & (fixtures["Date"] >= as_of)]

    history = read_history(args.history)
    if len(fixtures):
        # El modelo solo ve partidos anteriores a la fecha de publicación.
        model = fit_goal_model(matches, as_of)
        updated = append_predictions(history, predict_fixtures(model, fixtures, as_of))
    else:
        updated = history
    Path(args.history).parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(args.history, index=False, date_format="%Y-%m-%d", lineterminator="\n")
    print(f"Fecha de publicación: {as_of.date()}. Partidos pendientes en fixtures: {len(fixtures)}. "
          f"Predicciones nuevas: {len(updated) - len(history)}. Total en el historial: {len(updated)}.")


if __name__ == "__main__":
    main()
