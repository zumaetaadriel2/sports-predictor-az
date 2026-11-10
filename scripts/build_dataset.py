"""Limpia los CSV de data/raw/ y guarda una sola tabla en data/processed/matches.csv.

Uso (desde la raíz del repo):
    python scripts/build_dataset.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sports_predictor.football.data import load_matches  # noqa: E402


def main() -> None:
    df = load_matches(ROOT / "data" / "raw")
    out = ROOT / "data" / "processed" / "matches.csv"
    df.to_csv(out, index=False)
    print(f"{len(df)} partidos, {df['Season'].nunique()} temporadas, "
          f"{df['Date'].min().date()} a {df['Date'].max().date()}")
    print(df.groupby("Season").size().to_string())
    print(f"Guardado en {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
