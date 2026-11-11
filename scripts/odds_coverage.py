"""Revisa qué cuotas 1X2 trae cada temporada en los CSV crudos de data/raw/.

Una casa cuenta como "cuota 1X2" si el archivo tiene sus tres columnas <prefijo>H, <prefijo>D y
<prefijo>A. Para cada temporada se cuenta en cuántos partidos están las tres presentes.
Guarda la tabla en reports/odds_coverage.csv (filas = casa, columnas = temporada).

Uso (desde la raíz del repo):
    python scripts/odds_coverage.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sports_predictor.football.data import season_label  # noqa: E402


def one_x_two_prefixes(columns) -> list[str]:
    cols = set(columns)
    # HT (HTH/HTD/HTA no existe; HTR sí) y similares no chocan: se exige el trío completo.
    return sorted(c[:-1] for c in cols if c.endswith("H") and c[:-1] + "D" in cols and c[:-1] + "A" in cols)


def main() -> None:
    coverage = {}
    for path in sorted((ROOT / "data" / "raw").glob("E0_*.csv")):
        raw = pd.read_csv(path, encoding="utf-8-sig").dropna(subset=["HomeTeam"])
        coverage[season_label(path)] = {
            p: int(raw[[p + "H", p + "D", p + "A"]].notna().all(axis=1).sum())
            for p in one_x_two_prefixes(raw.columns)
        }
    table = pd.DataFrame(coverage).fillna(0).astype(int)
    # Primero las casas presentes en más temporadas.
    order = (table > 0).sum(axis=1).sort_values(ascending=False, kind="stable").index
    table = table.loc[order]
    table.index.name = "prefijo"

    out = ROOT / "reports" / "odds_coverage.csv"
    table.to_csv(out)
    print(table.to_string())
    print(f"\nGuardado en {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
