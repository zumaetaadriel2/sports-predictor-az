"""Carga y limpieza de los CSV de football-data.co.uk.

Cada temporada viene en un archivo `<liga>_<temporada>.csv` (por ejemplo `E0_1516.csv`).
Los archivos cambian de formato con los años: la fecha pasa de dd/mm/yy a dd/mm/yyyy,
algunos traen BOM, y las columnas de cuotas cambian de nombre (ver docs/ODDS_COLUMNS.md).
Aquí se deja todo en una sola tabla con columnas estables.
"""
from pathlib import Path

import pandas as pd

RESULT_COLS = ["FTHG", "FTAG", "FTR"]
STATS_COLS = ["HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"]

# Cuotas 1X2 que se conservan, con nombre estable -> nombres posibles en el CSV original.
# Hasta 2018/19 el promedio y el máximo del mercado se llamaban BbAv*/BbMx* (Betbrain);
# desde 2019/20 se llaman Avg*/Max*. Son el mismo concepto, así que se unifican.
ODDS_SOURCES = {
    "B365": ["B365"],  # Bet365, antes del partido
    "PS": ["PS"],  # Pinnacle, antes del partido
    "PSC": ["PSC"],  # Pinnacle, cierre
    "Avg": ["Avg", "BbAv"],  # promedio del mercado, antes del partido
    "Max": ["Max", "BbMx"],  # máximo del mercado, antes del partido
    "AvgC": ["AvgC"],  # promedio del mercado, cierre (desde 2019/20)
}
# Cuotas de más/menos de 2.5 goles: promedio del mercado antes del partido.
# Mismo cambio de nombre que en 1X2: BbAv>2.5 hasta 2018/19, Avg>2.5 desde 2019/20.
OVER_UNDER_SOURCES = {
    "AvgOver25": ["Avg>2.5", "BbAv>2.5"],
    "AvgUnder25": ["Avg<2.5", "BbAv<2.5"],
}
ODDS_COLS = [f"{name}{o}" for name in ODDS_SOURCES for o in "HDA"] + list(OVER_UNDER_SOURCES)

CLEAN_COLS = ["Season", "Date", "HomeTeam", "AwayTeam"] + RESULT_COLS + STATS_COLS + ODDS_COLS


def season_label(path: Path) -> str:
    """'E0_1516.csv' -> '2015/16'."""
    code = Path(path).stem.split("_")[-1]
    return f"20{code[:2]}/{code[2:]}"


def load_season(path: Path) -> pd.DataFrame:
    """Lee un CSV de una temporada y lo devuelve limpio, con las columnas de CLEAN_COLS."""
    path = Path(path)
    raw = pd.read_csv(path, encoding="utf-8-sig")  # utf-8-sig descarta el BOM si existe

    # Algunos archivos traen filas vacías al final.
    raw = raw.dropna(subset=["Date", "HomeTeam", "AwayTeam"], how="all")

    key = ["Date", "HomeTeam", "AwayTeam"] + RESULT_COLS
    missing = raw[key].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"{path.name}: {int(missing.sum())} partidos sin fecha, equipos o resultado")

    df = pd.DataFrame({"Season": season_label(path)}, index=raw.index)

    # La fecha viene como dd/mm/yy o dd/mm/yyyy según la temporada.
    dates = raw["Date"].str.strip()
    fmt = "%d/%m/%y" if dates.str.len().max() == 8 else "%d/%m/%Y"
    df["Date"] = pd.to_datetime(dates, format=fmt)

    df["HomeTeam"] = raw["HomeTeam"].str.strip()
    df["AwayTeam"] = raw["AwayTeam"].str.strip()
    df["FTHG"] = raw["FTHG"].astype(int)
    df["FTAG"] = raw["FTAG"].astype(int)
    df["FTR"] = raw["FTR"].str.strip()

    for col in STATS_COLS:
        df[col] = raw[col] if col in raw.columns else float("nan")

    for name, candidates in ODDS_SOURCES.items():
        source = next((c for c in candidates if f"{c}H" in raw.columns), None)
        for o in "HDA":
            df[f"{name}{o}"] = raw[f"{source}{o}"] if source else float("nan")

    for name, candidates in OVER_UNDER_SOURCES.items():
        source = next((c for c in candidates if c in raw.columns), None)
        df[name] = raw[source] if source else float("nan")

    _validate(df, path.name)
    # Orden estable: los partidos del mismo día conservan el orden del archivo.
    return df.sort_values("Date", kind="stable").reset_index(drop=True)[CLEAN_COLS]


def _validate(df: pd.DataFrame, name: str) -> None:
    expected = pd.Series("D", index=df.index)
    expected[df["FTHG"] > df["FTAG"]] = "H"
    expected[df["FTHG"] < df["FTAG"]] = "A"
    if not (df["FTR"] == expected).all():
        raise ValueError(f"{name}: FTR no coincide con los goles")
    if df.duplicated(["HomeTeam", "AwayTeam"]).any():
        raise ValueError(f"{name}: hay partidos duplicados (mismo local y visitante)")
    if (df["HomeTeam"] == df["AwayTeam"]).any():
        raise ValueError(f"{name}: un equipo juega contra sí mismo")
    if (df[ODDS_COLS] <= 1).any().any():
        raise ValueError(f"{name}: hay cuotas menores o iguales a 1")


# Última temporada de desarrollo. Las posteriores están reservadas para la prueba final.
LAST_DEV_SEASON = "2025/26"


def load_matches(raw_dir: Path, league: str = "E0", include_holdout: bool = False) -> pd.DataFrame:
    """Carga todas las temporadas de una liga, en orden cronológico.

    Por defecto deja fuera las temporadas posteriores a LAST_DEV_SEASON aunque su CSV exista: están
    reservadas para la prueba final y ninguna evaluación de desarrollo debe verlas.
    """
    files = sorted(Path(raw_dir).glob(f"{league}_*.csv"))
    if not files:
        raise FileNotFoundError(f"No hay CSV {league}_*.csv en {raw_dir}. Corre scripts/download_data.py")
    if not include_holdout:
        files = [f for f in files if season_label(f) <= LAST_DEV_SEASON]
    df = pd.concat([load_season(f) for f in files], ignore_index=True)
    return df.sort_values("Date", kind="stable").reset_index(drop=True)
