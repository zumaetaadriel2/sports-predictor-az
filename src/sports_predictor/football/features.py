"""Features previas al partido para el modelo de gradient boosting.

Regla: la fila de un partido solo usa partidos anteriores de cada equipo. Se consigue con
`shift(1)` dentro de cada equipo antes de promediar, así el partido actual nunca entra en su
propia media. La forma se arrastra de una temporada a la siguiente.
"""
import numpy as np
import pandas as pd

FORM_WINDOW = 5  # partidos recientes que se promedian
MAX_REST_DAYS = 14  # el descanso se recorta: más de dos semanas ya no aporta (parones, verano)

FEATURE_COLS = [
    f"{side}_{name}"
    for side in ("home", "away")
    for name in ("form_points", "form_goals_for", "form_goals_against", "form_shots_for",
                 "form_shots_against", "venue_points", "rest_days")
]


def form_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Forma reciente, forma en su condición (local/visitante) y días de descanso de ambos equipos.

    `matches` debe venir en orden cronológico. Devuelve una fila por partido, con el mismo índice.
    Un equipo sin partidos anteriores queda en NaN (LightGBM acepta valores faltantes).
    Solo se conocen los partidos de liga: el descanso no ve copas ni competiciones europeas.
    """
    points_home = matches["FTR"].map({"H": 3, "D": 1, "A": 0})
    points_away = matches["FTR"].map({"H": 0, "D": 1, "A": 3})
    sides = {
        "home": pd.DataFrame({
            "match": matches.index, "team": matches["HomeTeam"], "date": matches["Date"], "points": points_home,
            "goals_for": matches["FTHG"], "goals_against": matches["FTAG"],
            "shots_for": matches["HST"], "shots_against": matches["AST"],
        }),
        "away": pd.DataFrame({
            "match": matches.index, "team": matches["AwayTeam"], "date": matches["Date"], "points": points_away,
            "goals_for": matches["FTAG"], "goals_against": matches["FTHG"],
            "shots_for": matches["AST"], "shots_against": matches["HST"],
        }),
    }
    for side, table in sides.items():
        table["side"] = side
    # Una fila por equipo y partido. Orden estable por fecha: ningún equipo juega dos veces el mismo día.
    long = pd.concat(sides.values(), ignore_index=True).sort_values("date", kind="stable")
    if long.duplicated(["team", "date"]).any():
        raise ValueError("Un equipo juega dos veces el mismo día: la forma podría usar un partido de esa misma fecha")

    def previous_mean(group_keys, column):
        return long.groupby(group_keys)[column].transform(
            lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=1).mean()
        )

    for name in ("points", "goals_for", "goals_against", "shots_for", "shots_against"):
        long[f"form_{name}"] = previous_mean("team", name)
    # Forma solo en la misma condición: últimos partidos de local del local, de visitante del visitante.
    long["venue_points"] = previous_mean(["team", "side"], "points")
    rest = long.groupby("team")["date"].diff().dt.days
    long["rest_days"] = np.minimum(rest, MAX_REST_DAYS)

    names = [c.split("_", 1)[1] for c in FEATURE_COLS[: len(FEATURE_COLS) // 2]]
    out = pd.DataFrame(index=matches.index)
    for side in ("home", "away"):
        rows = long[long["side"] == side].set_index("match")
        for name in names:
            out[f"{side}_{name}"] = rows[name].reindex(matches.index)
    return out[FEATURE_COLS]
