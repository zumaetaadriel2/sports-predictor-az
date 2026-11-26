"""Predicción de partidos que todavía no se jugaron, con el modelo Dixon-Coles + antigüedad.

Es el mismo modelo que evalúa scripts/evaluate_models.py, ajustado con todos los partidos
anteriores a una fecha. Las predicciones publicadas se guardan en un historial que nunca se reescribe.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from sports_predictor.football.poisson import (
    expected_goals,
    fit_poisson,
    fit_rho,
    outcome_probabilities,
    over_probability,
    score_matrices,
)
from sports_predictor.odds import implied_probabilities

HALF_LIFE_DAYS = 365.0  # la vida media elegida en las ocho ventanas de reports/final_report.md
PENALTY = 10.0
MATCH_KEY = ["Date", "HomeTeam", "AwayTeam"]
PREDICTION_COLS = MATCH_KEY + [
    "published_at", "exp_goals_home", "exp_goals_away", "p_home", "p_draw", "p_away", "p_over25",
    "market_home", "market_draw", "market_away", "market_over25",
]


def load_fixtures(path: Path, league: str = "E0") -> pd.DataFrame:
    """Lee el archivo de próximos partidos de football-data.co.uk y deja los de una liga.

    Conserva las cuotas promedio del mercado, si vienen, para comparar con el modelo.
    """
    raw = pd.read_csv(path, encoding="utf-8-sig")
    raw = raw[raw["Div"] == league]
    out = pd.DataFrame({
        "Date": pd.to_datetime(raw["Date"].str.strip(), format="%d/%m/%Y"),
        "HomeTeam": raw["HomeTeam"].str.strip(),
        "AwayTeam": raw["AwayTeam"].str.strip(),
    })
    for name, source in [("AvgH", "AvgH"), ("AvgD", "AvgD"), ("AvgA", "AvgA"),
                         ("AvgOver25", "Avg>2.5"), ("AvgUnder25", "Avg<2.5")]:
        out[name] = raw[source] if source in raw.columns else float("nan")
    return out.sort_values("Date", kind="stable").reset_index(drop=True)


def fit_goal_model(matches: pd.DataFrame, as_of, half_life_days: float = HALF_LIFE_DAYS, penalty: float = PENALTY):
    """Ajusta el modelo con los partidos de fecha estrictamente anterior a `as_of`.

    Devuelve un diccionario con los equipos conocidos, los parámetros del Poisson y `rho`.
    """
    as_of = pd.Timestamp(as_of)
    past = matches[matches["Date"] < as_of]
    teams = sorted(set(past["HomeTeam"]) | set(past["AwayTeam"]))
    index = {team: i for i, team in enumerate(teams)}
    home, away = past["HomeTeam"].map(index).to_numpy(), past["AwayTeam"].map(index).to_numpy()
    home_goals, away_goals = past["FTHG"].to_numpy(), past["FTAG"].to_numpy()
    age_days = (as_of - past["Date"]).dt.days.to_numpy()
    weights = 0.5 ** (age_days / half_life_days)

    params = fit_poisson(home, away, home_goals, away_goals, len(teams), penalty, weights=weights)
    rate_home, rate_away = expected_goals(params, home, away, len(teams))
    rho = fit_rho(rate_home, rate_away, home_goals, away_goals, weights)
    return {"teams": teams, "params": params, "rho": rho}


def _rates(model, home_teams, away_teams):
    """Goles esperados; un equipo sin partidos anteriores se trata como equipo promedio (ataque y defensa 0)."""
    n = len(model["teams"])
    index = {team: i for i, team in enumerate(model["teams"])}
    params = model["params"]
    # Se añade una casilla extra en 0 para los equipos desconocidos.
    extended = np.concatenate([params[:2], params[2:2 + n], [0.0], params[2 + n:], [0.0]])
    home = np.array([index.get(team, n) for team in home_teams])
    away = np.array([index.get(team, n) for team in away_teams])
    return expected_goals(extended, home, away, n + 1)


def predict_fixtures(model, fixtures: pd.DataFrame, published_at) -> pd.DataFrame:
    """Probabilidades del modelo y del mercado para cada partido de `fixtures`."""
    rate_home, rate_away = _rates(model, fixtures["HomeTeam"], fixtures["AwayTeam"])
    matrices = score_matrices(rate_home, rate_away, np.full(len(fixtures), model["rho"]))
    probs = outcome_probabilities(matrices)
    market = implied_probabilities(fixtures[["AvgH", "AvgD", "AvgA"]])
    market_ou = implied_probabilities(fixtures[["AvgUnder25", "AvgOver25"]])
    out = fixtures[MATCH_KEY].copy()
    out["published_at"] = pd.Timestamp(published_at).normalize()
    out["exp_goals_home"], out["exp_goals_away"] = rate_home, rate_away
    out["p_home"], out["p_draw"], out["p_away"] = probs.T
    out["p_over25"] = over_probability(matrices)
    out["market_home"], out["market_draw"], out["market_away"] = market.T
    out["market_over25"] = market_ou[:, 1]
    numeric = [c for c in PREDICTION_COLS if c not in MATCH_KEY + ["published_at"]]
    out[numeric] = out[numeric].round(4)
    return out[PREDICTION_COLS]


def team_strengths(model, teams) -> pd.DataFrame:
    """Goles esperados a favor y en contra de cada equipo contra un rival promedio, en campo neutral."""
    n = len(model["teams"])
    index = {team: i for i, team in enumerate(model["teams"])}
    base = model["params"][0]
    attack, defense = model["params"][2:2 + n], model["params"][2 + n:]
    rows = [
        {"team": t, "goals_for": float(np.exp(base + attack[index[t]])) if t in index else float(np.exp(base)),
         "goals_against": float(np.exp(base - defense[index[t]])) if t in index else float(np.exp(base))}
        for t in teams
    ]
    out = pd.DataFrame(rows)
    out["difference"] = out["goals_for"] - out["goals_against"]
    return out.sort_values("difference", ascending=False).reset_index(drop=True)


def read_history(path: Path) -> pd.DataFrame:
    """Lee el historial de predicciones publicadas; si no existe, devuelve uno vacío."""
    if not Path(path).exists():
        return pd.DataFrame(columns=PREDICTION_COLS)
    return pd.read_csv(path, parse_dates=["Date", "published_at"])[PREDICTION_COLS]


def append_predictions(history: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Añade al historial solo los partidos que aún no estaban. Lo ya publicado nunca se modifica."""
    known = set(map(tuple, history[MATCH_KEY].itertuples(index=False)))
    is_new = [key not in known for key in map(tuple, new[MATCH_KEY].itertuples(index=False))]
    frames = [f for f in (history, new[is_new]) if len(f)]
    if not frames:
        return history
    return pd.concat(frames, ignore_index=True)[PREDICTION_COLS]
