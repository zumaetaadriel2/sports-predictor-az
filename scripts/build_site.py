"""Genera el sitio estático de la demo en _site/ (HTML, CSS y gráficos).

Lee los partidos de data/raw/ (incluida la temporada en curso), el historial de predicciones
publicadas (site_data/predictions.csv) y los resultados de la evaluación (reports/).
No entrena nada nuevo salvo el modelo Dixon-Coles + antigüedad de la temporada en curso.

Uso (desde la raíz del repo):
    python scripts/build_site.py
    python scripts/build_site.py --as-of 2026-11-05 --history otra/ruta.csv   # ver el sitio como en esa fecha
"""
import argparse
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventana: solo guarda el archivo
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from jinja2 import Environment, FileSystemLoader, select_autoescape  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sports_predictor import __version__  # noqa: E402
from sports_predictor.football.baselines import outcome_index  # noqa: E402
from sports_predictor.football.data import load_matches  # noqa: E402
from sports_predictor.football.poisson import (  # noqa: E402
    outcome_probabilities,
    score_matrices,
    walk_forward_expected_goals,
)
from sports_predictor.football.predict import (  # noqa: E402
    HALF_LIFE_DAYS,
    MATCH_KEY,
    PENALTY,
    fit_goal_model,
    read_history,
    team_strengths,
)
from sports_predictor.metrics import log_loss  # noqa: E402
from sports_predictor.odds import implied_probabilities  # noqa: E402

OUT = ROOT / "_site"
REPO_URL = "https://github.com/zumaetaadriel2/sports-predictor-az"
OUTCOME_NAMES = {"H": "Local", "D": "Empate", "A": "Visitante"}
MODEL_COLS = ["p_home", "p_draw", "p_away"]
MARKET_COLS = ["market_home", "market_draw", "market_away"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--as-of", default=None, help="Construye el sitio como en esa fecha AAAA-MM-DD (por defecto, hoy en UTC)")
    parser.add_argument("--history", default=str(ROOT / "site_data" / "predictions.csv"), help="Archivo de historial")
    args = parser.parse_args()
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()

    matches = load_matches(ROOT / "data" / "raw", include_holdout=True)
    matches = matches[matches["Date"] < as_of].reset_index(drop=True)
    season = matches["Season"].iloc[-1]
    history = read_history(args.history)
    history = history[history["published_at"] <= as_of]

    # Predicciones publicadas: las que ya tienen resultado y las que siguen pendientes.
    merged = history.merge(matches[MATCH_KEY + ["FTHG", "FTAG", "FTR"]], on=MATCH_KEY, how="left")
    upcoming = merged[merged["FTR"].isna() & (merged["Date"] >= as_of)].sort_values("Date")
    settled = merged[merged["FTR"].notna()].sort_values("Date", ascending=False)

    context = {
        "version": __version__,
        "updated": _date(as_of),
        "repo_url": REPO_URL,
        "season": season,
        "upcoming": [_card(row) for row in upcoming.itertuples(index=False)],
        "settled": [_settled_row(row) for row in settled.head(20).itertuples(index=False)],
        "settled_summary": _settled_summary(settled),
        "backtest": _season_backtest(matches, season, OUT / "season_log_loss.png"),
        "strengths": _strengths(matches, season, as_of),
        "models": _historical_models(),
        "half_life": int(HALF_LIFE_DAYS),
    }

    OUT.mkdir(exist_ok=True)
    env = Environment(loader=FileSystemLoader(ROOT / "site" / "templates"), autoescape=select_autoescape(["html"]))
    for page in ("index.html", "metodologia.html"):
        (OUT / page).write_text(env.get_template(page).render(**context), encoding="utf-8")
    shutil.copy(ROOT / "site" / "static" / "style.css", OUT / "style.css")
    shutil.copy(ROOT / "reports" / "model_calibration.png", OUT / "model_calibration.png")
    print(f"Sitio generado en {OUT.relative_to(ROOT)}/ con fecha {as_of.date()}: "
          f"{len(upcoming)} partidos próximos, {len(settled)} predicciones publicadas con resultado.")


def _date(value) -> str:
    return pd.Timestamp(value).strftime("%d/%m/%Y")


def _percent(values) -> list[int]:
    """Porcentajes enteros que suman 100 (para que las barras no dejen huecos)."""
    values = np.asarray(values, dtype=float)
    if np.isnan(values).any():
        return []
    scaled = values / values.sum() * 100
    floors = np.floor(scaled).astype(int)
    for i in np.argsort(scaled - floors)[::-1][: 100 - floors.sum()]:
        floors[i] += 1
    return [int(v) for v in floors]


def _card(row) -> dict:
    return {
        "date": _date(row.Date), "home": row.HomeTeam, "away": row.AwayTeam,
        "model": _percent([row.p_home, row.p_draw, row.p_away]),
        "market": _percent([row.market_home, row.market_draw, row.market_away]),
        "goals_home": f"{row.exp_goals_home:.2f}", "goals_away": f"{row.exp_goals_away:.2f}",
        "over": f"{row.p_over25:.0%}",
        "market_over": "" if np.isnan(row.market_over25) else f"{row.market_over25:.0%}",
    }


def _settled_row(row) -> dict:
    card = _card(row)
    card["score"] = f"{int(row.FTHG)}-{int(row.FTAG)}"
    card["result"] = OUTCOME_NAMES[row.FTR]
    card["p_result"] = f"{[row.p_home, row.p_draw, row.p_away]['HDA'.index(row.FTR)]:.0%}"
    return card


def _settled_summary(settled: pd.DataFrame):
    """Log loss del modelo y del mercado en las predicciones publicadas que ya tienen resultado."""
    with_market = settled.dropna(subset=MARKET_COLS)
    if not len(with_market):
        return None
    outcomes = outcome_index(with_market["FTR"])
    return {
        "n": len(with_market),
        "model": f"{log_loss(with_market[MODEL_COLS].to_numpy(), outcomes):.4f}",
        "market": f"{log_loss(with_market[MARKET_COLS].to_numpy(), outcomes):.4f}",
    }


def _season_backtest(matches: pd.DataFrame, season: str, chart: Path):
    """Modelo contra mercado en la temporada en curso, recalculado partido a partido.

    Cada predicción solo usa partidos anteriores a su fecha, pero se calcula ahora: es una
    reconstrucción, no un registro de lo que se publicó en su momento.
    """
    first = int((matches["Season"] == season).to_numpy().argmax())
    rate_home, rate_away, rho = walk_forward_expected_goals(matches, first, PENALTY, HALF_LIFE_DAYS)
    current = matches.iloc[first:]
    model = outcome_probabilities(score_matrices(rate_home[first:], rate_away[first:], rho[first:]))
    market = implied_probabilities(current[["AvgH", "AvgD", "AvgA"]])
    keep = ~np.isnan(market).any(axis=1)
    if not keep.any():
        return None
    outcomes = outcome_index(current["FTR"])[keep]
    model, market = model[keep], market[keep]

    rows = np.arange(len(outcomes))
    counts = rows + 1
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for probs, label in [(model, "Modelo"), (market, "Mercado")]:
        ax.plot(counts, np.cumsum(-np.log(probs[rows, outcomes])) / counts, label=label)
    ax.set_xlabel("Partidos jugados en la temporada")
    ax.set_ylabel("Log loss acumulado (menos es mejor)")
    ax.legend()
    ax.grid(alpha=0.3)
    chart.parent.mkdir(exist_ok=True)
    fig.savefig(chart, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return {
        "n": len(outcomes), "chart": chart.name,
        "model": f"{log_loss(model, outcomes):.4f}", "market": f"{log_loss(market, outcomes):.4f}",
    }


def _strengths(matches: pd.DataFrame, season: str, as_of) -> list[dict]:
    current = matches[matches["Season"] == season]
    teams = sorted(set(current["HomeTeam"]) | set(current["AwayTeam"]))
    table = team_strengths(fit_goal_model(matches, as_of), teams)
    return [
        {"rank": i + 1, "team": r.team, "goals_for": f"{r.goals_for:.2f}", "goals_against": f"{r.goals_against:.2f}",
         "difference": f"{r.difference:+.2f}"}
        for i, r in enumerate(table.itertuples(index=False))
    ]


def _historical_models() -> list[dict]:
    """Métricas 1X2 de la evaluación histórica. Todas las temporadas de prueba tienen 380 partidos,
    así que el promedio por temporada es igual al valor global del informe."""
    metrics = pd.read_csv(ROOT / "reports" / "model_metrics_by_season.csv")
    table = metrics[metrics["market"] == "1X2"].groupby("model")[["log_loss", "brier", "rps"]].mean()
    table = table.sort_values("log_loss", ascending=False)
    return [
        {"name": name, "log_loss": f"{r.log_loss:.4f}", "brier": f"{r.brier:.4f}", "rps": f"{r.rps:.4f}",
         "highlight": name == "Dixon-Coles + antigüedad", "market": name.startswith("Mercado")}
        for name, r in zip(table.index, table.itertuples(index=False))
    ]


if __name__ == "__main__":
    main()
