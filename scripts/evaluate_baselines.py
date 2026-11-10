"""Evalúa los baselines (frecuencias históricas y Elo) contra el mercado, con división temporal.

Ventana expansiva: se entrena con las temporadas hasta N y se prueba en la N+1. Guarda:
    reports/baseline_metrics_by_season.csv   una fila por temporada de prueba y modelo
    reports/baseline_metrics.md              resumen legible
    reports/baseline_calibration.csv         tabla de calibración por modelo
    reports/baseline_calibration.png         curva de calibración

No hay nada aleatorio en estos modelos, así que el resultado es el mismo en cada corrida.

Uso (desde la raíz del repo):
    python scripts/evaluate_baselines.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventana: solo guarda el archivo
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sports_predictor.calibration import calibration_table  # noqa: E402
from sports_predictor.football.baselines import (  # noqa: E402
    elo_probabilities,
    elo_rating_diffs,
    frequency_probabilities,
    outcome_index,
)
from sports_predictor.football.data import load_matches  # noqa: E402
from sports_predictor.metrics import brier_score, log_loss, rps  # noqa: E402
from sports_predictor.odds import implied_probabilities  # noqa: E402
from sports_predictor.validation import expanding_season_splits  # noqa: E402

MIN_TRAIN_SEASONS = 3
ELO_K = 20.0
ELO_HOME_ADVANTAGE = 60.0
MODELS = ["Frecuencias históricas", "Elo con ventaja de local", "Mercado: promedio", "Mercado: Bet365"]


def main() -> None:
    matches = load_matches(ROOT / "data" / "raw")
    outcomes = outcome_index(matches["FTR"])
    # El Elo se calcula una sola vez sobre toda la historia: el valor de cada partido solo usa partidos anteriores.
    diffs = elo_rating_diffs(matches, k=ELO_K, home_advantage=ELO_HOME_ADVANTAGE)
    seasons = list(matches["Season"].unique())  # ya vienen en orden cronológico

    rows, pooled = [], {m: [] for m in MODELS}
    pooled_outcomes = []
    for train_seasons, test_season in expanding_season_splits(seasons, MIN_TRAIN_SEASONS):
        train = (matches["Season"].isin(train_seasons)).to_numpy()
        test = (matches["Season"] == test_season).to_numpy()
        assert matches.loc[train, "Date"].max() < matches.loc[test, "Date"].min()  # el pasado va antes del futuro

        test_matches = matches[test]
        probs = {
            "Frecuencias históricas": frequency_probabilities(matches[train], int(test.sum())),
            "Elo con ventaja de local": elo_probabilities(diffs[train], outcomes[train], diffs[test]),
            "Mercado: promedio": implied_probabilities(test_matches[["AvgH", "AvgD", "AvgA"]]),
            "Mercado: Bet365": implied_probabilities(test_matches[["B365H", "B365D", "B365A"]]),
        }
        pooled_outcomes.append(outcomes[test])
        for model, p in probs.items():
            assert not np.isnan(p).any(), f"{model}: faltan probabilidades en {test_season}"
            pooled[model].append(p)
            rows.append({
                "test_season": test_season,
                "train_seasons": len(train_seasons),
                "n_matches": int(test.sum()),
                "model": model,
                "log_loss": log_loss(p, outcomes[test]),
                "brier": brier_score(p, outcomes[test]),
                "rps": rps(p, outcomes[test]),
            })

    by_season = pd.DataFrame(rows)
    by_season.to_csv(ROOT / "reports" / "baseline_metrics_by_season.csv", index=False, float_format="%.5f")

    y = np.concatenate(pooled_outcomes)
    overall = pd.DataFrame([
        {
            "Modelo": m,
            "Log loss": log_loss(np.vstack(pooled[m]), y),
            "Brier": brier_score(np.vstack(pooled[m]), y),
            "RPS": rps(np.vstack(pooled[m]), y),
        }
        for m in MODELS
    ])

    test_seasons = list(by_season["test_season"].unique())
    lines = [
        "# Baselines: resultado 1X2 de la Premier League",
        "",
        "Generado por `python scripts/evaluate_baselines.py`. No editar a mano.",
        "",
        f"- **Datos:** {len(matches)} partidos, temporadas {seasons[0]} a {seasons[-1]}.",
        f"- **División temporal (ventana expansiva):** se entrena con todas las temporadas hasta N y se prueba en la N+1. "
        f"La primera prueba usa {MIN_TRAIN_SEASONS} temporadas de entrenamiento.",
        f"- **Temporadas de prueba:** {test_seasons[0]} a {test_seasons[-1]} ({len(test_seasons)} temporadas, {len(y)} partidos).",
        f"- **Elo:** K = {ELO_K:g}, ventaja de local = {ELO_HOME_ADVANTAGE:g} puntos. Valores fijos, sin ajustar.",
        "- **Mercado:** probabilidades implícitas de las cuotas con el margen removido (normalización proporcional).",
        "- En las tres métricas, menos es mejor.",
        "- **Limitaciones:** todos los equipos empiezan con el mismo Elo en la primera temporada, así que los primeros "
        "partidos de entrenamiento aportan diferencias poco informativas.",
        "",
        f"## Global (los {len(y)} partidos de prueba juntos)",
        "",
        _markdown_table(overall),
        "",
    ]
    for metric, title in [("log_loss", "Log loss"), ("brier", "Brier"), ("rps", "RPS")]:
        pivot = by_season.pivot(index="test_season", columns="model", values=metric)[MODELS]
        pivot.insert(0, "Temporada de prueba", pivot.index)
        lines += [f"## {title} por temporada de prueba", "", _markdown_table(pivot), ""]
    calibration = pd.concat(
        [calibration_table(np.vstack(pooled[m]), y).assign(model=m) for m in MODELS], ignore_index=True
    )
    calibration.to_csv(ROOT / "reports" / "baseline_calibration.csv", index=False, float_format="%.5f")
    _plot_calibration(calibration, ROOT / "reports" / "baseline_calibration.png")
    lines += [
        "## Calibración",
        "",
        "Se juntan las tres probabilidades de cada partido (local, empate, visitante) y se agrupan en tramos de 10 puntos. "
        "`predicho` es la probabilidad promedio que dio el modelo en el tramo; `observado` es la frecuencia con la que "
        "ese resultado ocurrió. Un modelo bien calibrado tiene ambos valores parecidos. Los tramos con pocos casos "
        "(`n` bajo) son ruidosos.",
        "",
        "![Curva de calibración](baseline_calibration.png)",
        "",
    ]
    for m in MODELS:
        t = calibration[calibration["model"] == m]
        table = pd.DataFrame({
            "Tramo": [f"{lo:.1f} a {hi:.1f}" for lo, hi in zip(t["bin_low"], t["bin_high"])],
            "n": t["n"].to_numpy(),
            "predicho": t["predicted"].to_numpy(),
            "observado": t["observed"].to_numpy(),
        })
        lines += [f"### {m}", "", _markdown_table(table), ""]
    (ROOT / "reports" / "baseline_metrics.md").write_text("\n".join(lines), encoding="utf-8")

    print(overall.to_string(index=False, float_format="%.4f"))
    print("\nGuardado en reports/: baseline_metrics.md, baseline_metrics_by_season.csv, "
          "baseline_calibration.csv y baseline_calibration.png")


def _plot_calibration(calibration: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Calibración perfecta")
    for model, t in calibration.groupby("model", sort=False):
        ax.plot(t["predicted"], t["observed"], marker="o", label=model)
    ax.set_xlabel("Probabilidad predicha")
    ax.set_ylabel("Frecuencia observada")
    ax.set_title("Calibración de los baselines (partidos de prueba)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _markdown_table(df: pd.DataFrame) -> str:
    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep] + body)


if __name__ == "__main__":
    main()
