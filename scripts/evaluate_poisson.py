"""Evalúa el modelo de goles de Poisson contra los baselines y el mercado, con división temporal.

Dos mercados:
    1X2                    Poisson contra frecuencias históricas, Elo fijo y el mercado.
    Más/menos de 2.5 goles Poisson contra la frecuencia histórica y el mercado.

Misma ventana expansiva que scripts/evaluate_baselines.py (entrenar hasta la temporada N, probar en
la N+1). El Poisson se reajusta en cada fecha con todos los partidos anteriores. Guarda:
    reports/poisson_metrics.md              resumen legible
    reports/poisson_metrics_by_season.csv   una fila por temporada de prueba, mercado y modelo
    reports/poisson_calibration.png         curva de calibración del 1X2

No hay nada aleatorio, así que el resultado es el mismo en cada corrida.

Uso (desde la raíz del repo):
    python scripts/evaluate_poisson.py
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
from sports_predictor.football.poisson import (  # noqa: E402
    outcome_probabilities,
    over_probability,
    score_matrices,
    walk_forward_expected_goals,
)
from sports_predictor.metrics import brier_score, log_loss, rps  # noqa: E402
from sports_predictor.odds import implied_probabilities  # noqa: E402
from sports_predictor.validation import expanding_season_splits  # noqa: E402

MIN_TRAIN_SEASONS = 3
ELO_K = 20.0
ELO_HOME_ADVANTAGE = 60.0
POISSON_PENALTY = 10.0
MODELS_1X2 = ["Frecuencias históricas", "Elo fijo", "Poisson", "Mercado: promedio"]
MODELS_OU = ["Frecuencia histórica", "Poisson", "Mercado: promedio"]


def main() -> None:
    matches = load_matches(ROOT / "data" / "raw")
    seasons = list(matches["Season"].unique())  # ya vienen en orden cronológico
    splits = list(expanding_season_splits(seasons, MIN_TRAIN_SEASONS))

    outcomes = outcome_index(matches["FTR"])
    over = ((matches["FTHG"] + matches["FTAG"]) > 2.5).to_numpy().astype(int)  # 1 = más de 2.5 goles

    elo_diffs = elo_rating_diffs(matches, k=ELO_K, home_advantage=ELO_HOME_ADVANTAGE)

    # Poisson: una sola pasada. La predicción de cada fecha solo usa partidos de fechas anteriores.
    first_test = int((matches["Season"] == splits[0][1]).to_numpy().argmax())
    rate_home, rate_away = walk_forward_expected_goals(matches, first_test, penalty=POISSON_PENALTY)

    rows = []
    pooled = {"1X2": {m: [] for m in MODELS_1X2}, "OU": {m: [] for m in MODELS_OU}}
    pooled_y = {"1X2": [], "OU": []}
    for train_seasons, test_season in splits:
        train = (matches["Season"].isin(train_seasons)).to_numpy()
        test = (matches["Season"] == test_season).to_numpy()
        assert matches.loc[train, "Date"].max() < matches.loc[test, "Date"].min()  # el pasado va antes del futuro

        test_matches = matches[test]
        matrices = score_matrices(rate_home[test], rate_away[test])
        p_over = over_probability(matrices)
        freq_over = over[train].mean()
        probs = {
            "1X2": {
                "Frecuencias históricas": frequency_probabilities(matches[train], int(test.sum())),
                "Elo fijo": elo_probabilities(elo_diffs[train], outcomes[train], elo_diffs[test]),
                "Poisson": outcome_probabilities(matrices),
                "Mercado: promedio": implied_probabilities(test_matches[["AvgH", "AvgD", "AvgA"]]),
            },
            # Columnas en orden (menos de 2.5, más de 2.5), para que la clase 1 sea "más de 2.5".
            "OU": {
                "Frecuencia histórica": np.tile([1 - freq_over, freq_over], (int(test.sum()), 1)),
                "Poisson": np.column_stack([1 - p_over, p_over]),
                "Mercado: promedio": implied_probabilities(test_matches[["AvgUnder25", "AvgOver25"]]),
            },
        }
        y = {"1X2": outcomes[test], "OU": over[test]}
        for market in probs:
            pooled_y[market].append(y[market])
            for model, p in probs[market].items():
                assert not np.isnan(p).any(), f"{model}: faltan probabilidades en {test_season}"
                pooled[market][model].append(p)
                rows.append({
                    "test_season": test_season,
                    "market": market,
                    "model": model,
                    "n_matches": int(test.sum()),
                    "log_loss": log_loss(p, y[market]),
                    "brier": brier_score(p, y[market]),
                    "rps": rps(p, y[market]) if market == "1X2" else float("nan"),
                })

    by_season = pd.DataFrame(rows)
    by_season.to_csv(ROOT / "reports" / "poisson_metrics_by_season.csv", index=False, float_format="%.5f")

    y_1x2 = np.concatenate(pooled_y["1X2"])
    y_ou = np.concatenate(pooled_y["OU"])
    stacked = {market: {m: np.vstack(ps) for m, ps in pooled[market].items()} for market in pooled}
    overall_1x2 = pd.DataFrame([
        {"Modelo": m, "Log loss": log_loss(p, y_1x2), "Brier": brier_score(p, y_1x2), "RPS": rps(p, y_1x2)}
        for m, p in stacked["1X2"].items()
    ])
    overall_ou = pd.DataFrame([
        {"Modelo": m, "Log loss": log_loss(p, y_ou), "Brier": brier_score(p, y_ou)}
        for m, p in stacked["OU"].items()
    ])

    tested = (matches["Season"].isin([s for _, s in splits])).to_numpy()
    goals = pd.DataFrame([
        {"Equipo": "Local", "Goles esperados (promedio)": rate_home[tested].mean(),
         "Goles reales (promedio)": matches.loc[tested, "FTHG"].mean()},
        {"Equipo": "Visitante", "Goles esperados (promedio)": rate_away[tested].mean(),
         "Goles reales (promedio)": matches.loc[tested, "FTAG"].mean()},
    ])

    calibration = pd.concat(
        [calibration_table(stacked["1X2"][m], y_1x2).assign(model=m) for m in ["Poisson", "Mercado: promedio"]],
        ignore_index=True,
    )
    _plot_calibration(calibration, ROOT / "reports" / "poisson_calibration.png")

    test_seasons = [s for _, s in splits]
    lines = [
        "# Modelo de goles: Poisson independiente",
        "",
        "Generado por `python scripts/evaluate_poisson.py`. No editar a mano.",
        "",
        f"- **Datos:** {len(matches)} partidos, temporadas {seasons[0]} a {seasons[-1]}.",
        f"- **Temporadas de prueba:** {test_seasons[0]} a {test_seasons[-1]} ({len(test_seasons)} temporadas, {len(y_1x2)} partidos), "
        "con la misma ventana expansiva que los baselines.",
        "- **Poisson:** cada equipo tiene una fuerza de ataque y una de defensa, más una ventaja de local común. "
        "Los goles de cada equipo se suponen Poisson e independientes. El modelo se reajusta en cada fecha con todos los "
        f"partidos anteriores, todos con el mismo peso. Penalización = {POISSON_PENALTY:g}, valor fijo sin ajustar.",
        f"- **Elo fijo:** K = {ELO_K:g}, ventaja de local = {ELO_HOME_ADVANTAGE:g}.",
        "- **Mercado:** promedio de las casas, con el margen removido (normalización proporcional).",
        "- En todas las métricas, menos es mejor.",
        "- **Limitaciones:** todavía no tiene la corrección de Dixon-Coles para marcadores bajos ni da más peso a los "
        "partidos recientes. Un equipo recién ascendido empieza como equipo promedio hasta que acumula partidos.",
        "",
        f"## Resultado 1X2 (los {len(y_1x2)} partidos de prueba juntos)",
        "",
        _markdown_table(overall_1x2),
        "",
        f"## Más/menos de 2.5 goles (los {len(y_ou)} partidos de prueba juntos)",
        "",
        f"En los partidos de prueba hubo más de 2.5 goles en el {over[tested].mean():.1%} de los casos.",
        "",
        _markdown_table(overall_ou),
        "",
        "## Goles esperados contra goles reales",
        "",
        _markdown_table(goals),
        "",
    ]
    for market, title, models in [("1X2", "1X2", MODELS_1X2), ("OU", "más/menos de 2.5 goles", MODELS_OU)]:
        pivot = by_season[by_season["market"] == market].pivot(index="test_season", columns="model", values="log_loss")[models]
        pivot.insert(0, "Temporada de prueba", pivot.index)
        lines += [f"## Log loss por temporada de prueba: {title}", "", _markdown_table(pivot), ""]
    lines += [
        "## Calibración del 1X2",
        "",
        "Se juntan las tres probabilidades de cada partido y se agrupan en tramos de 10 puntos. "
        "`predicho` es la probabilidad promedio del tramo y `observado` la frecuencia con la que el resultado ocurrió.",
        "",
        "![Curva de calibración](poisson_calibration.png)",
        "",
    ]
    for m in ["Poisson", "Mercado: promedio"]:
        t = calibration[calibration["model"] == m]
        table = pd.DataFrame({
            "Tramo": [f"{lo:.1f} a {hi:.1f}" for lo, hi in zip(t["bin_low"], t["bin_high"])],
            "n": t["n"].to_numpy(),
            "predicho": t["predicted"].to_numpy(),
            "observado": t["observed"].to_numpy(),
        })
        lines += [f"### {m}", "", _markdown_table(table), ""]
    (ROOT / "reports" / "poisson_metrics.md").write_text("\n".join(lines), encoding="utf-8")

    print("1X2")
    print(overall_1x2.to_string(index=False, float_format="%.4f"))
    print("\nMás/menos de 2.5 goles")
    print(overall_ou.to_string(index=False, float_format="%.4f"))
    print("\nGuardado en reports/: poisson_metrics.md, poisson_metrics_by_season.csv y poisson_calibration.png")


def _plot_calibration(calibration: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Calibración perfecta")
    for model, t in calibration.groupby("model", sort=False):
        ax.plot(t["predicted"], t["observed"], marker="o", label=model)
    ax.set_xlabel("Probabilidad predicha")
    ax.set_ylabel("Frecuencia observada")
    ax.set_title("Calibración del 1X2: Poisson y mercado (partidos de prueba)")
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
