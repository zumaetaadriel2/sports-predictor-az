"""Evaluación final: todos los modelos contra el mercado, con división temporal.

Modelos de 1X2, en el orden de la especificación:
    Frecuencias históricas, Elo fijo, Poisson, Dixon-Coles, Dixon-Coles con peso por antigüedad,
    LightGBM y el mercado (promedio de las casas, sin margen).
En más/menos de 2.5 goles se comparan los modelos de goles contra la frecuencia histórica y el mercado.

Ventana expansiva: se entrena con las temporadas hasta N y se prueba en la N+1. Todo lo que se elige
(la vida media del peso por antigüedad) se elige en cada ventana usando solo temporadas de entrenamiento.

Uso (desde la raíz del repo):
    python scripts/evaluate_models.py                # desarrollo: temporadas hasta 2025/26
    python scripts/evaluate_models.py --final-test   # prueba final: solo la temporada reservada 2026/27

Salidas del modo normal, en reports/: final_report.md, model_metrics_by_season.csv y model_calibration.png.
La prueba final escribe únicamente reports/final_test.md.
"""
import argparse
import sys
from pathlib import Path

import lightgbm as lgb
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
from sports_predictor.football.data import LAST_DEV_SEASON, load_matches  # noqa: E402
from sports_predictor.football.features import FEATURE_COLS, form_features  # noqa: E402
from sports_predictor.football.poisson import (  # noqa: E402
    outcome_probabilities,
    over_probability,
    score_matrices,
    walk_forward_expected_goals,
)
from sports_predictor.metrics import brier_score, log_loss, rps  # noqa: E402
from sports_predictor.odds import implied_probabilities  # noqa: E402
from sports_predictor.validation import expanding_season_splits  # noqa: E402

SEED = 0
MIN_TRAIN_SEASONS = 3
ELO_K = 20.0
ELO_HOME_ADVANTAGE = 60.0
POISSON_PENALTY = 10.0
# Vidas medias candidatas, en días. None = todos los partidos pesan igual.
HALF_LIVES = [None, 1095, 730, 365, 180]
# LightGBM: valores conservadores y fijos, sin ajustar (hay pocos miles de partidos).
LGBM_PARAMS = dict(
    objective="multiclass", n_estimators=150, learning_rate=0.03, num_leaves=4, min_child_samples=50,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
    random_state=SEED, n_jobs=1, deterministic=True, force_row_wise=True, verbose=-1,
)
GBM_FEATURES = ["elo_diff", "expected_goals_home", "expected_goals_away"] + FEATURE_COLS

FREQ, ELO, POISSON, DC, PW, DCW, GBM, MARKET = (
    "Frecuencias históricas", "Elo fijo", "Poisson", "Dixon-Coles", "Poisson + antigüedad",
    "Dixon-Coles + antigüedad", "LightGBM", "Mercado: promedio",
)
MODELS_1X2 = [FREQ, ELO, POISSON, DC, PW, DCW, GBM, MARKET]
MODELS_OU = ["Frecuencia histórica", POISSON, DC, PW, DCW, MARKET]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--final-test", action="store_true", help="Evalúa solo la temporada reservada (2026/27)")
    args = parser.parse_args()

    matches = load_matches(ROOT / "data" / "raw", include_holdout=args.final_test)
    seasons = list(matches["Season"].unique())  # ya vienen en orden cronológico
    splits = list(expanding_season_splits(seasons, MIN_TRAIN_SEASONS))
    if args.final_test:
        if seasons[-1] <= LAST_DEV_SEASON:
            sys.exit("No hay temporada reservada en data/raw/. Corre: python scripts/download_data.py --include-current")
        splits = splits[-1:]  # solo la última ventana: entrenar con todo lo anterior, probar en la reservada

    outcomes = outcome_index(matches["FTR"])
    over = ((matches["FTHG"] + matches["FTAG"]) > 2.5).to_numpy().astype(int)  # 1 = más de 2.5 goles
    season = matches["Season"].to_numpy()

    elo_diffs = elo_rating_diffs(matches, k=ELO_K, home_advantage=ELO_HOME_ADVANTAGE)
    form = form_features(matches)

    # Modelos de goles: una pasada por cada vida media, desde la segunda temporada. La predicción de
    # cada fecha solo usa partidos anteriores, así que también sirve para validar dentro del entrenamiento.
    first = int((season == seasons[1]).argmax())
    goals = {h: walk_forward_expected_goals(matches, first, POISSON_PENALTY, h) for h in HALF_LIVES}
    dc_probs = {h: np.full((len(matches), 3), np.nan) for h in HALF_LIVES}
    for h, (rate_home, rate_away, rho) in goals.items():
        dc_probs[h][first:] = outcome_probabilities(score_matrices(rate_home[first:], rate_away[first:], rho[first:]))

    rows, chosen, importances = [], [], []
    pooled = {"1X2": {m: [] for m in MODELS_1X2}, "OU": {m: [] for m in MODELS_OU}}
    pooled_y = {"1X2": [], "OU": []}
    for train_seasons, test_season in splits:
        train = np.isin(season, train_seasons)
        test = season == test_season
        assert matches.loc[train, "Date"].max() < matches.loc[test, "Date"].min()  # el pasado va antes del futuro
        n_test = int(test.sum())

        # Vida media: la de menor log loss 1X2 en las temporadas de entrenamiento que ya tienen predicción.
        val = train & (np.arange(len(matches)) >= first)
        best = min(HALF_LIVES, key=lambda h: log_loss(dc_probs[h][val], outcomes[val]))
        chosen.append({"Temporada de prueba": test_season, "Vida media elegida (días)": "sin peso" if best is None else str(best)})

        def matrices(half_life, dixon_coles):
            rate_home, rate_away, rho = goals[half_life]
            return score_matrices(rate_home[test], rate_away[test], rho[test] if dixon_coles else None)

        m_poisson, m_dc = matrices(None, False), matrices(None, True)
        m_pw, m_dcw = matrices(best, False), matrices(best, True)

        # LightGBM: Elo, goles esperados del modelo de goles elegido y forma reciente. Todo es previo al partido.
        features = form.assign(
            elo_diff=elo_diffs, expected_goals_home=goals[best][0], expected_goals_away=goals[best][1]
        )[GBM_FEATURES]
        gbm = lgb.LGBMClassifier(**LGBM_PARAMS).fit(features[train], outcomes[train])
        gain = gbm.booster_.feature_importance(importance_type="gain")
        importances.append(gain / gain.sum())

        test_matches = matches[test]
        freq_over = over[train].mean()
        probs = {
            "1X2": {
                FREQ: frequency_probabilities(matches[train], n_test),
                ELO: elo_probabilities(elo_diffs[train], outcomes[train], elo_diffs[test]),
                POISSON: outcome_probabilities(m_poisson),
                DC: outcome_probabilities(m_dc),
                PW: outcome_probabilities(m_pw),
                DCW: outcome_probabilities(m_dcw),
                GBM: gbm.predict_proba(features[test]),
                MARKET: implied_probabilities(test_matches[["AvgH", "AvgD", "AvgA"]]),
            },
            # Columnas en orden (menos de 2.5, más de 2.5), para que la clase 1 sea "más de 2.5".
            "OU": {
                "Frecuencia histórica": np.tile([1 - freq_over, freq_over], (n_test, 1)),
                POISSON: _two_columns(over_probability(m_poisson)),
                DC: _two_columns(over_probability(m_dc)),
                PW: _two_columns(over_probability(m_pw)),
                DCW: _two_columns(over_probability(m_dcw)),
                MARKET: implied_probabilities(test_matches[["AvgUnder25", "AvgOver25"]]),
            },
        }
        y = {"1X2": outcomes[test], "OU": over[test]}
        for market in probs:
            pooled_y[market].append(y[market])
            for model, p in probs[market].items():
                assert not np.isnan(p).any(), f"{model}: faltan probabilidades en {test_season}"
                pooled[market][model].append(p)
                rows.append({
                    "test_season": test_season, "market": market, "model": model, "n_matches": n_test,
                    "log_loss": log_loss(p, y[market]), "brier": brier_score(p, y[market]),
                    "rps": rps(p, y[market]) if market == "1X2" else float("nan"),
                })

    by_season = pd.DataFrame(rows)
    y_1x2, y_ou = np.concatenate(pooled_y["1X2"]), np.concatenate(pooled_y["OU"])
    stacked = {market: {m: np.vstack(ps) for m, ps in pooled[market].items()} for market in pooled}
    overall_1x2 = pd.DataFrame([
        {"Modelo": m, "Log loss": log_loss(p, y_1x2), "Brier": brier_score(p, y_1x2), "RPS": rps(p, y_1x2)}
        for m, p in stacked["1X2"].items()
    ])
    overall_ou = pd.DataFrame([
        {"Modelo": m, "Log loss": log_loss(p, y_ou), "Brier": brier_score(p, y_ou)} for m, p in stacked["OU"].items()
    ])
    setup = [
        f"- **Elo fijo:** K = {ELO_K:g}, ventaja de local = {ELO_HOME_ADVANTAGE:g}.",
        f"- **Modelos de goles:** penalización = {POISSON_PENALTY:g}, reajuste en cada fecha con los partidos anteriores. "
        "Dixon-Coles corrige los marcadores 0-0, 1-0, 0-1 y 1-1. El peso por antigüedad hace que un partido cuente la mitad "
        "cada vez que pasa una vida media; la vida media se elige en cada ventana, entre "
        f"{['sin peso' if h is None else h for h in HALF_LIVES]} días, con el log loss de las temporadas de entrenamiento.",
        f"- **LightGBM:** {len(GBM_FEATURES)} variables previas al partido: diferencia de Elo, goles esperados del modelo de "
        f"goles, y forma de los últimos 5 partidos de cada equipo (puntos, goles, tiros a puerta, puntos en su condición de "
        f"local o visitante, días de descanso). Hiperparámetros fijos, sin ajustar; semilla {SEED}.",
        "- **Mercado:** promedio de las casas, con el margen removido (normalización proporcional).",
        "- En todas las métricas, menos es mejor.",
    ]

    if args.final_test:
        test_season = splits[0][1]
        test_dates = matches.loc[season == test_season, "Date"]
        lines = [
            f"# Prueba final: temporada {test_season} (parcial)",
            "",
            "Generado por `python scripts/evaluate_models.py --final-test`. No editar a mano.",
            "",
            f"Esta temporada estuvo reservada: ningún modelo ni decisión de diseño la usó. Se entrena con {seasons[0]} a "
            f"{splits[0][0][-1]} y se prueba en los {len(y_1x2)} partidos jugados entre el {test_dates.min().date()} y el "
            f"{test_dates.max().date()}. Con tan pocos partidos, las diferencias pequeñas entre modelos no son concluyentes.",
            "",
            *setup,
            f"- **Vida media elegida:** {chosen[0]['Vida media elegida (días)']}.",
            "",
            "## Resultado 1X2", "", _markdown_table(overall_1x2), "",
            "## Más/menos de 2.5 goles", "", _markdown_table(overall_ou), "",
        ]
        (ROOT / "reports" / "final_test.md").write_text("\n".join(lines), encoding="utf-8")
        print(overall_1x2.to_string(index=False, float_format="%.4f"))
        print(overall_ou.to_string(index=False, float_format="%.4f"))
        print("\nGuardado en reports/final_test.md")
        return

    by_season.to_csv(ROOT / "reports" / "model_metrics_by_season.csv", index=False, float_format="%.7f")
    calibrated = [ELO, DCW, GBM, MARKET]
    calibration = pd.concat(
        [calibration_table(stacked["1X2"][m], y_1x2).assign(model=m) for m in calibrated], ignore_index=True
    )
    _plot_calibration(calibration, ROOT / "reports" / "model_calibration.png")
    importance = pd.DataFrame({"Variable": GBM_FEATURES, "Importancia (ganancia)": np.mean(importances, axis=0)})
    importance = importance.sort_values("Importancia (ganancia)", ascending=False)

    test_seasons = [s for _, s in splits]
    lines = [
        "# Informe final: resultado 1X2 y goles de la Premier League",
        "",
        "Generado por `python scripts/evaluate_models.py`. No editar a mano.",
        "",
        f"- **Datos:** {len(matches)} partidos, temporadas {seasons[0]} a {seasons[-1]}.",
        f"- **División temporal (ventana expansiva):** se entrena con las temporadas hasta N y se prueba en la N+1. "
        f"Temporadas de prueba: {test_seasons[0]} a {test_seasons[-1]} ({len(test_seasons)} temporadas, {len(y_1x2)} partidos).",
        *setup,
        "",
        f"## Resultado 1X2 (los {len(y_1x2)} partidos de prueba juntos)", "", _markdown_table(overall_1x2), "",
        f"## Más/menos de 2.5 goles (los {len(y_ou)} partidos de prueba juntos)", "", _markdown_table(overall_ou), "",
        "## Vida media elegida en cada ventana", "", _markdown_table(pd.DataFrame(chosen)), "",
    ]
    for market, title, models in [("1X2", "1X2", MODELS_1X2), ("OU", "más/menos de 2.5 goles", MODELS_OU)]:
        pivot = by_season[by_season["market"] == market].pivot(index="test_season", columns="model", values="log_loss")[models]
        pivot.insert(0, "Temporada de prueba", pivot.index)
        lines += [f"## Log loss por temporada de prueba: {title}", "", _markdown_table(pivot), ""]
    lines += [
        "## Variables más importantes de LightGBM",
        "",
        "Parte de la mejora total del modelo que aporta cada variable, promediada entre las ventanas.",
        "",
        _markdown_table(importance),
        "",
        "## Calibración del 1X2",
        "",
        "Se juntan las tres probabilidades de cada partido y se agrupan en tramos de 10 puntos. "
        "`predicho` es la probabilidad promedio del tramo y `observado` la frecuencia con la que el resultado ocurrió.",
        "",
        "![Curva de calibración](model_calibration.png)",
        "",
    ]
    for m in calibrated:
        t = calibration[calibration["model"] == m]
        table = pd.DataFrame({
            "Tramo": [f"{lo:.1f} a {hi:.1f}" for lo, hi in zip(t["bin_low"], t["bin_high"])],
            "n": t["n"].to_numpy(), "predicho": t["predicted"].to_numpy(), "observado": t["observed"].to_numpy(),
        })
        lines += [f"### {m}", "", _markdown_table(table), ""]
    (ROOT / "reports" / "final_report.md").write_text("\n".join(lines), encoding="utf-8")

    print("1X2")
    print(overall_1x2.to_string(index=False, float_format="%.4f"))
    print("\nMás/menos de 2.5 goles")
    print(overall_ou.to_string(index=False, float_format="%.4f"))
    print("\nGuardado en reports/: final_report.md, model_metrics_by_season.csv y model_calibration.png")


def _two_columns(p_over) -> np.ndarray:
    return np.column_stack([1 - p_over, p_over])


def _plot_calibration(calibration: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Calibración perfecta")
    for model, t in calibration.groupby("model", sort=False):
        ax.plot(t["predicted"], t["observed"], marker="o", label=model)
    ax.set_xlabel("Probabilidad predicha")
    ax.set_ylabel("Frecuencia observada")
    ax.set_title("Calibración del 1X2 (partidos de prueba)")
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
