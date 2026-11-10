"""Baselines de fútbol para el resultado 1X2: frecuencias históricas y Elo con ventaja de local.

Las probabilidades siempre van en el orden de OUTCOMES: local (H), empate (D), visitante (A).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

OUTCOMES = ["H", "D", "A"]


def outcome_index(ftr: pd.Series) -> np.ndarray:
    """'H'/'D'/'A' -> 0/1/2."""
    return ftr.map({o: i for i, o in enumerate(OUTCOMES)}).to_numpy()


def frequency_probabilities(train: pd.DataFrame, n_test: int) -> np.ndarray:
    """Baseline ingenuo: a todo partido le asigna la frecuencia de H/D/A vista en entrenamiento."""
    freq = train["FTR"].value_counts(normalize=True).reindex(OUTCOMES, fill_value=0.0).to_numpy()
    return np.tile(freq, (n_test, 1))


def elo_rating_diffs(
    matches: pd.DataFrame,
    k: float = 20.0,
    home_advantage: float = 60.0,
    initial: float = 1500.0,
) -> np.ndarray:
    """Diferencia de Elo antes de cada partido: (local + ventaja de local) - visitante.

    `matches` debe venir en orden cronológico. El valor de cada partido se calcula ANTES de ver su
    resultado y solo depende de partidos anteriores, así que no hay fuga de datos.

    - Resultado esperado del local: E = 1 / (1 + 10 ** (-diff / 400)).
    - Tras el partido: rating_local += k * (S - E) y rating_visitante -= k * (S - E),
      con S = 1 si gana el local, 0.5 si empatan, 0 si pierde.
    - Un equipo que aparece por primera vez (ascendido) entra con el promedio de los ratings de los
      equipos que dejaron la liga la temporada anterior. En la primera temporada todos entran con `initial`.
      Para saber quién juega cada temporada se usa la lista de equipos del calendario, que se conoce
      antes de que empiece; no se usa ningún resultado.
    """
    ratings: dict[str, float] = {}
    diffs = np.empty(len(matches))
    prev_teams: set[str] = set()
    season = None
    score = {"H": 1.0, "D": 0.5, "A": 0.0}

    for i, row in enumerate(matches.itertuples(index=False)):
        if row.Season != season:
            season = row.Season
            in_season = matches["Season"].to_numpy() == season
            teams = set(matches.loc[in_season, "HomeTeam"]) | set(matches.loc[in_season, "AwayTeam"])
            gone = [ratings[t] for t in prev_teams - teams]
            entry = float(np.mean(gone)) if gone else initial
            for team in teams - prev_teams:
                ratings[team] = entry
            prev_teams = teams

        diff = ratings[row.HomeTeam] + home_advantage - ratings[row.AwayTeam]
        diffs[i] = diff
        expected = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
        delta = k * (score[row.FTR] - expected)
        ratings[row.HomeTeam] += delta
        ratings[row.AwayTeam] -= delta

    return diffs


def elo_probabilities(train_diffs, train_outcomes, test_diffs) -> np.ndarray:
    """Convierte la diferencia de Elo en probabilidades H/D/A.

    El Elo solo da un "resultado esperado" entre 0 y 1; no separa el empate. Para obtener tres
    probabilidades se ajusta una regresión logística multinomial con una sola variable (la
    diferencia de Elo), usando únicamente los partidos de entrenamiento.
    """
    scale = 400.0  # deja la variable en un rango cómodo para el optimizador
    model = LogisticRegression(C=1e6, max_iter=1000)  # C grande: prácticamente sin regularización
    model.fit(np.asarray(train_diffs).reshape(-1, 1) / scale, train_outcomes)
    # predict_proba devuelve las columnas en el orden de las clases 0, 1, 2, es decir H, D, A.
    return model.predict_proba(np.asarray(test_diffs).reshape(-1, 1) / scale)
