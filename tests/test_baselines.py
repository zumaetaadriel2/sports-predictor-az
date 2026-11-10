import numpy as np
import pandas as pd
import pytest

from sports_predictor.football.baselines import (
    elo_probabilities,
    elo_rating_diffs,
    frequency_probabilities,
    outcome_index,
)


def make_matches(rows):
    return pd.DataFrame(rows, columns=["Season", "HomeTeam", "AwayTeam", "FTR"])


MATCHES = make_matches([
    ("s1", "A", "B", "H"),
    ("s1", "B", "C", "D"),
    ("s1", "C", "A", "A"),
    ("s1", "B", "A", "A"),
    ("s2", "A", "B", "H"),  # C desciende, D asciende
    ("s2", "D", "A", "D"),
    ("s2", "B", "D", "H"),
])


def test_outcome_index():
    assert list(outcome_index(pd.Series(["H", "D", "A"]))) == [0, 1, 2]


def test_frequency_probabilities():
    train = pd.DataFrame({"FTR": ["H", "H", "D", "A"]})
    probs = frequency_probabilities(train, n_test=3)
    assert probs.shape == (3, 3)
    assert probs[0] == pytest.approx([0.5, 0.25, 0.25])


def test_elo_first_matches_by_hand():
    diffs = elo_rating_diffs(MATCHES, k=20, home_advantage=60, initial=1500)
    # Partido 1: todos en 1500, así que la diferencia es solo la ventaja de local.
    assert diffs[0] == pytest.approx(60)
    # Tras ganar A: delta = 20 * (1 - E(60)).
    delta = 20 * (1 - 1 / (1 + 10 ** (-60 / 400)))
    # Partido 2: B (1500 - delta) recibe a C (1500).
    assert diffs[1] == pytest.approx(60 - delta)


def test_elo_is_zero_sum_and_promoted_team_gets_relegated_average():
    k, hfa = 20, 60
    diffs = elo_rating_diffs(MATCHES, k=k, home_advantage=hfa)
    # Se reconstruyen los ratings a mano a partir de las diferencias previas a cada partido.
    ratings = {"A": 1500.0, "B": 1500.0, "C": 1500.0}
    score = {"H": 1.0, "D": 0.5, "A": 0.0}
    for (season, home, away, ftr), diff in zip(MATCHES[:4].itertuples(index=False), diffs[:4]):
        assert diff == pytest.approx(ratings[home] + hfa - ratings[away])
        delta = k * (score[ftr] - 1 / (1 + 10 ** (-diff / 400)))
        ratings[home] += delta
        ratings[away] -= delta
    assert sum(ratings.values()) == pytest.approx(4500)
    # s2, partido 2: D entra con el rating final de C (el único que salió). A ya jugó un partido en s2.
    delta = k * (1 - 1 / (1 + 10 ** (-diffs[4] / 400)))
    assert diffs[5] == pytest.approx(ratings["C"] + hfa - (ratings["A"] + delta))


def test_elo_has_no_leakage_from_future_results():
    changed = MATCHES.copy()
    changed.loc[4:, "FTR"] = ["A", "A", "A"]  # se cambian los resultados desde el partido 5
    original = elo_rating_diffs(MATCHES)
    modified = elo_rating_diffs(changed)
    # Hasta el partido 5 inclusive, la diferencia previa al partido no puede cambiar.
    assert modified[:5] == pytest.approx(original[:5])
    assert modified[5] != pytest.approx(original[5])


def test_elo_probabilities_are_valid_and_monotonic():
    rng = np.random.default_rng(0)
    diffs = rng.normal(0, 150, size=2000)
    p_home = 1 / (1 + 10 ** (-diffs / 400))
    outcomes = np.where(rng.random(2000) < 0.25, 1, np.where(rng.random(2000) < p_home, 0, 2))
    probs = elo_probabilities(diffs, outcomes, np.array([-300.0, 0.0, 300.0]))
    assert probs.shape == (3, 3)
    assert probs.sum(axis=1) == pytest.approx(np.ones(3))
    assert probs[0, 0] < probs[1, 0] < probs[2, 0]  # más Elo a favor -> más probabilidad de local
    assert probs[0, 2] > probs[1, 2] > probs[2, 2]
