import numpy as np
import pytest

from sports_predictor.betting import flat_bets


def test_flat_bets_by_hand():
    probs = np.array([
        [0.50, 0.30, 0.20],  # valor: 0.0, -0.1, 0.0 -> sin ventaja positiva
        [0.60, 0.25, 0.15],  # valor: +0.2 en local -> apuesta al local, gana
        [0.20, 0.30, 0.50],  # valor: +0.5 en visitante -> apuesta al visitante, pierde
    ])
    odds = np.array([
        [2.0, 3.0, 5.0],
        [2.0, 3.6, 6.0],
        [4.0, 3.0, 3.0],
    ])
    outcomes = [0, 0, 1]
    bet, profit = flat_bets(probs, odds, outcomes)
    assert list(bet) == [False, True, True]
    assert profit == pytest.approx([1.0, -1.0])


def test_threshold_filters_small_edges():
    probs = np.array([[0.55, 0.25, 0.20], [0.70, 0.20, 0.10]])
    odds = np.array([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]])  # valor local: +0.1 y +0.4
    bet, _ = flat_bets(probs, odds, [0, 0], threshold=0.2)
    assert list(bet) == [False, True]


def test_missing_odds_are_ignored():
    probs = np.array([[0.9, 0.05, 0.05]])
    odds = np.array([[np.nan, 3.0, 25.0]])  # sin cuota de local: el mejor valor es el visitante
    bet, profit = flat_bets(probs, odds, [2])
    assert bet[0]
    assert profit == pytest.approx([24.0])


def test_fair_odds_from_true_probabilities_break_even():
    # Si las cuotas son justas respecto de las probabilidades reales, la ganancia esperada es cero.
    rng = np.random.default_rng(0)
    true = rng.dirichlet([3, 2, 3], size=40000)
    outcomes = np.array([rng.choice(3, p=p) for p in true])
    model = rng.dirichlet([3, 2, 3], size=40000)  # un modelo sin información
    bet, profit = flat_bets(model, 1 / true, outcomes)
    assert profit.mean() == pytest.approx(0.0, abs=0.05)
