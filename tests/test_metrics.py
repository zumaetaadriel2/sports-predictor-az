import numpy as np
import pytest
from sklearn.metrics import log_loss as sklearn_log_loss

from sports_predictor.metrics import brier_score, log_loss, rps
from sports_predictor.odds import implied_probabilities
from sports_predictor.validation import expanding_season_splits


def test_perfect_prediction_scores_zero():
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    outcomes = [0, 2]
    assert log_loss(probs, outcomes) == pytest.approx(0.0, abs=1e-12)
    assert brier_score(probs, outcomes) == 0.0
    assert rps(probs, outcomes) == 0.0


def test_uniform_prediction_known_values():
    probs = np.full((1, 3), 1 / 3)
    assert log_loss(probs, [0]) == pytest.approx(np.log(3))
    assert brier_score(probs, [0]) == pytest.approx(2 / 3)  # (2/3)^2 + (1/3)^2 + (1/3)^2
    assert rps(probs, [0]) == pytest.approx(5 / 18)  # ((1/3-1)^2 + (2/3-1)^2) / 2
    assert rps(probs, [1]) == pytest.approx(1 / 9)  # ((1/3)^2 + (2/3-1)^2) / 2


def test_hand_computed_example():
    probs = np.array([[0.5, 0.3, 0.2]])
    assert log_loss(probs, [1]) == pytest.approx(-np.log(0.3))
    assert brier_score(probs, [1]) == pytest.approx(0.25 + 0.49 + 0.04)
    assert rps(probs, [1]) == pytest.approx((0.5**2 + 0.2**2) / 2)


def test_log_loss_matches_sklearn():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet([1, 1, 1], size=50)
    outcomes = rng.integers(0, 3, size=50)
    assert log_loss(probs, outcomes) == pytest.approx(sklearn_log_loss(outcomes, probs, labels=[0, 1, 2]))


def test_rps_penalizes_distance_between_classes():
    # Misma probabilidad para lo que ocurrió (visitante), pero el resto está "más lejos" en el primer caso.
    far = np.array([[0.6, 0.0, 0.4]])
    near = np.array([[0.0, 0.6, 0.4]])
    assert rps(far, [2]) > rps(near, [2])
    assert log_loss(far, [2]) == pytest.approx(log_loss(near, [2]))


def test_implied_probabilities_remove_margin():
    probs = implied_probabilities([[2.0, 3.0, 4.0]])
    assert probs.sum() == pytest.approx(1.0)
    assert probs[0] == pytest.approx(np.array([6, 4, 3]) / 13)


def test_implied_probabilities_fair_odds_unchanged():
    assert implied_probabilities([[2.0, 4.0, 4.0]])[0] == pytest.approx([0.5, 0.25, 0.25])


def test_implied_probabilities_missing_odds_give_nan():
    assert np.isnan(implied_probabilities([[2.0, np.nan, 4.0]])).all()


def test_expanding_splits_train_on_past_only():
    splits = list(expanding_season_splits(["a", "b", "c", "d", "e"], min_train_seasons=3))
    assert splits == [(["a", "b", "c"], "d"), (["a", "b", "c", "d"], "e")]
