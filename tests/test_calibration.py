import numpy as np
import pytest

from sports_predictor.calibration import calibration_table


def test_hand_computed_bins():
    probs = np.array([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]])
    table = calibration_table(probs, [0, 1], n_bins=10)
    # Tramos no vacíos: [0.2, 0.3) con los dos 0.2, [0.3, 0.4) con los dos 0.3, [0.5, 0.6) con los dos 0.5.
    assert list(table["bin_low"]) == pytest.approx([0.2, 0.3, 0.5])
    assert list(table["n"]) == [2, 2, 2]
    assert list(table["predicted"]) == pytest.approx([0.2, 0.3, 0.5])
    assert list(table["observed"]) == pytest.approx([0.0, 0.5, 0.5])


def test_probability_one_goes_to_last_bin():
    table = calibration_table(np.array([[1.0, 0.0, 0.0]]), [0], n_bins=10)
    assert list(table["bin_low"]) == pytest.approx([0.0, 0.9])
    assert list(table["n"]) == [2, 1]
    assert list(table["observed"]) == pytest.approx([0.0, 1.0])


def test_well_calibrated_probabilities_stay_near_diagonal():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet([2, 2, 2], size=20000)
    outcomes = np.array([rng.choice(3, p=p) for p in probs])
    table = calibration_table(probs, outcomes)
    big = table[table["n"] >= 500]
    assert np.abs(big["predicted"] - big["observed"]).max() < 0.03
    assert table["n"].sum() == 60000
