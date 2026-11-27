import numpy as np
import pytest

from sports_predictor.ensemble import (
    blend,
    fit_blend_weights,
    fit_temperature,
    simplex_grid,
    temperature_scale,
)


def test_simplex_grid():
    grid = list(simplex_grid(3, step=0.1))
    assert len(grid) == 66  # combinaciones de 10 unidades en 3 partes
    assert all(sum(w) == pytest.approx(1.0) and min(w) >= 0 for w in grid)
    assert (1.0, 0.0, 0.0) in grid and (0.0, 0.0, 1.0) in grid


def test_blend_by_hand():
    a = np.array([[0.6, 0.3, 0.1]])
    b = np.array([[0.2, 0.3, 0.5]])
    assert blend([a, b], (0.5, 0.5))[0] == pytest.approx([0.4, 0.3, 0.3])
    assert blend([a, b], (1.0, 0.0))[0] == pytest.approx(a[0])


def simulate(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    true = rng.dirichlet([3, 2, 3], size=n)
    outcomes = np.array([rng.choice(3, p=p) for p in true])
    return true, outcomes, rng


def test_fit_blend_weights_prefers_the_informative_model():
    true, outcomes, rng = simulate()
    noise = rng.dirichlet([3, 2, 3], size=len(true))  # sin relación con el resultado
    weights = fit_blend_weights([noise, true], outcomes)
    assert weights[1] >= 0.8


def test_temperature_scale_properties():
    p = np.array([[0.7, 0.2, 0.1]])
    assert temperature_scale(p, 1.0) == pytest.approx(p)
    assert temperature_scale(p, 1000.0)[0] == pytest.approx([1 / 3] * 3, abs=1e-2)
    assert temperature_scale(p, 0.5)[0, 0] > 0.7  # T < 1 afila
    assert temperature_scale(p, 2.0)[0, 0] < 0.7  # T > 1 suaviza
    assert temperature_scale(p, 2.0).sum() == pytest.approx(1.0)


def test_fit_temperature_recovers_overconfidence():
    true, outcomes, _ = simulate()
    overconfident = temperature_scale(true, 0.6)  # el modelo exagera su confianza
    t = fit_temperature(overconfident, outcomes)
    assert t == pytest.approx(1 / 0.6, abs=0.2)  # la temperatura deshace la exageración
