import numpy as np
import pandas as pd
import pytest

from sports_predictor.football.poisson import (
    expected_goals,
    fit_poisson,
    fit_rho,
    outcome_probabilities,
    over_probability,
    score_matrices,
    walk_forward_expected_goals,
)


def simulate(n_teams=6, rounds=40, seed=0):
    """Liga simulada todos contra todos con fuerzas conocidas."""
    rng = np.random.default_rng(seed)
    attack = np.linspace(-0.4, 0.4, n_teams)
    defense = np.linspace(0.3, -0.3, n_teams)
    home, away = [], []
    for _ in range(rounds):
        for h in range(n_teams):
            for a in range(n_teams):
                if h != a:
                    home.append(h)
                    away.append(a)
    home, away = np.array(home), np.array(away)
    rate_h = np.exp(0.1 + 0.3 + attack[home] - defense[away])
    rate_a = np.exp(0.1 + attack[away] - defense[home])
    return home, away, rng.poisson(rate_h), rng.poisson(rate_a), rate_h, rate_a


def test_fit_recovers_known_rates():
    home, away, hg, ag, rate_h, rate_a = simulate(rounds=200)
    params = fit_poisson(home, away, hg, ag, n_teams=6, penalty=0.01)
    fit_h, fit_a = expected_goals(params, home, away, 6)
    assert params[1] == pytest.approx(0.3, abs=0.06)  # ventaja de local
    assert np.abs(fit_h / rate_h - 1).max() < 0.15
    assert np.abs(fit_a / rate_a - 1).max() < 0.15


def test_team_without_matches_is_average():
    home, away, hg, ag, *_ = simulate()
    params = fit_poisson(home, away, hg, ag, n_teams=7)  # el equipo 6 no juega
    assert params[2 + 6] == pytest.approx(0.0, abs=1e-6)  # ataque
    assert params[2 + 7 + 6] == pytest.approx(0.0, abs=1e-6)  # defensa


def test_score_matrix_by_hand():
    m = score_matrices([1.5], [1.0])
    assert m.shape == (1, 11, 11)
    assert m.sum() == pytest.approx(1.0)
    # P(2-1) = Poisson(2; 1.5) * Poisson(1; 1.0)
    expected = (np.exp(-1.5) * 1.5**2 / 2) * (np.exp(-1.0) * 1.0)
    assert m[0, 2, 1] == pytest.approx(expected, rel=1e-6)


def test_outcome_and_over_probabilities():
    m = score_matrices([1.5, 1.0], [1.0, 1.0])
    probs = outcome_probabilities(m)
    assert probs.sum(axis=1) == pytest.approx([1.0, 1.0])
    assert probs[0, 0] > probs[0, 2]  # el que espera más goles gana más seguido
    assert probs[1, 0] == pytest.approx(probs[1, 2])  # tasas iguales: simétrico
    # Total de goles ~ Poisson(2.5): P(más de 2.5) = 1 - P(0) - P(1) - P(2)
    lam = 2.5
    expected = 1 - np.exp(-lam) * (1 + lam + lam**2 / 2)
    assert over_probability(m)[0] == pytest.approx(expected, abs=1e-6)


def make_matches():
    rows = []
    teams = ["A", "B", "C", "D"]
    rng = np.random.default_rng(3)
    day = pd.Timestamp("2020-01-01")
    for r in range(12):
        order = rng.permutation(teams)
        for h, a in [(order[0], order[1]), (order[2], order[3])]:
            rows.append((day + pd.Timedelta(days=7 * r), h, a, int(rng.poisson(1.5)), int(rng.poisson(1.1))))
    return pd.DataFrame(rows, columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])


def test_walk_forward_has_no_leakage():
    matches = make_matches()
    first = 8
    rate_h, rate_a, rho = walk_forward_expected_goals(matches, first)
    assert not np.isnan(rho[first:]).any()
    assert np.isnan(rate_h[:first]).all()
    assert not np.isnan(rate_h[first:]).any()

    changed = matches.copy()
    cut = 16  # primer partido de una fecha
    changed.loc[cut:, ["FTHG", "FTAG"]] = 9
    new_h, new_a, _ = walk_forward_expected_goals(changed, first, half_life_days=30)
    rate_h, rate_a, _ = walk_forward_expected_goals(matches, first, half_life_days=30)
    # Las predicciones hasta la fecha del cambio (inclusive) no pueden cambiar.
    assert new_h[first:cut + 2] == pytest.approx(rate_h[first:cut + 2])
    assert new_a[first:cut + 2] == pytest.approx(rate_a[first:cut + 2])
    assert not new_h[cut + 2:] == pytest.approx(rate_h[cut + 2:])


def test_walk_forward_matches_direct_fit():
    matches = make_matches()
    rate_h, _, _ = walk_forward_expected_goals(matches, 8, penalty=5.0)
    teams = sorted(set(matches["HomeTeam"]) | set(matches["AwayTeam"]))
    idx = {t: i for i, t in enumerate(teams)}
    past = matches.iloc[:22]  # todo lo anterior a la última fecha
    params = fit_poisson(past["HomeTeam"].map(idx), past["AwayTeam"].map(idx), past["FTHG"], past["FTAG"], 4, 5.0)
    last = matches.iloc[22:]
    direct_h, _ = expected_goals(params, last["HomeTeam"].map(idx), last["AwayTeam"].map(idx), 4)
    assert rate_h[22:] == pytest.approx(direct_h, rel=1e-5)


def test_weights_equal_to_repeating_matches():
    home, away, hg, ag, *_ = simulate(rounds=2)
    weights = np.where(np.arange(len(home)) % 2 == 0, 2.0, 1.0)
    twice = weights == 2.0
    repeated = [np.concatenate([x, x[twice]]) for x in (home, away, hg, ag)]
    weighted = fit_poisson(home, away, hg, ag, n_teams=6, weights=weights)
    assert weighted == pytest.approx(fit_poisson(*repeated, n_teams=6), abs=1e-4)


def test_half_life_downweights_old_matches():
    matches = make_matches()
    plain, _, _ = walk_forward_expected_goals(matches, 8)
    weighted, _, _ = walk_forward_expected_goals(matches, 8, half_life_days=14)
    assert not weighted[8:] == pytest.approx(plain[8:])


def test_dixon_coles_matrix_by_hand():
    lam, mu, rho = 1.4, 1.1, -0.1
    plain = score_matrices([lam], [mu])[0]
    dc = score_matrices([lam], [mu], rho=[rho])[0]
    assert dc.sum() == pytest.approx(1.0)
    # La corrección mueve probabilidad entre los cuatro marcadores bajos y casi no cambia el total.
    assert dc[0, 0] / plain[0, 0] == pytest.approx(1 - lam * mu * rho, rel=1e-3)
    assert dc[1, 1] / plain[1, 1] == pytest.approx(1 - rho, rel=1e-3)
    assert dc[0, 1] / plain[0, 1] == pytest.approx(1 + lam * rho, rel=1e-3)
    assert dc[3, 2] == pytest.approx(plain[3, 2], rel=1e-3)
    # rho negativo: más empates que el Poisson independiente.
    assert outcome_probabilities(dc[None])[0, 1] > outcome_probabilities(plain[None])[0, 1]


def test_fit_rho_recovers_known_value():
    rng = np.random.default_rng(5)
    lam, mu, rho = 1.4, 1.1, -0.12
    matrix = score_matrices([lam], [mu], rho=[rho])[0]
    draws = rng.choice(matrix.size, size=40000, p=matrix.ravel())
    hg, ag = np.unravel_index(draws, matrix.shape)
    estimate = fit_rho(np.full(40000, lam), np.full(40000, mu), hg, ag)
    assert estimate == pytest.approx(rho, abs=0.03)
