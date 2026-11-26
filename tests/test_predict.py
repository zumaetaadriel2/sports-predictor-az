import numpy as np
import pandas as pd
import pytest

from sports_predictor.football.poisson import walk_forward_expected_goals
from sports_predictor.football.predict import (
    PREDICTION_COLS,
    append_predictions,
    fit_goal_model,
    load_fixtures,
    predict_fixtures,
    read_history,
    team_strengths,
)


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


def fixtures_for(matches, rows):
    f = matches.iloc[rows][["Date", "HomeTeam", "AwayTeam"]].reset_index(drop=True)
    f["AvgH"], f["AvgD"], f["AvgA"] = 2.0, 3.5, 4.0
    f["AvgOver25"], f["AvgUnder25"] = 1.8, 2.0
    return f


def test_matches_the_evaluated_model():
    # Predecir la última fecha "en vivo" da lo mismo que el recorrido usado en la evaluación.
    matches = make_matches()
    rate_h, rate_a, rho = walk_forward_expected_goals(matches, 8, half_life_days=365.0)
    last = [22, 23]
    model = fit_goal_model(matches, matches["Date"].iloc[22])
    pred = predict_fixtures(model, fixtures_for(matches, last), "2020-03-17")
    assert pred["exp_goals_home"].to_numpy() == pytest.approx(rate_h[last], abs=1e-3)
    assert pred["exp_goals_away"].to_numpy() == pytest.approx(rate_a[last], abs=1e-3)
    assert model["rho"] == pytest.approx(rho[22], abs=1e-3)


def test_model_ignores_matches_on_or_after_the_date():
    matches = make_matches()
    changed = matches.copy()
    changed.loc[20:, ["FTHG", "FTAG"]] = 9
    as_of = matches["Date"].iloc[20]
    a, b = fit_goal_model(matches, as_of), fit_goal_model(changed, as_of)
    assert a["params"] == pytest.approx(b["params"])
    assert a["rho"] == pytest.approx(b["rho"])


def test_prediction_columns_and_probabilities():
    matches = make_matches()
    model = fit_goal_model(matches, "2020-06-01")
    pred = predict_fixtures(model, fixtures_for(matches, [0, 1]), "2020-06-01")
    assert list(pred.columns) == PREDICTION_COLS
    assert (pred[["p_home", "p_draw", "p_away"]].sum(axis=1) - 1).abs().max() < 1e-3
    assert (pred[["market_home", "market_draw", "market_away"]].sum(axis=1) - 1).abs().max() < 1e-3
    assert pred["market_over25"].iloc[0] == pytest.approx((1 / 1.8) / (1 / 1.8 + 1 / 2.0), abs=1e-4)


def test_unknown_team_is_treated_as_average():
    matches = make_matches()
    model = fit_goal_model(matches, "2020-06-01")
    fixtures = fixtures_for(matches, [0])
    fixtures.loc[0, ["HomeTeam", "AwayTeam"]] = ["Nuevo", "Otro nuevo"]
    pred = predict_fixtures(model, fixtures, "2020-06-01")
    base, home_adv = model["params"][0], model["params"][1]
    assert pred["exp_goals_home"].iloc[0] == pytest.approx(np.exp(base + home_adv), abs=1e-3)
    assert pred["exp_goals_away"].iloc[0] == pytest.approx(np.exp(base), abs=1e-3)
    strengths = team_strengths(model, ["A", "Nuevo"])
    assert set(strengths["team"]) == {"A", "Nuevo"}


def test_history_never_rewrites_published_predictions(tmp_path):
    matches = make_matches()
    model = fit_goal_model(matches, "2020-06-01")
    first = predict_fixtures(model, fixtures_for(matches, [0, 1]), "2020-06-01")
    history = append_predictions(read_history(tmp_path / "no_existe.csv"), first)
    assert len(history) == 2

    later = predict_fixtures(model, fixtures_for(matches, [1, 2]), "2020-06-08")
    later["p_home"] = 0.99  # aunque el modelo cambie, lo ya publicado no se toca
    history = append_predictions(history, later)
    assert len(history) == 3
    assert history["p_home"].iloc[1] == first["p_home"].iloc[1]
    assert history["published_at"].iloc[1] == pd.Timestamp("2020-06-01")

    path = tmp_path / "predictions.csv"
    history.to_csv(path, index=False, date_format="%Y-%m-%d")
    pd.testing.assert_frame_equal(read_history(path), history, check_dtype=False)


def test_load_fixtures_filters_league_and_parses_dates(tmp_path):
    text = (
        "﻿Div,Date,Time,HomeTeam,AwayTeam,AvgH,AvgD,AvgA,Avg>2.5,Avg<2.5\n"
        "E1,06/11/2026,20:00,X,Y,2,3,4,1.9,1.9\n"
        "E0,07/11/2026,15:00,A,B,1.5,4,6,1.7,2.1\n"
    )
    path = tmp_path / "fixtures.csv"
    path.write_text(text, encoding="utf-8")
    fixtures = load_fixtures(path)
    assert len(fixtures) == 1
    assert fixtures["Date"].iloc[0] == pd.Timestamp("2026-11-07")
    assert fixtures["AvgOver25"].iloc[0] == 1.7
