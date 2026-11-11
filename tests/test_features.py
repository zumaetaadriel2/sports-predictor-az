import numpy as np
import pandas as pd
import pytest

from sports_predictor.football.features import FEATURE_COLS, form_features


def make_matches():
    rows = [
        ("2020-01-01", "A", "B", 2, 0, "H", 5, 1),
        ("2020-01-04", "B", "A", 1, 1, "D", 3, 4),
        ("2020-01-10", "A", "C", 0, 1, "A", 2, 6),
        ("2020-02-20", "C", "A", 3, 3, "D", 7, 7),
    ]
    df = pd.DataFrame(rows, columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HST", "AST"])
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def test_columns_and_first_match_is_nan():
    f = form_features(make_matches())
    assert list(f.columns) == FEATURE_COLS
    assert f.iloc[0].isna().all()  # nadie tiene partidos anteriores


def test_values_by_hand():
    f = form_features(make_matches())
    # Partido 3 (A recibe a C): A lleva una victoria de local y un empate de visitante.
    assert f.loc[2, "home_form_points"] == pytest.approx(2.0)  # (3 + 1) / 2
    assert f.loc[2, "home_form_goals_for"] == pytest.approx(1.5)  # (2 + 1) / 2
    assert f.loc[2, "home_form_goals_against"] == pytest.approx(0.5)
    assert f.loc[2, "home_form_shots_for"] == pytest.approx(4.5)  # (5 + 4) / 2
    assert f.loc[2, "home_venue_points"] == pytest.approx(3.0)  # solo su partido de local
    assert f.loc[2, "home_rest_days"] == 6
    assert np.isnan(f.loc[2, "away_form_points"])  # C debuta
    # Partido 4: A descansa 41 días, se recorta a 14. C viene de ganar de visitante y no tiene partidos de local.
    assert f.loc[3, "away_rest_days"] == 14
    assert f.loc[3, "home_form_points"] == pytest.approx(3.0)
    assert np.isnan(f.loc[3, "home_venue_points"])
    assert f.loc[3, "away_venue_points"] == pytest.approx(1.0)  # A de visitante: un empate


def test_no_leakage_from_current_or_future_results():
    matches = make_matches()
    changed = matches.copy()
    changed.loc[2:, ["FTHG", "FTAG", "HST", "AST"]] = 9
    changed.loc[2:, "FTR"] = "H"
    original, modified = form_features(matches), form_features(changed)
    # Las features de los partidos 1 a 3 no dependen del resultado del partido 3 ni de los posteriores.
    pd.testing.assert_frame_equal(original.iloc[:3], modified.iloc[:3])
    assert not original.iloc[3].equals(modified.iloc[3])
