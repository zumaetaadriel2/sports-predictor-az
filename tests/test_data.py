import pandas as pd
import pytest

from sports_predictor.football.data import CLEAN_COLS, load_matches, load_season, season_label

OLD = (
    "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,B365H,B365D,B365A,BbAvH,BbAvD,BbAvA,PSCH,PSCD,PSCA\n"
    "E0,14/08/16,B,C,1,1,D,9,9,2.0,3.0,4.0,2.1,3.1,4.1,2.2,3.2,4.2\n"
    "E0,13/08/16,A,B,2,0,H,10,5,1.5,4.0,6.0,1.6,4.1,6.1,1.7,4.2,6.2\n"
    ",,,,,,,,,,,,,,,,,\n"
)
NEW = (
    "﻿Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,B365H,B365D,B365A,AvgH,AvgD,AvgA,AvgCH,AvgCD,AvgCA\n"
    "E0,09/08/2019,20:00,A,C,0,3,A,4,12,5.0,4.0,1.6,5.1,4.1,1.7,5.2,4.2,1.8\n"
)


@pytest.fixture
def raw_dir(tmp_path):
    (tmp_path / "E0_1617.csv").write_text(OLD, encoding="utf-8")
    (tmp_path / "E0_1920.csv").write_text(NEW, encoding="utf-8")
    return tmp_path


def test_season_label():
    assert season_label("data/raw/E0_1516.csv") == "2015/16"
    assert season_label("E0_2526.csv") == "2025/26"


def test_load_season_parses_both_date_formats_and_sorts(raw_dir):
    old = load_season(raw_dir / "E0_1617.csv")
    assert list(old["Date"]) == [pd.Timestamp("2016-08-13"), pd.Timestamp("2016-08-14")]
    assert len(old) == 2  # la fila vacía se descarta
    new = load_season(raw_dir / "E0_1920.csv")
    assert new["Date"].iloc[0] == pd.Timestamp("2019-08-09")
    assert new["Season"].iloc[0] == "2019/20"


def test_columns_are_stable_across_seasons(raw_dir):
    old = load_season(raw_dir / "E0_1617.csv")
    new = load_season(raw_dir / "E0_1920.csv")
    assert list(old.columns) == list(new.columns) == CLEAN_COLS


def test_market_average_is_unified(raw_dir):
    old = load_season(raw_dir / "E0_1617.csv")
    new = load_season(raw_dir / "E0_1920.csv")
    assert old["AvgH"].iloc[0] == 1.6  # venía como BbAvH
    assert new["AvgH"].iloc[0] == 5.1
    assert old["AvgCH"].isna().all()  # no existía antes de 2019/20
    assert new["PSCH"].isna().all()  # no está en este archivo de prueba


def test_load_matches_is_chronological(raw_dir):
    df = load_matches(raw_dir)
    assert len(df) == 3
    assert df["Date"].is_monotonic_increasing


def test_inconsistent_result_is_rejected(tmp_path):
    bad = OLD.replace("A,B,2,0,H", "A,B,2,0,A")
    (tmp_path / "E0_1617.csv").write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError, match="FTR"):
        load_season(tmp_path / "E0_1617.csv")


def test_missing_result_is_rejected(tmp_path):
    bad = OLD.replace("A,B,2,0,H", "A,B,,,")
    (tmp_path / "E0_1617.csv").write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError, match="sin fecha"):
        load_season(tmp_path / "E0_1617.csv")
