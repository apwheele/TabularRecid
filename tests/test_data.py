import numpy as np
import pandas as pd
import pytest

from tabrecid import data, metrics


@pytest.fixture(scope="module")
def raw():
    return data.load_raw()


def test_outcomes_exclusive(raw):
    # Each person has at most one first-arrest year, which is what makes the
    # later rounds a risk set.
    years = raw[["Recidivism_Arrest_Year1", "Recidivism_Arrest_Year2", "Recidivism_Arrest_Year3"]].astype(int)
    assert years.sum(axis=1).max() == 1


@pytest.mark.parametrize("round_", [1, 2, 3])
def test_no_outcomes_or_race_in_features(raw, round_):
    cols = set(data.features_for(round_))
    assert not cols & set(data.OUTCOMES)
    assert "Race" not in cols and "Training_Sample" not in cols and "ID" not in cols
    assert set(data.engineered_features(raw.head(50), round_).columns).isdisjoint(data.OUTCOMES)


def test_round1_has_no_supervision(raw):
    assert not set(data.features_for(1)) & set(data.SUPERVISION)
    assert set(data.SUPERVISION) <= set(data.features_for(2))


def test_risk_set(raw):
    r2 = data.risk_set(raw, 2)
    r3 = data.risk_set(raw, 3)
    assert not r2["Recidivism_Arrest_Year1"].any()
    assert not (r3["Recidivism_Arrest_Year1"] | r3["Recidivism_Arrest_Year2"]).any()
    assert len(r2) == (~raw["Recidivism_Arrest_Year1"]).sum()
    assert (r3["y"] == r3["Recidivism_Arrest_Year3"].astype(int)).all()


@pytest.mark.parametrize("round_", [1, 2, 3])
def test_engineered_numeric(raw, round_):
    fe = data.engineered_features(raw, round_)
    assert fe.notna().all().all()
    assert all(pd.api.types.is_float_dtype(t) for t in fe.dtypes)
    assert fe.loc[raw["Prior_Arrest_Episodes_Felony"] == "10 or more", "Prior_Arrest_Episodes_Felony"].eq(10).all()


def test_raw_keeps_text(raw):
    rf = data.raw_features(raw, 1)
    assert "10 or more" in set(rf["Prior_Arrest_Episodes_Felony"].dropna())
    assert rf["Gang_Affiliated"].isna().sum() == raw["Gang_Affiliated"].isna().sum()


def test_nij_fpr():
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0.6, 0.2, 0.7, 0.1, 0.9, 0.3])
    assert metrics.nij_fpr(y, p) == 0.5
    assert metrics.brier([0, 1], [0.0, 1.0]) == 0.0
