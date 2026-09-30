"""NIJ recidivism data, split into the three challenge rounds.

Round r predicts an arrest in year r, among people with no arrest in the
years before r. Round 1 uses what is known at release. Rounds 2 and 3 add the
supervision fields NIJ released for those rounds. Race is kept for the
metrics and is never a feature. The later-year outcomes are never features.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "NIJ_s_Recidivism_Challenge_Full_Dataset.csv"

# Known at release (round 1), in file order, minus Race.
RELEASE = [
    "Gender",
    "Age_at_Release",
    "Residence_PUMA",
    "Gang_Affiliated",
    "Supervision_Risk_Score_First",
    "Supervision_Level_First",
    "Education_Level",
    "Dependents",
    "Prison_Offense",
    "Prison_Years",
    "Prior_Arrest_Episodes_Felony",
    "Prior_Arrest_Episodes_Misd",
    "Prior_Arrest_Episodes_Violent",
    "Prior_Arrest_Episodes_Property",
    "Prior_Arrest_Episodes_Drug",
    "Prior_Arrest_Episodes_PPViolationCharges",
    "Prior_Arrest_Episodes_DVCharges",
    "Prior_Arrest_Episodes_GunCharges",
    "Prior_Conviction_Episodes_Felony",
    "Prior_Conviction_Episodes_Misd",
    "Prior_Conviction_Episodes_Viol",
    "Prior_Conviction_Episodes_Prop",
    "Prior_Conviction_Episodes_Drug",
    "Prior_Conviction_Episodes_PPViolationCharges",
    "Prior_Conviction_Episodes_DomesticViolenceCharges",
    "Prior_Conviction_Episodes_GunCharges",
    "Prior_Revocations_Parole",
    "Prior_Revocations_Probation",
    "Condition_MH_SA",
    "Condition_Cog_Ed",
    "Condition_Other",
]

# Supervision activity fields, released for rounds 2 and 3.
SUPERVISION = [
    "Violations_ElectronicMonitoring",
    "Violations_Instruction",
    "Violations_FailToReport",
    "Violations_MoveWithoutPermission",
    "Delinquency_Reports",
    "Program_Attendances",
    "Program_UnexcusedAbsences",
    "Residence_Changes",
    "Avg_Days_per_DrugTest",
    "DrugTests_THC_Positive",
    "DrugTests_Cocaine_Positive",
    "DrugTests_Meth_Positive",
    "DrugTests_Other_Positive",
    "Percent_Days_Employed",
    "Jobs_Per_Year",
    "Employment_Exempt",
]

OUTCOMES = [
    "Recidivism_Within_3years",
    "Recidivism_Arrest_Year1",
    "Recidivism_Arrest_Year2",
    "Recidivism_Arrest_Year3",
]

# Ordinal codes from Wheeler (2021), variance of leaderboard metrics.
ORDINAL = {
    "Gender": {"M": 1, "F": 0},
    "Age_at_Release": {
        "18-22": 6,
        "23-27": 5,
        "28-32": 4,
        "33-37": 3,
        "38-42": 2,
        "43-47": 1,
        "48 or older": 0,
    },
    "Supervision_Level_First": {"Standard": 0, "High": 1, "Specialized": 2},
    "Education_Level": {
        "Less than HS diploma": 0,
        "High School Diploma": 1,
        "At least some college": 2,
    },
    "Prison_Offense": {
        "Drug": 0,
        "Other": 1,
        "Property": 2,
        "Violent/Non-Sex": 3,
        "Violent/Sex": 4,
    },
    "Prison_Years": {
        "Less than 1 year": 0,
        "1-2 years": 1,
        "Greater than 2 to 3 years": 2,
        "More than 3 years": 3,
    },
}

GROUPS = ["Black men", "White men", "Black women", "White women"]


def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW)


def features_for(round_: int) -> list[str]:
    return RELEASE if round_ == 1 else RELEASE + SUPERVISION


def risk_set(raw: pd.DataFrame, round_: int) -> pd.DataFrame:
    """People still at risk at the start of year `round_`, with y for that year."""
    at_risk = np.ones(len(raw), dtype=bool)
    for prior in range(1, round_):
        at_risk &= ~raw[f"Recidivism_Arrest_Year{prior}"].astype(bool).to_numpy()
    out = raw.loc[at_risk].copy()
    out["y"] = out[f"Recidivism_Arrest_Year{round_}"].astype(int)
    return out.reset_index(drop=True)


def naive_set(raw: pd.DataFrame, round_: int) -> pd.DataFrame:
    """Everyone, with y for year `round_`. Earlier failures count as y = 0."""
    out = raw.copy()
    out["y"] = out[f"Recidivism_Arrest_Year{round_}"].astype(int)
    return out.reset_index(drop=True)


def split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame[frame["Training_Sample"] == 1].reset_index(drop=True)
    test = frame[frame["Training_Sample"] == 0].reset_index(drop=True)
    return train, test


def group_labels(frame: pd.DataFrame) -> np.ndarray:
    race = frame["Race"].map({"BLACK": "Black", "WHITE": "White"})
    sex = frame["Gender"].map({"M": "men", "F": "women"})
    return (race + " " + sex).to_numpy()


# Raw features ----------------------------------------------------------------


def raw_features(frame: pd.DataFrame, round_: int) -> pd.DataFrame:
    """Columns as NIJ ships them. Text and true/false fields become strings.

    Counts such as "10 or more" stay as text categories. Missing stays missing.
    """
    out = frame[features_for(round_)].copy()
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]) or pd.api.types.is_bool_dtype(out[col]):
            vals = out[col].astype(object)
            out[col] = vals.where(vals.isna(), vals.astype(str)).astype("string")
    return out


def raw_is_text(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])]


# Engineered features ---------------------------------------------------------


def _leading_int(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.split(" ", n=1).str[0]
    return pd.to_numeric(text, errors="coerce")


def engineered_features(frame: pd.DataFrame, round_: int) -> pd.DataFrame:
    """Numeric encoding in the spirit of the blog post and Circo and Wheeler.

    Ordered categories get ordered codes, top-coded counts keep the leading
    number, true/false is 0/1, and missing is -1 with a flag for the fields
    where missing is common. Two totals are added for prior arrests and
    convictions. Supervision fields in rounds 2 and 3 get the same treatment,
    plus a flag for people with no drug tests.
    """
    out = pd.DataFrame(index=frame.index)
    for col in features_for(round_):
        ser = frame[col]
        if col in ORDINAL:
            out[col] = ser.map(ORDINAL[col])
        elif pd.api.types.is_bool_dtype(ser):
            out[col] = ser.astype(int)
        elif not pd.api.types.is_numeric_dtype(ser):
            if set(ser.dropna().unique()) <= {True, False, "True", "False"}:
                out[col] = ser.map({True: 1, False: 0, "True": 1, "False": 0})
            else:
                out[col] = _leading_int(ser)
        else:
            out[col] = ser.astype(float)

    for col in ["Gang_Affiliated", "Supervision_Risk_Score_First", "Supervision_Level_First", "Prison_Offense"]:
        out[f"{col}_missing"] = out[col].isna().astype(int)

    out["Prior_Arrests_Total"] = out["Prior_Arrest_Episodes_Felony"] + out["Prior_Arrest_Episodes_Misd"]
    out["Prior_Convictions_Total"] = out["Prior_Conviction_Episodes_Felony"] + out["Prior_Conviction_Episodes_Misd"]

    if round_ > 1:
        out["No_DrugTests"] = out["Avg_Days_per_DrugTest"].isna().astype(int)
        out["Employment_missing"] = out["Percent_Days_Employed"].isna().astype(int)

    return out.fillna(-1).astype(float)
