"""Load the saved results and summarize them for the paper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def samples() -> pd.DataFrame:
    files = sorted((RESULTS / "samples").glob("*.csv"))
    frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return frame.drop_duplicates(["model", "features", "round", "size", "rep", "group"])


def summary(frame: pd.DataFrame, by=("model", "features", "round", "size", "full", "group")) -> pd.DataFrame:
    """Mean, standard deviation, min, and max over the replications."""
    g = frame.groupby(list(by))
    out = g.agg(
        reps=("rep", "nunique"),
        brier=("brier", "mean"),
        brier_sd=("brier", "std"),
        brier_min=("brier", "min"),
        brier_max=("brier", "max"),
        auc=("auc", "mean"),
        auc_sd=("auc", "std"),
        auc_min=("auc", "min"),
        auc_max=("auc", "max"),
        fpr50=("fpr50", "mean"),
        flag50=("flag50", "mean"),
        base_rate=("base_rate", "mean"),
        mean_pred=("mean_pred", "mean"),
        n=("n", "mean"),
    )
    return out.reset_index()


def wins(frame: pd.DataFrame, a: str, b: str, metric: str = "brier", group: str = "All",
         features: str = "raw") -> pd.DataFrame:
    """Share of paired replications in which model a beats model b, by round and size."""
    sub = frame[(frame["group"] == group) & (frame["features"] == features)]
    wide = sub.pivot_table(index=["round", "size", "rep"], columns="model", values=metric)
    wide = wide.dropna(subset=[a, b])
    better = wide[a] < wide[b] if metric == "brier" else wide[a] > wide[b]
    return better.groupby(level=["round", "size"]).mean().rename("share").reset_index()


def leaderboard() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "nij_leaderboard.csv")


def conditional() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "conditional.csv")


def preds(round_: int, features: str, model: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / "preds" / f"r{round_}_{features}_{model}.csv.gz")


def fairness(pred: pd.DataFrame, cut: float = 0.5) -> pd.DataFrame:
    """NIJ false positive rate, C / (C + D), by race within sex and overall."""
    rows = []
    for sex, sex_lab in [("M", "Men"), ("F", "Women"), (None, "All")]:
        part = pred if sex is None else pred[pred["Gender"] == sex]
        row = {"sex": sex_lab}
        for race, lab in [("BLACK", "Black"), ("WHITE", "White")]:
            r = part[(part["Race"] == race) & (part["y"] == 0)]
            row[f"fpr_{lab}"] = float((r["p"] >= cut).mean())
            row[f"n_neg_{lab}"] = len(r)
        row["diff"] = row["fpr_Black"] - row["fpr_White"]
        row["brier"] = float(np.mean((part["p"] - part["y"]) ** 2))
        row["nij_score"] = (1 - row["brier"]) * (1 - abs(row["diff"]))
        rows.append(row)
    return pd.DataFrame(rows)
