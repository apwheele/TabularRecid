"""Brier score, AUC, and the NIJ false positive rate, overall and by group."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from tabrecid.data import GROUPS


def brier(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def auc(y, p) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def nij_fpr(y, p, cut: float = 0.5) -> float:
    """False positives over all non-recidivists, C / (C + D) in the NIJ rubric."""
    y = np.asarray(y)
    flag = np.asarray(p) >= cut
    neg = y == 0
    return float(flag[neg].mean()) if neg.any() else float("nan")


def subsets(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    male = (frame["Gender"] == "M").to_numpy()
    black = (frame["Race"] == "BLACK").to_numpy()
    out = {
        "All": np.ones(len(frame), dtype=bool),
        "Men": male,
        "Women": ~male,
        "Black": black,
        "White": ~black,
    }
    for g in GROUPS:
        race, sex = g.split()
        out[g] = (black if race == "Black" else ~black) & (male if sex == "men" else ~male)
    return out


def score_table(frame: pd.DataFrame, y, p) -> pd.DataFrame:
    """One row per subset with n, base rate, mean prediction, Brier, AUC, FPR at 0.5."""
    y = np.asarray(y)
    p = np.asarray(p)
    rows = []
    for name, mask in subsets(frame).items():
        rows.append({
            "group": name,
            "n": int(mask.sum()),
            "base_rate": float(y[mask].mean()),
            "mean_pred": float(p[mask].mean()),
            "brier": brier(y[mask], p[mask]),
            "auc": auc(y[mask], p[mask]),
            "fpr50": nij_fpr(y[mask], p[mask]),
            "flag50": float((p[mask] >= 0.5).mean()),
        })
    return pd.DataFrame(rows)
