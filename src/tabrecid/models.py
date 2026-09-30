"""Model wrappers. Each takes train features, train y, test features, and a seed,
and returns the predicted probability of an arrest for the test rows.

The tree models get a light amount of tuning an analyst would do without much
effort: the forest picks its leaf size by out-of-bag Brier score, and the two
boosted models pick the number of trees by early stopping on a 20% slice of the
training sample and are then refit on all of it. The foundation models are
used as they ship.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

THREADS = int(os.environ.get("TABRECID_THREADS", "3"))
LEAF_GRID = [1, 5, 10, 25, 50]


def _text_cols(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])]


def _obj(frame: pd.DataFrame) -> pd.DataFrame:
    """Text columns as plain objects, with np.nan for missing."""
    out = frame.astype(object)
    return out.where(frame.notna(), np.nan)


def _ordinal(xtr, xte):
    """Integer codes for text columns (alphabetical order), missing left as NaN."""
    text = _text_cols(xtr)
    if not text:
        return xtr, xte, text
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                         encoded_missing_value=np.nan)
    cols = list(xtr.columns)
    codes_tr = pd.DataFrame(enc.fit_transform(_obj(xtr[text])), columns=text, index=xtr.index)
    codes_te = pd.DataFrame(enc.transform(_obj(xte[text])), columns=text, index=xte.index)
    xtr = pd.concat([xtr.drop(columns=text), codes_tr], axis=1)[cols].astype(float)
    xte = pd.concat([xte.drop(columns=text), codes_te], axis=1)[cols].astype(float)
    return xtr, xte, text


# Random forest -----------------------------------------------------------------


def random_forest(xtr, ytr, xte, seed):
    xtr, xte, _ = _ordinal(xtr, xte)
    best, best_score = None, np.inf
    for leaf in LEAF_GRID:
        rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=leaf, oob_score=True,
                                    n_jobs=THREADS, random_state=seed)
        rf.fit(xtr, ytr)
        oob = rf.oob_decision_function_[:, 1]
        ok = ~np.isnan(oob)
        score = np.mean((oob[ok] - np.asarray(ytr)[ok]) ** 2)
        if score < best_score:
            best, best_score = rf, score
    return best.predict_proba(xte)[:, 1]


# Logistic regression -------------------------------------------------------------


def logit(xtr, ytr, xte, seed):
    """L2 logistic regression, penalty chosen by 5-fold CV Brier score.

    Text columns are dummy coded with missing as its own level, numeric columns
    get a missing indicator and zero fill, and everything is standardized.
    """
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    text = _text_cols(xtr)
    both = pd.concat([xtr, xte], ignore_index=True)
    numeric = [c for c in both.columns if c not in text]
    parts = [both[numeric].astype(float)]
    miss = both[numeric].isna()
    miss = miss.loc[:, miss.any()]
    if len(miss.columns):
        parts.append(miss.astype(float).add_suffix("_missing"))
    parts[0] = parts[0].fillna(0.0)
    if text:
        parts.append(pd.get_dummies(_obj(both[text]).fillna("NA").astype(str), drop_first=True, dtype=float))
    design = pd.concat(parts, axis=1).to_numpy()
    n = len(xtr)
    scaler = StandardScaler().fit(design[:n])
    ztr, zte = scaler.transform(design[:n]), scaler.transform(design[n:])
    model = LogisticRegressionCV(
        Cs=np.logspace(-4, 2, 13), cv=StratifiedKFold(5, shuffle=True, random_state=seed),
        scoring="neg_brier_score", max_iter=5000, n_jobs=1,
    )
    model.fit(ztr, np.asarray(ytr))
    return model.predict_proba(zte)[:, 1]


# LightGBM ----------------------------------------------------------------------


def _as_category(xtr, xte):
    text = _text_cols(xtr)
    xtr = xtr.copy()
    xte = xte.copy()
    for c in text:
        levels = sorted(set(xtr[c].dropna()) | set(xte[c].dropna()))
        xtr[c] = pd.Categorical(xtr[c].astype(object), categories=levels)
        xte[c] = pd.Categorical(xte[c].astype(object), categories=levels)
    return xtr, xte


def lightgbm(xtr, ytr, xte, seed):
    import lightgbm as lgb

    xtr, xte = _as_category(xtr, xte)
    ytr = np.asarray(ytr)
    params = dict(learning_rate=0.02, n_estimators=3000, random_state=seed,
                  n_jobs=THREADS, verbose=-1)
    fit_x, val_x, fit_y, val_y = train_test_split(xtr, ytr, test_size=0.2, stratify=ytr,
                                                  random_state=seed)
    probe = lgb.LGBMClassifier(**params)
    probe.fit(fit_x, fit_y, eval_set=[(val_x, val_y)], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    params["n_estimators"] = max(int(probe.best_iteration_ or 1), 1)
    final = lgb.LGBMClassifier(**params)
    final.fit(xtr, ytr)
    return final.predict_proba(xte)[:, 1]


# CatBoost ----------------------------------------------------------------------


def catboost(xtr, ytr, xte, seed):
    from catboost import CatBoostClassifier

    text = _text_cols(xtr)
    xtr = xtr.copy()
    xte = xte.copy()
    for c in text:
        xtr[c] = xtr[c].astype(object).fillna("NA").astype(str)
        xte[c] = xte[c].astype(object).fillna("NA").astype(str)
    ytr = np.asarray(ytr)
    params = dict(learning_rate=0.03, iterations=3000, random_seed=seed,
                  thread_count=THREADS, verbose=False, allow_writing_files=False)
    fit_x, val_x, fit_y, val_y = train_test_split(xtr, ytr, test_size=0.2, stratify=ytr,
                                                  random_state=seed)
    probe = CatBoostClassifier(**params, od_type="Iter", od_wait=100)
    probe.fit(fit_x, fit_y, cat_features=text, eval_set=(val_x, val_y), use_best_model=True)
    params["iterations"] = max(int(probe.get_best_iteration()) + 1, 1)
    final = CatBoostClassifier(**params)
    final.fit(xtr, ytr, cat_features=text)
    return final.predict_proba(xte)[:, 1]


# Tabular foundation models (NVIDIA structured-data-models) ---------------------

_SDM_CACHE: dict = {}


def _sdm_model(name: str, device: str):
    import sdm

    key = (name, device)
    if key not in _SDM_CACHE:
        if name.startswith("kumo_"):
            model = sdm.models.KumoTabular(task="classification", size=name.split("_")[1], device=device)
        elif name == "tabicl":
            model = sdm.models.TabICLv2(task="classification", device=device)
        else:
            raise ValueError(name)
        _SDM_CACHE[key] = model
    return _SDM_CACHE[key]


def sdm_predict(name, xtr, ytr, xte, seed, device=None, estimators=8):
    import sdm
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    text = _text_cols(xtr)
    both = pd.concat([xtr, xte], ignore_index=True)
    both["y"] = pd.array(list(np.asarray(ytr).astype(int).astype(str)) + [None] * len(xte), dtype="string")
    stypes = {c: ("categorical" if c in text else "numerical") for c in xtr.columns}
    stypes["y"] = "categorical"
    table = sdm.TableTensor.from_pandas(both, stypes, device=device)
    n = len(xtr)
    model = _sdm_model(name, device)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    with torch.no_grad():
        out = model(
            x_context=table[:n].drop_columns("y"),
            y_context=table[:n, "y"],
            x_query=table[n:].drop_columns("y"),
            num_estimators=estimators,
            generator=gen,
        )
    labels = [str(c) for c in out.columns[sdm.Stype.numerical]]
    probs = out.numerical.float().cpu().numpy()
    result = probs[:, labels.index("1")]
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


# TabPFN v2 -----------------------------------------------------------------------


def tabpfn(xtr, ytr, xte, seed):
    import torch
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    cols = list(xtr.columns)
    xtr, xte, text = _ordinal(xtr, xte)
    model = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2,
        device="cuda" if torch.cuda.is_available() else "cpu",
        ignore_pretraining_limits=True,
        categorical_features_indices=[cols.index(c) for c in text] or None,
        random_state=int(seed),
    )
    model.fit(xtr.to_numpy(), np.asarray(ytr))
    classes = list(model.classes_)
    return model.predict_proba(xte.to_numpy())[:, classes.index(1)]


MODELS = {
    "tabpfn": tabpfn,
    "logit": logit,
    "rf": random_forest,
    "lgbm": lightgbm,
    "catboost": catboost,
    "kumo_small": lambda a, b, c, s: sdm_predict("kumo_small", a, b, c, s),
    "kumo_medium": lambda a, b, c, s: sdm_predict("kumo_medium", a, b, c, s),
    "tabicl": lambda a, b, c, s: sdm_predict("tabicl", a, b, c, s),
}

LABELS = {
    "logit": "Logistic regression",
    "rf": "Random forest",
    "lgbm": "LightGBM",
    "catboost": "CatBoost",
    "kumo_small": "Kumo Tabular",
    "kumo_medium": "Kumo Tabular (medium)",
    "tabicl": "TabICLv2",
    "tabpfn": "TabPFN v2",
}
