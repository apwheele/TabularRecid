"""Fit each model on random training subsamples and score it on the NIJ test sample.

For each round, sample size, and replication, the same subsample of the NIJ
training data is given to every model. The NIJ test sample (restricted to the
round's risk set) is never subsampled. Results are appended to
results/samples/{features}_{model}.csv, one row per subgroup, so the script can
be stopped and restarted. Predictions for the first replication at the full
training size go to results/preds/.

Examples:
    uv run python scripts/02_samples.py --models rf,lgbm,catboost
    uv run python scripts/02_samples.py --models kumo_small --features fe --sizes full
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import psutil

from tabrecid import data, metrics
from tabrecid.models import MODELS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "samples"
PREDS = ROOT / "results" / "preds"
SIZES = [500, 1000, 2000, 4000, 8000, 16000]
REPS = 10


def sizes_for(n_train: int, which: str) -> list[int]:
    if which == "full":
        return [n_train]
    return [s for s in SIZES if s < n_train] + [n_train]


def done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    old = pd.read_csv(path, usecols=["round", "size", "rep"])
    return set(map(tuple, old.drop_duplicates().to_numpy().tolist()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="rf,lgbm,catboost,kumo_small")
    ap.add_argument("--features", default="raw", choices=["raw", "fe"])
    ap.add_argument("--rounds", default="1,2,3")
    ap.add_argument("--sizes", default="all", choices=["all", "full"])
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--max-size", type=int, default=None, help="skip sizes above this (drops the full sample)")
    args = ap.parse_args()

    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS") else 10)
    OUT.mkdir(parents=True, exist_ok=True)
    PREDS.mkdir(parents=True, exist_ok=True)
    make = data.raw_features if args.features == "raw" else data.engineered_features
    raw = data.load_raw()
    names = args.models.split(",")

    for round_ in [int(r) for r in args.rounds.split(",")]:
        train, test = data.split(data.risk_set(raw, round_))
        x_test = make(test, round_)
        sizes = sizes_for(len(train), args.sizes)
        if args.max_size:
            sizes = [s for s in sizes if s <= args.max_size]
        for size in sizes:
            for rep in range(args.reps):
                if size == len(train):
                    sub = train
                else:
                    sub = train.sample(size, random_state=1000 * round_ + rep).reset_index(drop=True)
                x_sub = make(sub, round_)
                for name in names:
                    path = OUT / f"{args.features}_{name}.csv"
                    if (round_, size, rep) in done_keys(path):
                        continue
                    t0 = time.time()
                    p = MODELS[name](x_sub, sub["y"].to_numpy(), x_test, rep)
                    secs = time.time() - t0
                    tab = metrics.score_table(test, test["y"], p)
                    tab.insert(0, "seconds", secs)
                    tab.insert(0, "full", size == len(train))
                    tab.insert(0, "rep", rep)
                    tab.insert(0, "size", size)
                    tab.insert(0, "round", round_)
                    tab.insert(0, "features", args.features)
                    tab.insert(0, "model", name)
                    tab.to_csv(path, mode="a", header=not path.exists(), index=False)
                    if size == len(train) and rep == 0:
                        pred = test[["ID", "Gender", "Race", "y"]].copy()
                        pred["p"] = p
                        pred.to_csv(PREDS / f"r{round_}_{args.features}_{name}.csv.gz", index=False)
                    allrow = tab[tab["group"] == "All"].iloc[0]
                    print(f"r{round_} {args.features} {name} n={size} rep={rep} "
                          f"brier={allrow['brier']:.4f} auc={allrow['auc']:.4f} {secs:.0f}s", flush=True)


if __name__ == "__main__":
    main()
