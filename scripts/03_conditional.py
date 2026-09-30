"""Why rounds 2 and 3 need the risk set.

Fit each model two ways on the full NIJ training sample and score both on the
round's test sample (people with no arrest in earlier years, which is who NIJ
scored):

  risk set - train only on people with no arrest in earlier years
  naive    - train on everyone, so an earlier arrest counts as y = 0

Writes results/conditional.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import psutil

from tabrecid import data, metrics
from tabrecid.models import MODELS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "conditional.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="catboost,kumo_small")
    ap.add_argument("--features", default="fe", choices=["raw", "fe"])
    args = ap.parse_args()
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)

    make = data.raw_features if args.features == "raw" else data.engineered_features
    raw = data.load_raw()
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["model", "round", "train"])
    for round_ in [2, 3]:
        _, test = data.split(data.risk_set(raw, round_))
        x_test = make(test, round_)
        sets = {
            "risk set": data.split(data.risk_set(raw, round_))[0],
            "naive": data.split(data.naive_set(raw, round_))[0],
        }
        for name in args.models.split(","):
            for label, train in sets.items():
                hit = done[(done["model"] == name) & (done["round"] == round_) & (done["train"] == label)]
                if len(hit):
                    continue
                p = MODELS[name](make(train, round_), train["y"].to_numpy(), x_test, 0)
                tab = metrics.score_table(test, test["y"], p)
                tab.insert(0, "n_train", len(train))
                tab.insert(0, "train", label)
                tab.insert(0, "round", round_)
                tab.insert(0, "features", args.features)
                tab.insert(0, "model", name)
                tab.to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
                a = tab[tab["group"] == "All"].iloc[0]
                print(f"r{round_} {name} {label} n={len(train)} brier={a['brier']:.4f} "
                      f"auc={a['auc']:.4f} mean_pred={a['mean_pred']:.3f} base={a['base_rate']:.3f}", flush=True)


if __name__ == "__main__":
    main()
