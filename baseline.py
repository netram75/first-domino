# two simple rules, no learning
#   loudest: blame whoever's error rate went up the most
#   first:   blame whoever started looking weird first
# usage: python baseline.py <public_dir> <submission_out> [--rule loudest|first]

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def loudest(m):
    keys = ["incident_id", "service"]
    early = m[m.minute < 10].groupby(keys)[["error_rate", "latency_ms"]].median()
    late = m[m.minute >= 35].groupby(keys)[["error_rate", "latency_ms"]].mean()
    # error jump first, latency ratio only breaks ties
    return (late.error_rate - early.error_rate).fillna(1) + 1e-3 * (late.latency_ms / early.latency_ms).fillna(0)


def first(m):
    keys = ["incident_id", "service"]
    base = m[m.minute < 10].groupby(keys)[["latency_ms", "error_rate"]]
    med = base.median()
    spread = (base.quantile(0.75) - base.quantile(0.25)).clip(lower=1e-3) + 0.05 * med
    m = m.join(med, on=keys, rsuffix="_med").join(spread, on=keys, rsuffix="_spread")
    z = np.maximum((m.latency_ms - m.latency_ms_med) / m.latency_ms_spread,
                   (m.error_rate - m.error_rate_med) / m.error_rate_spread)
    weird = (z > 5) | m.latency_ms.isna()
    m["weird"] = weird & (m.minute >= 10)
    m = m.sort_values(keys + ["minute"])
    # needs two weird minutes in a row
    m["weird2"] = m.weird & m.groupby(keys).weird.shift(-1, fill_value=False)
    onset = m[m.weird2].groupby(keys).minute.min()
    onset = onset.reindex(med.index).fillna(99)
    size = pd.Series(z.values, index=pd.MultiIndex.from_frame(m[keys])).groupby(keys).max()
    return -onset + 1e-4 * size.reindex(med.index).fillna(0).clip(upper=100)


def top3(score):
    score = score.rename("score").reset_index().sort_values(["incident_id", "score"], ascending=[True, False])
    return score.groupby("incident_id").service.apply(lambda s: " ".join(s.head(3))).reset_index(name="ranking")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("public")
    ap.add_argument("out")
    ap.add_argument("--rule", default="loudest", choices=["loudest", "first"])
    args = ap.parse_args()

    metrics = pd.read_csv(Path(args.public) / "test" / "metrics.csv")
    score = loudest(metrics) if args.rule == "loudest" else first(metrics)
    top3(score).to_csv(args.out, index=False)
    print("wrote", args.out)
