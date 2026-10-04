# reference solution
# idea: for every service, work out when it started acting weird and how much,
# then use the call graph - a root cause goes weird first, and the stuff it calls is still fine.
# trains from scratch on every run.
# usage: python solution.py <public_dir> <submission_out> [--no-graph]

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

MINUTES = 45
BASE = 10  # first 10 minutes = what normal looks like
METRICS = ["latency_ms", "error_rate", "cpu", "rps", "log_errors"]
KINDS = ["gateway", "service", "db", "cache", "queue"]
# columns that come from the call graph (dropped with --no-graph)
GRAPH = ["n_callees", "n_callers", "weird_callees", "callee_earlier", "callee_gap", "weird_callers", "caller_gap",
         "n_above", "weird_above", "weird_below", "explains", "clean_below", "caller_weird_frac", "bottom_callers",
         "callee_shared_weird"]


def per_service(metrics):
    m = metrics.sort_values(["incident_id", "service", "minute"])
    keys = m[["incident_id", "service"]].iloc[::MINUTES].reset_index(drop=True)
    f = keys.copy()
    z = {}
    for col in METRICS:
        x = m[col].to_numpy(float).reshape(-1, MINUTES)
        base = np.nanmedian(x[:, :BASE], axis=1)
        q75, q25 = np.nanpercentile(x[:, :BASE], [75, 25], axis=1)
        spread = (q75 - q25) + 0.05 * np.abs(base) + {"error_rate": 2e-3, "cpu": 1, "latency_ms": 0.5}.get(col, 1)
        z[col] = (x - base[:, None]) / spread[:, None]
        after = z[col][:, BASE:]
        f[f"{col}_max"] = np.nanmax(after, axis=1).clip(-50, 200)
        f[f"{col}_min"] = np.nanmin(after, axis=1).clip(-50, 200)
        f[f"{col}_late"] = np.nanmean(z[col][:, -10:], axis=1).clip(-50, 200)
        f[f"{col}_ratio"] = np.log1p(np.nanmean(x[:, -10:], axis=1).clip(0)) - np.log1p(np.clip(base, 0, None))

    gone = np.isnan(m["latency_ms"].to_numpy(float).reshape(-1, MINUTES))
    weird = (z["latency_ms"] > 4) | (z["error_rate"] > 4) | (z["cpu"] > 4) | gone
    weird[:, :BASE] = False
    steady = weird[:, :-1] & weird[:, 1:]  # two weird minutes in a row
    f["onset"] = np.where(steady.any(axis=1), steady.argmax(axis=1), 99)
    for col in ["latency_ms", "error_rate", "cpu"]:
        s = (z[col][:, :-1] > 4) & (z[col][:, 1:] > 4)
        s[:, :BASE] = False
        f[f"{col}_onset"] = np.where(s.any(axis=1), s.argmax(axis=1), 99)
    f["missing"] = gone[:, BASE:].sum(axis=1)
    f["weird_minutes"] = weird.sum(axis=1)
    return f


def add_graph(f, services, calls, deploys):
    f = f.merge(services, on=["incident_id", "service"], how="left")
    for k in KINDS:
        f[f"is_{k}"] = (f.kind == k).astype(int)

    first = f.groupby("incident_id").onset.transform("min")
    f["onset_gap"] = (f.onset - first).clip(upper=99)
    f["onset_rank"] = f.groupby("incident_id").onset.rank(method="min")
    for col in ["latency_ms_max", "error_rate_max", "cpu_max", "rps_max"]:
        f[f"{col}_share"] = f[col] / f.groupby("incident_id")[col].transform("max").clip(lower=1e-6)

    onset = dict(zip(zip(f.incident_id, f.service), f.onset))
    down, up = defaultdict(list), defaultdict(list)
    for row in calls.itertuples():
        down[(row.incident_id, row.caller)].append(row.callee)
        up[(row.incident_id, row.callee)].append(row.caller)

    def reach(inc, start, graph):
        seen, todo = set(), [start]
        while todo:
            for nxt in graph[(inc, todo.pop())]:
                if nxt not in seen:
                    seen.add(nxt)
                    todo.append(nxt)
        return seen

    n_weird = (f.onset < 99).groupby(f.incident_id).sum().to_dict()
    is_weird = {k: o < 99 for k, o in onset.items()}
    # weird, and nothing it calls is weird -> the bottom of a weird chain
    bottom = {(i, s): w and not any(is_weird[(i, c)] for c in down[(i, s)]) for (i, s), w in is_weird.items()}
    rows = []
    for inc, s, t in zip(f.incident_id, f.service, f.onset):
        callees, callers = down[(inc, s)], up[(inc, s)]
        callee_on = [onset[(inc, c)] for c in callees]
        caller_on = [onset[(inc, c)] for c in callers]
        # if something i call has other callers that are also weird, i'm probably a victim of it
        shared = [sum(is_weird[(inc, o)] for o in up[(inc, c)] if o != s) for c in callees]
        above = reach(inc, s, up)    # everything that depends on s
        below = reach(inc, s, down)  # everything s depends on
        weird_above = sum(onset[(inc, a)] < 99 for a in above)
        weird_below = sum(onset[(inc, b)] < 99 for b in below)
        rows.append({
            "n_callees": len(callees), "n_callers": len(callers),
            "weird_callees": sum(o < 99 for o in callee_on),
            "callee_earlier": sum(o < t for o in callee_on),
            "callee_gap": (min(callee_on) - t) if callee_on else 99,
            "weird_callers": sum(o < 99 for o in caller_on),
            "caller_gap": (min(caller_on) - t) if caller_on else 99,
            "n_above": len(above), "weird_above": weird_above, "weird_below": weird_below,
            "explains": weird_above / max(n_weird[inc] - (t < 99), 1),
            "clean_below": int(weird_below == 0),
            "caller_weird_frac": sum(o < 99 for o in caller_on) / max(len(callers), 1),
            "bottom_callers": sum(bottom[(inc, c)] for c in callers),
            "callee_shared_weird": max(shared, default=0),
        })
    f = pd.concat([f.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

    dep = deploys.merge(f[["incident_id", "service", "onset"]], on=["incident_id", "service"])
    near = dep[(dep.minute >= dep.onset - 2) & (dep.minute <= dep.onset + 1)]
    f["deploy_near_onset"] = f.set_index(["incident_id", "service"]).index.isin(
        near.set_index(["incident_id", "service"]).index).astype(int)
    f["deploys"] = f.merge(deploys.groupby(["incident_id", "service"]).size().rename("d").reset_index(),
                           on=["incident_id", "service"], how="left").d.fillna(0).values
    return f


def build(public, part):
    d = Path(public) / part
    f = per_service(pd.read_csv(d / "metrics.csv"))
    return add_graph(f, pd.read_csv(d / "services.csv"), pd.read_csv(d / "calls.csv"), pd.read_csv(d / "deploys.csv"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("public")
    ap.add_argument("out")
    ap.add_argument("--no-graph", action="store_true", help="only per-service features, to see what the graph adds")
    args = ap.parse_args()
    public, out = args.public, args.out

    train = build(public, "train")
    test = build(public, "test")

    roots = pd.read_csv(Path(public) / "train_labels.csv")
    roots = {(i, s) for i, r in zip(roots.incident_id, roots.root_causes) for s in r.split()}
    y = np.array([(i, s) in roots for i, s in zip(train.incident_id, train.service)], int)

    cols = [c for c in train.columns if c not in ("incident_id", "service", "kind")]
    if args.no_graph:
        cols = [c for c in cols if c not in GRAPH]
    model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                           l2_regularization=1.0, random_state=0)
    model.fit(train[cols], y)

    test["score"] = model.predict_proba(test[cols])[:, 1]
    test = test.sort_values(["incident_id", "score"], ascending=[True, False])
    sub = test.groupby("incident_id").service.apply(lambda s: " ".join(s.head(3))).reset_index(name="ranking")
    sub.to_csv(out, index=False)
    print("wrote", out)
