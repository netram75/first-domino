# reference solution
# idea: for every service, work out when it started acting weird and how much,
# then use the call graph - a root cause goes weird first, and the stuff it calls is still fine.
# v2: host clocks can be off, so line them up first using traffic. services with no metrics
# still get ranked from their callers.
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
         "callee_shared_weird", "hidden_callees", "sibling_earlier", "clock", "only_suspect_for", "only_suspect_frac"]
MAX_LAG = 6


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

    # cpu per request, so a traffic spike (more requests, more cpu) doesn't look like a problem
    load = (m["cpu"].to_numpy(float) / m["rps"].to_numpy(float).clip(0.01)).reshape(-1, MINUTES)
    base = np.nanmedian(load[:, :BASE], axis=1)
    q75, q25 = np.nanpercentile(load[:, :BASE], [75, 25], axis=1)
    z["load"] = (load - base[:, None]) / ((q75 - q25) + 0.05 * np.abs(base) + 1e-6)[:, None]
    f["load_max"] = np.nanmax(z["load"][:, BASE:], axis=1).clip(-50, 200)
    f["load_late"] = np.nanmean(z["load"][:, -10:], axis=1).clip(-50, 200)

    gone = np.isnan(m["latency_ms"].to_numpy(float).reshape(-1, MINUTES))
    weird = (z["latency_ms"] > 4) | (z["error_rate"] > 4) | (z["load"] > 4) | gone
    weird[:, :BASE] = False
    steady = weird[:, :-1] & weird[:, 1:]  # two weird minutes in a row
    f["onset_first"] = np.where(steady.any(axis=1), steady.argmax(axis=1), 99)
    # real trouble sticks around, a cron job or a blip doesn't. so the onset is where the trouble
    # starts that's still there most of the time until the end
    left = np.cumsum(weird[:, ::-1], axis=1)[:, ::-1] / np.arange(MINUTES, 0, -1)
    lasting = weird & (left >= 0.7)
    lasting[:, :BASE] = False
    f["onset"] = np.where(lasting.any(axis=1), lasting.argmax(axis=1), 99)
    for col in ["latency_ms", "error_rate", "cpu"]:
        s = (z[col][:, :-1] > 4) & (z[col][:, 1:] > 4)
        s[:, :BASE] = False
        f[f"{col}_onset"] = np.where(s.any(axis=1), s.argmax(axis=1), 99)
    late = np.nan_to_num(np.maximum.reduce([np.nanmean(z[c][:, -10:], axis=1) for c in ("latency_ms", "error_rate", "load")]))
    f["affected"] = ((late > 3) | (gone[:, -10:].sum(axis=1) >= 3)).astype(int)  # still in trouble at the end
    f["missing"] = gone[:, BASE:].sum(axis=1)
    f["weird_minutes"] = weird.sum(axis=1)
    # did traffic jump right before it went weird? (flooded by someone else's failure)
    rps_z = np.nan_to_num(z["rps"], nan=0.0)
    on = f["onset_first"].to_numpy()
    f["rps_before_onset"] = [rps_z[i, max(0, o - 3):o].mean() if 3 <= o < 99 else 0.0 for i, o in enumerate(on)]
    return f


def best_lag(caller, callee):
    # same traffic wiggles show up in caller and callee in the same (real) minute
    best, best_r = 0, -1.0
    n = len(callee)
    for lag in range(-MAX_LAG, MAX_LAG + 1):
        a = callee[max(0, -lag):min(n, n - lag)]
        b = caller[max(0, lag):min(n, n + lag)]
        if a.std() > 0 and b.std() > 0:
            r = np.corrcoef(a, b)[0, 1]
            if r > best_r:
                best, best_r = lag, r
    return best, best_r


def clock_offsets(metrics, calls):
    m = metrics.sort_values(["incident_id", "service", "minute"])
    keys = m[["incident_id", "service"]].iloc[::MINUTES]
    x = pd.DataFrame(np.log(m["rps"].to_numpy(float).reshape(-1, MINUTES).clip(0.01)))
    x = np.diff(x.interpolate(axis=1, limit_direction="both").to_numpy(), axis=1)
    series = dict(zip(zip(keys.incident_id, keys.service), x))

    out = {}
    for inc, g in calls.groupby("incident_id"):
        links = []
        for a, b in zip(g.caller, g.callee):
            if (inc, a) in series and (inc, b) in series:
                lag, r = best_lag(series[(inc, a)], series[(inc, b)])
                links.append((r, a, b, lag))
        # start from the busiest caller and grow along the best matching calls
        known = {}
        links.sort(reverse=True)
        while links:
            if not known:
                known[links[0][1]] = 0
            for i, (r, a, b, lag) in enumerate(links):
                if (a in known) != (b in known):
                    if a in known:
                        known[b] = known[a] - lag
                    else:
                        known[a] = known[b] + lag
                    links.pop(i)
                    break
            else:
                links = [link for link in links if link[1] not in known and link[2] not in known]
                if links:
                    known[links[0][1]] = 0  # a part that isn't connected to the rest
        for s, k in known.items():
            out[(inc, s)] = k
    return out


def add_graph(f, calls, deploys):
    for k in KINDS:
        f[f"is_{k}"] = (f.kind == k).astype(int)

    first = f.groupby("incident_id").onset.transform("min")
    f["onset_gap"] = (f.onset - first).clip(upper=99)
    f["onset_rank"] = f.groupby("incident_id").onset.rank(method="min")
    for col in ["latency_ms_max", "error_rate_max", "cpu_max", "rps_max"]:
        f[f"{col}_share"] = f[col] / f.groupby("incident_id")[col].transform("max").clip(lower=1e-6)

    # for the graph only count services that are still in trouble at the end
    onset = dict(zip(zip(f.incident_id, f.service), np.where(f.affected.fillna(0) == 1, f.onset, 99)))
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

    is_weird = {k: o < 99 for k, o in onset.items()}
    n_weird = pd.Series(is_weird.values(), index=[k[0] for k in is_weird]).groupby(level=0).sum().to_dict()
    seen = dict(zip(zip(f.incident_id, f.service), f.observed))
    # weird, and nothing it calls is weird -> the bottom of a weird chain
    bottom = {(i, s): w and not any(is_weird[(i, c)] for c in down[(i, s)]) for (i, s), w in is_weird.items()}
    rows = []
    for inc, s, t in zip(f.incident_id, f.service, f.onset):
        callees, callers = down[(inc, s)], up[(inc, s)]
        callee_on = [onset[(inc, c)] for c in callees]
        caller_on = [onset[(inc, c)] for c in callers]
        # if something i call has other callers that are also weird, i'm probably a victim of it
        shared = [sum(is_weird[(inc, o)] for o in up[(inc, c)] if o != s) for c in callees]
        # something next to me (same callers) went weird before me -> maybe its traffic landed on me
        siblings = {c for p in callers for c in down[(inc, p)]} - {s}
        # weird callers that have nothing else to blame: everything else they call looks fine
        blame = 0
        for p in callers:
            if is_weird[(inc, p)]:
                blame += not any(is_weird[(inc, c)] or not seen[(inc, c)] for c in down[(inc, p)] if c != s)
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
            "hidden_callees": sum(not seen[(inc, c)] for c in callees),
            "sibling_earlier": sum(onset[(inc, c)] < t for c in siblings),
            "only_suspect_for": blame,
            "only_suspect_frac": blame / max(len(callers), 1),
        })
    f = pd.concat([f.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

    dep = deploys.merge(f[["incident_id", "service", "onset"]], on=["incident_id", "service"])
    near = dep[(dep.minute >= dep.onset - 2) & (dep.minute <= dep.onset + 1)]
    f["deploy_near_onset"] = f.set_index(["incident_id", "service"]).index.isin(
        near.set_index(["incident_id", "service"]).index).astype(int)
    f["deploys"] = f.merge(deploys.groupby(["incident_id", "service"]).size().rename("d").reset_index(),
                           on=["incident_id", "service"], how="left").d.fillna(0).values
    return f


def build(public, part, use_graph=True):
    d = Path(public) / part
    metrics, calls = pd.read_csv(d / "metrics.csv"), pd.read_csv(d / "calls.csv")
    f = per_service(metrics)
    if use_graph:
        clock = clock_offsets(metrics, calls)
        f["clock"] = [clock.get(k, 0) for k in zip(f.incident_id, f.service)]
        for col in ["onset", "onset_first", "latency_ms_onset", "error_rate_onset", "cpu_onset"]:
            f[col] = np.where(f[col] < 99, f[col] - f.clock, 99)
    # services with no metrics are still suspects, they just have no numbers of their own
    f = pd.read_csv(d / "services.csv").merge(f, on=["incident_id", "service"], how="left")
    f["observed"] = f.onset.notna().astype(int)
    f["onset"] = f.onset.fillna(99)
    return add_graph(f, calls, pd.read_csv(d / "deploys.csv"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("public")
    ap.add_argument("out")
    ap.add_argument("--no-graph", action="store_true", help="only per-service features, to see what the graph adds")
    args = ap.parse_args()
    public, out = args.public, args.out

    train = build(public, "train", not args.no_graph)
    test = build(public, "test", not args.no_graph)

    roots = pd.read_csv(Path(public) / "train_labels.csv")
    roots = {(i, s) for i, r in zip(roots.incident_id, roots.root_causes) for s in r.split()}
    y = np.array([(i, s) in roots for i, s in zip(train.incident_id, train.service)], int)

    cols = [c for c in train.columns if c not in ("incident_id", "service", "kind")]
    if args.no_graph:
        cols = [c for c in cols if c not in GRAPH]
    # 5 models with different seeds, averaged. a bit steadier than one
    test["score"] = 0.0
    for seed in range(5):
        model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                               l2_regularization=1.0, random_state=seed)
        model.fit(train[cols], y)
        test["score"] += model.predict_proba(test[cols])[:, 1] / 5
    test = test.sort_values(["incident_id", "score"], ascending=[True, False])
    sub = test.groupby("incident_id").service.apply(lambda s: " ".join(s.head(3))).reset_index(name="ranking")
    sub.to_csv(out, index=False)
    print("wrote", out)
