# simulates small microservice systems and breaks one (sometimes two) of the services
# train pool = normal incidents
# test pool = shifted: bigger systems, new failure types, third party apis, more noise
# everything is random, nothing real

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

MINUTES = 45

NAMES = {
    "train": {
        "gateway": ["web", "api-gw", "edge", "mobile-bff"],
        "service": ["auth", "cart", "checkout", "orders", "payments", "search", "catalog", "inventory", "pricing",
                    "users", "notify", "reviews", "shipping", "recs", "promo", "ledger", "profile", "wishlist",
                    "coupons", "tax"],
        "db": ["orders-db", "users-db", "catalog-db", "ledger-db", "reviews-db", "pg-main"],
        "cache": ["session-cache", "cart-cache", "price-cache", "redis-main"],
        "queue": ["events-q", "mail-q", "jobs-q"],
    },
    # only show up in the shifted pool
    "test": {
        "gateway": ["storefront", "partner-api", "graphql"],
        "service": ["billing", "kyc", "loyalty", "refunds", "fraud", "routing", "tracking", "invoices", "subs",
                    "gift-cards", "ratings", "otp", "geo", "feeds", "bookings", "settlement", "ads", "chat",
                    "media", "audit"],
        "db": ["billing-db", "kyc-db", "events-db", "mongo-main", "tracking-db"],
        "cache": ["geo-cache", "token-cache", "feed-cache"],
        "queue": ["kafka-main", "sms-q", "audit-q"],
        "external": ["sms-provider", "payment-gateway", "maps-api", "email-api"],
    },
}

FAULTS = {"train": ["slow", "errors", "crash", "cpu", "gray"], "test": ["leak", "partial"]}
TIMEOUTS = [50, 100, 200, 300, 500, 800, 1000, 2000, 3000, 5000]


def pick_names(r, kind, n, split):
    pool = list(NAMES["train"].get(kind, []))
    if split == "test":
        pool += NAMES["test"].get(kind, [])
    r.shuffle(pool)
    names = pool[:n]
    for k in range(len(names), n):  # ran out of names
        names.append(f"{pool[k % len(pool)]}-{k // len(pool) + 1}")
    return names


def make_system(r, split):
    big = split == "test"
    depth = r.choice([3, 4, 5]) if big else r.choice([2, 3])
    n_gw = r.choice([1, 2, 2]) if big else r.choice([1, 1, 2])
    n_svc = r.randint(14, 24) if big else r.randint(7, 13)
    n_ext = r.randint(1, 3) if big else 0
    stores = [r.choice(["db", "db", "cache", "queue"]) for _ in range(r.randint(6, 9) if big else r.randint(3, 6))]

    nodes = {}
    for name in pick_names(r, "gateway", n_gw, split):
        nodes[name] = {"kind": "gateway", "level": 0}
    levels = list(range(1, depth + 1)) + [r.randint(1, depth) for _ in range(n_svc - depth)]
    for name, level in zip(pick_names(r, "service", n_svc, split), levels):
        nodes[name] = {"kind": "service", "level": level}
    for kind in ("db", "cache", "queue"):
        for name in pick_names(r, kind, stores.count(kind), split):
            nodes[name] = {"kind": kind, "level": depth + 1}
    for name in pick_names(r, "external", n_ext, split):
        nodes[name] = {"kind": "external", "level": depth + 1}

    # who calls who. calls only go down a level or more, so there are no loops
    edges = {}
    for name, info in nodes.items():
        if info["kind"] == "gateway":
            options = [m for m in nodes if nodes[m]["level"] == 1]
            k = r.randint(2, 4)
        elif info["kind"] == "service":
            options = [m for m in nodes if nodes[m]["level"] > info["level"]]
            k = r.randint(1, 3)
        else:
            continue
        for callee in r.sample(options, min(k, len(options))):
            edges[(name, callee)] = None
    # nobody calls it -> give it a caller
    for name, info in nodes.items():
        if info["kind"] != "gateway" and not any(b == name for _, b in edges):
            callers = [m for m in nodes if nodes[m]["level"] < info["level"] and nodes[m]["kind"] in ("gateway", "service")]
            if info["level"] > 1:
                callers = [m for m in callers if nodes[m]["kind"] == "service"]
            edges[(r.choice(callers), name)] = None

    for name, info in nodes.items():
        kind = info["kind"]
        lo, hi = {"gateway": (2, 8), "service": (5, 40), "db": (2, 15), "cache": (0.3, 2),
                  "queue": (1, 5), "external": (60, 250)}[kind]
        info["own_lat"] = r.uniform(lo, hi)
        info["base_err"] = r.uniform(0.002, 0.01) if kind == "external" else r.uniform(0.0002, 0.003)
        info["verbose"] = 1.0 if r.random() < 0.3 else r.uniform(0.02, 0.3)  # how much it logs per error
        info["cpu_idle"] = r.uniform(3, 10)
        info["cpu_normal"] = r.uniform(15, 50)
        # health check also pings its dependencies -> restarts when they fail (bad idea, but common)
        info["deep_health"] = kind in ("gateway", "service") and r.random() < 0.25

    for key in edges:
        calls = 1
        if nodes[key[1]]["kind"] in ("db", "cache") and r.random() < 0.2:
            calls = r.randint(4, 12)  # n+1 queries
        elif r.random() < 0.2:
            calls = r.randint(2, 3)
        edges[key] = {"calls": calls, "retries": r.choice([0, 0, 1, 2, 3]), "lag": r.choice([0, 0, 0, 1])}

    # normal latency and traffic, needed for timeouts and cpu
    order = sorted(nodes, key=lambda n: -nodes[n]["level"])  # callees first
    normal_lat = {}
    for n in order:
        normal_lat[n] = nodes[n]["own_lat"] + sum(e["calls"] * normal_lat[b] for (a, b), e in edges.items() if a == n)
        nodes[n]["normal_lat"] = normal_lat[n]
    for (a, b), e in edges.items():
        want = normal_lat[b] * r.uniform(4, 10)
        e["timeout_ms"] = next((x for x in TIMEOUTS if x >= want), TIMEOUTS[-1])

    normal_rps = {n: 0.0 for n in nodes}
    for n in reversed(order):  # callers first
        if nodes[n]["kind"] == "gateway":
            normal_rps[n] = nodes[n]["rps"] = r.uniform(100, 1500)
        for (a, b), e in edges.items():
            if a == n:
                normal_rps[b] += normal_rps[n] * e["calls"]
    for n, info in nodes.items():
        info["cpu_per_rps"] = (info["cpu_normal"] - info["cpu_idle"]) / max(normal_rps[n], 1)

    return {"nodes": nodes, "edges": edges, "order": order}


def pick_roots(r, system, split):
    nodes = system["nodes"]
    weight = {"gateway": 0.3, "service": 1.0, "db": 1.5, "cache": 1.0, "queue": 0.7, "external": 1.5}
    names = list(nodes)
    n_roots = 2 if r.random() < (0.25 if split == "test" else 0.05) else 1
    t0 = r.randint(15, 32)
    roots = []
    n_callers = {n: sum(b == n for _, b in system["edges"]) for n in names}
    for i in range(n_roots):
        if split == "test" and r.random() < 0.45:
            kind = r.choice(FAULTS["test"])
        else:
            kind = r.choice(FAULTS["train"])
        options = [n for n in names if n not in [f["service"] for f in roots]]
        if kind == "gray":  # usually a shared thing (db, cache...) behind a bad network link
            options = [n for n in options if n_callers[n] >= 2] or options
        service = r.choices(options, weights=[weight[nodes[n]["kind"]] for n in options])[0]
        roots.append({"service": service, "kind": kind, "t0": t0 if i == 0 else min(38, max(10, t0 + r.randint(-5, 5)))})
    return roots


def pick_noise(r, system, roots, split):
    nodes = system["nodes"]
    shifted = split == "test"
    root_names = [f["service"] for f in roots]
    others = [n for n in nodes if n not in root_names and nodes[n]["kind"] != "gateway"]
    t0 = roots[0]["t0"]

    n_flaky = r.choice([1, 1, 2]) if shifted else (1 if r.random() < 0.5 else 0)
    flaky = r.sample(others, n_flaky)
    cron = []
    if r.random() < (0.7 if shifted else 0.4):
        cron.append((r.choice([n for n in others if n not in flaky]), r.choice([10, 15]), r.randint(0, 9)))

    deploys = []
    for f in roots:
        if f["kind"] in ("slow", "errors", "crash") and r.random() < 0.35:
            deploys.append((f["service"], f["t0"] - r.choice([0, 0, 1])))  # bad deploy
    for _ in range(r.choice([0, 1, 1, 2, 3])):
        deploys.append((r.choice(others), r.randint(0, MINUTES - 1)))
    if r.random() < (0.5 if shifted else 0.35):
        deploys.append((r.choice(others), max(0, min(MINUTES - 1, t0 + r.randint(-3, 3)))))  # right next to the incident

    spike = None
    if r.random() < (0.5 if shifted else 0.15):
        spike = (r.randint(10, 40), r.uniform(1.3, 1.8))  # traffic jump, not a failure
    return {"flaky": flaky, "cron": cron, "deploys": deploys, "spike": spike}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def lagged(a, d):
    return a if d == 0 else np.concatenate([np.repeat(a[:1], d), a[:-d]])


def apply_fault(rng, f, own_lat, own_err, cpu_extra, down, lost, t):
    s, kind, t0 = f["service"], f["kind"], f["t0"]
    step = np.clip(t - t0 + rng.uniform(0.2, 1), 0, 1)  # first minute is only partly broken
    if kind == "slow":
        own_lat[s] = own_lat[s] * (1 + step * (rng.uniform(4, 15) - 1))
    elif kind == "errors":
        own_err[s] = own_err[s] + step * rng.uniform(0.08, 0.5)
    elif kind == "partial":  # one bad replica, small signal
        own_err[s] = own_err[s] + step * rng.uniform(0.02, 0.06)
    elif kind == "cpu":
        ramp = np.clip((t - t0 + 1) / 3, 0, 1)
        cpu_extra[s] += ramp * rng.uniform(45, 60)
        own_lat[s] = own_lat[s] * (1 + ramp * (rng.uniform(2, 5) - 1))
    elif kind == "leak":  # slow climb, starts failing near the end
        ramp = np.clip((t - t0 + 1) / rng.uniform(12, 25), 0, 1)
        own_lat[s] = own_lat[s] * (1 + ramp * (rng.uniform(4, 8) - 1))
        own_err[s] = own_err[s] + ramp ** 3 * rng.uniform(0.05, 0.2)
        cpu_extra[s] += ramp * rng.uniform(10, 25)
    elif kind == "gray":  # network to it drops requests. it looks fine itself, callers time out
        lost[s] = step * rng.uniform(0.15, 0.5)
    elif kind == "crash":  # crash loop: down a few minutes, up, down again
        m = t0
        while m < MINUTES:
            d = int(rng.integers(2, 5))
            down[s][m:m + d] = True
            m += d + int(rng.integers(2, 6))


def simulate(r, system, roots, noise):
    nodes, edges, order = system["nodes"], system["edges"], system["order"]
    rng = np.random.default_rng(r.getrandbits(32))
    t = np.arange(MINUTES)
    callees = {n: [b for (a, b) in edges if a == n] for n in nodes}

    own_lat = {n: info["own_lat"] * rng.lognormal(0, 0.04, MINUTES) for n, info in nodes.items()}
    own_err = {n: np.full(MINUTES, info["base_err"]) for n, info in nodes.items()}
    cpu_extra = {n: np.zeros(MINUTES) for n in nodes}
    down = {n: np.zeros(MINUTES, bool) for n in nodes}
    lost = {n: np.zeros(MINUTES) for n in nodes}

    for n in noise["flaky"]:
        own_err[n] = rng.uniform(0.02, 0.06) + (rng.random(MINUTES) < 0.12) * rng.uniform(0.05, 0.15, MINUTES)
    for n, period, offset in noise["cron"]:
        busy = ((t - offset) % period) < 2
        own_lat[n] = own_lat[n] * np.where(busy, rng.uniform(2, 4), 1)
        cpu_extra[n] += busy * rng.uniform(25, 40)
    for n, minute in noise["deploys"]:
        own_lat[n][minute] *= 1.6  # restart blip
        cpu_extra[n][minute] += 15
    for f in roots:
        apply_fault(rng, f, own_lat, own_err, cpu_extra, down, lost, t)

    # latency and errors flow up from callees to callers
    lat, err, tries = {}, {}, {}
    for n in order:
        total = own_lat[n].copy()
        ok = 1 - np.clip(own_err[n], 0, 1)
        worst = np.zeros(MINUTES)
        for c in callees[n]:
            e = edges[(n, c)]
            lc, ec, lc_lost = lagged(lat[c], e["lag"]), lagged(err[c], e["lag"]), lagged(lost[c], e["lag"])
            timed_out = sigmoid(8 * (lc / e["timeout_ms"] - 1))
            q = 1 - (1 - ec) * (1 - timed_out) * (1 - lc_lost)  # one attempt fails
            attempts = sum(q ** i for i in range(e["retries"] + 1))
            wait = np.minimum(lc, e["timeout_ms"]) * (1 - lc_lost) + e["timeout_ms"] * lc_lost
            total = total + e["calls"] * wait * attempts
            ok = ok * (1 - q ** (e["retries"] + 1)) ** e["calls"]
            tries[(n, c)] = attempts
            worst = np.maximum(worst, q)
        if nodes[n]["deep_health"]:
            m = 1
            while m < MINUTES:
                if worst[m] > 0.5 and worst[m - 1] > 0.5:
                    down[n][m + 1:m + 3] = True
                    m += 6  # cooldown before it can restart again
                else:
                    m += 1
        lat[n] = np.where(down[n], own_lat[n], total)
        err[n] = np.where(down[n], 1.0, 1 - ok)
        # requests piling up while waiting cost cpu too
        cpu_extra[n] += 8 * np.clip(np.log2(total / nodes[n]["normal_lat"]), 0, 3)

    # traffic flows down. retries mean more traffic to whoever is failing
    rps = {n: np.zeros(MINUTES) for n in nodes}
    for n in reversed(order):
        if nodes[n]["kind"] == "gateway":
            trend = np.linspace(1, r.uniform(0.8, 1.25), MINUTES)
            if noise["spike"]:
                trend = trend * np.where(t >= noise["spike"][0], noise["spike"][1], 1)
            rps[n] = nodes[n]["rps"] * trend * rng.lognormal(0, 0.03, MINUTES)
        sending = np.where(down[n], 0, rps[n])
        for c in callees[n]:
            rps[c] = rps[c] + sending * edges[(n, c)]["calls"] * tries[(n, c)] * (1 - lost[c])

    cols = {k: [] for k in ("service", "minute", "rps", "latency_ms", "error_rate", "cpu", "log_errors")}
    for n, info in nodes.items():
        reqs = np.maximum(rps[n] * 60, 1).astype(int)
        cpu = info["cpu_idle"] + info["cpu_per_rps"] * rps[n] + cpu_extra[n] + rng.normal(0, 1.5, MINUTES)
        gone = down[n] | (rng.random(MINUTES) < 0.004)  # down, or the scrape just failed
        values = {
            "rps": np.round(rng.poisson(reqs) / 60, 2),
            "latency_ms": np.round(lat[n] * rng.lognormal(0, 0.06, MINUTES), 1),
            "error_rate": np.round(rng.binomial(reqs, np.clip(err[n], 0, 1)) / reqs, 5),
            "cpu": np.round(np.clip(cpu, 0, 100), 1),
            "log_errors": rng.poisson(reqs * np.clip(err[n], 0, 1) * info["verbose"]).astype(float),
        }
        cols["service"] += [n] * MINUTES
        cols["minute"].append(t)
        for k, v in values.items():
            cols[k].append(np.where(gone, np.nan, v))
    out = pd.DataFrame({k: (v if k == "service" else np.concatenate(v)) for k, v in cols.items()})
    out["log_errors"] = out["log_errors"].astype("Int64")
    return out


def make_pool(n, split, seed):
    r = random.Random(seed)
    tables = {"services": [], "calls": [], "metrics": [], "deploys": [], "labels": [], "meta": []}
    for i in range(n):
        inc = f"{split}{i:05d}"
        system = make_system(r, split)
        roots = pick_roots(r, system, split)
        noise = pick_noise(r, system, roots, split)
        m = simulate(r, system, roots, noise)
        m.insert(0, "incident_id", inc)
        tables["metrics"].append(m)

        nodes, edges = system["nodes"], system["edges"]
        tables["services"].append(pd.DataFrame({"incident_id": inc, "service": list(nodes),
                                                "kind": [v["kind"] for v in nodes.values()]}))
        tables["calls"].append(pd.DataFrame([{"incident_id": inc, "caller": a, "callee": b, "calls_per_request": e["calls"],
                                              "timeout_ms": e["timeout_ms"], "retries": e["retries"]}
                                             for (a, b), e in edges.items()]))
        tables["deploys"].append(pd.DataFrame([{"incident_id": inc, "service": s, "minute": mi}
                                               for s, mi in sorted(set(noise["deploys"]), key=lambda x: x[1])],
                                              columns=["incident_id", "service", "minute"]))
        tables["labels"].append(pd.DataFrame([{"incident_id": inc, "root_causes": " ".join(f["service"] for f in roots)}]))
        tables["meta"].append(pd.DataFrame([{
            "incident_id": inc, "split": split, "n_services": len(nodes), "n_roots": len(roots),
            "fault_kinds": " ".join(f["kind"] for f in roots), "t0": roots[0]["t0"],
            "root_kinds": " ".join(nodes[f["service"]]["kind"] for f in roots),
            "traffic_spike": int(noise["spike"] is not None), "n_flaky": len(noise["flaky"]),
            "cron": int(bool(noise["cron"])), "n_deploys": len(set(noise["deploys"])),
        }]))
    return {k: pd.concat(v, ignore_index=True) for k, v in tables.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--n-train", type=int, default=1200)
    ap.add_argument("--n-test", type=int, default=700)
    ap.add_argument("--out", default="raw")
    args = ap.parse_args()

    for split, n, seed in [("train", args.n_train, args.seed), ("test", args.n_test, args.seed + 1)]:
        out = Path(args.out) / f"{split}_pool"
        out.mkdir(parents=True, exist_ok=True)
        pool = make_pool(n, split, seed)
        for name, df in pool.items():
            df.to_csv(out / f"{name}.csv", index=False)
        print(f"{split} pool: {n} incidents, {len(pool['metrics'])} metric rows")
