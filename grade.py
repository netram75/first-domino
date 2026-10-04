# scores a submission with MAP@3
# submission: incident_id, ranking   (ranking = up to 3 service names, space separated, best guess first)

import argparse
import sys

import pandas as pd


def ap_at_3(pred, gold):
    hits, score = 0, 0.0
    for k, p in enumerate(pred[:3], 1):
        if p in gold:
            hits += 1
            score += hits / k
    return score / min(len(gold), 3)


def check(sub, services):
    if list(sub.columns) != ["incident_id", "ranking"]:
        sys.exit(f"columns should be incident_id,ranking, got {list(sub.columns)}")
    if sub.incident_id.duplicated().any():
        sys.exit("some incidents show up more than once")
    missing = set(services) - set(sub.incident_id)
    extra = set(sub.incident_id) - set(services)
    if missing or extra:
        sys.exit(f"{len(missing)} test incidents missing, {len(extra)} unknown ids")
    for row in sub.itertuples():
        names = str(row.ranking).split() if pd.notna(row.ranking) else []
        if len(names) > 3 or len(set(names)) != len(names):
            sys.exit(f"{row.incident_id}: ranking needs up to 3 different services")
        bad = [n for n in names if n not in services[row.incident_id]]
        if bad:
            sys.exit(f"{row.incident_id}: {bad} not in this incident")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--public", default="public")
    ap.add_argument("--private", default="private")
    ap.add_argument("--by-split", action="store_true", help="also show normal vs shifted")
    ap.add_argument("--by-fault", action="store_true", help="also show score per failure type")
    args = ap.parse_args()

    sub = pd.read_csv(args.submission)
    svc = pd.read_csv(f"{args.public}/test/services.csv")
    check(sub, svc.groupby("incident_id").service.apply(set).to_dict())

    gold = pd.read_csv(f"{args.private}/test_labels.csv").set_index("incident_id").root_causes.str.split()
    pred = sub.set_index("incident_id").ranking.fillna("").str.split()
    scores = pd.Series({i: ap_at_3(pred[i], set(g)) for i, g in gold.items()})
    print(f"MAP@3: {scores.mean():.4f}")

    meta = pd.read_csv(f"{args.private}/test_meta.csv").set_index("incident_id")
    if args.by_split:
        for flag, name in [(0, "normal"), (1, "shifted")]:
            print(f"  {name:8s} {scores[meta.shifted == flag].mean():.4f}")
    if args.by_fault:
        kinds = meta.fault_kinds.str.split().str[0]
        for kind in sorted(kinds.unique()):
            part = scores[kinds == kind]
            print(f"  {kind:8s} {part.mean():.4f}  ({len(part)} incidents)")
