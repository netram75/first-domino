# splits the raw pools into public/ (what a solver gets) and private/ (answers)
# test = some normal incidents held out from the train pool + the whole shifted pool, shuffled

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TABLES = ["services", "calls", "metrics", "deploys"]


def load(folder):
    return {name: pd.read_csv(folder / f"{name}.csv") for name in TABLES + ["labels", "meta"]}


def write(tables, ids, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        df = tables[name][tables[name].incident_id.isin(ids)].copy()
        df["incident_id"] = df.incident_id.map(ids)
        df.sort_values("incident_id", kind="stable").to_csv(folder / f"{name}.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--n-holdout", type=int, default=300)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    train = load(Path(args.raw) / "train_pool")
    shifted = load(Path(args.raw) / "test_pool")

    all_train = train["labels"].incident_id.tolist()
    held = set(rng.choice(all_train, args.n_holdout, replace=False))
    train_ids = [i for i in all_train if i not in held]
    test_ids = sorted(held) + shifted["labels"].incident_id.tolist()

    # fresh random ids so the name doesn't give away which pool it came from
    numbers = rng.permutation(len(train_ids) + len(test_ids))
    new_id = {old: f"inc{n:05d}" for old, n in zip(train_ids + test_ids, numbers)}
    tr_map = {i: new_id[i] for i in train_ids}
    te_map = {i: new_id[i] for i in test_ids}

    pub, priv = Path("public"), Path("private")
    write(train, tr_map, pub / "train")
    both = {k: pd.concat([train[k], shifted[k]], ignore_index=True) for k in train}
    write(both, te_map, pub / "test")

    labels = train["labels"][train["labels"].incident_id.isin(tr_map)].copy()
    labels["incident_id"] = labels.incident_id.map(tr_map)
    labels.sort_values("incident_id").to_csv(pub / "train_labels.csv", index=False)

    # sample submission: 3 random services per incident
    services = both["services"][both["services"].incident_id.isin(te_map)]
    sample = (services.groupby("incident_id").service
              .apply(lambda s: " ".join(rng.choice(s.values, 3, replace=False))).reset_index(name="ranking"))
    sample["incident_id"] = sample.incident_id.map(te_map)
    sample.sort_values("incident_id").to_csv(pub / "sample_submission.csv", index=False)

    priv.mkdir(exist_ok=True)
    test_labels = both["labels"][both["labels"].incident_id.isin(te_map)].copy()
    test_labels["incident_id"] = test_labels.incident_id.map(te_map)
    test_labels.sort_values("incident_id").to_csv(priv / "test_labels.csv", index=False)
    meta = both["meta"][both["meta"].incident_id.isin(te_map)].copy()
    meta["shifted"] = (meta.split == "test").astype(int)
    meta["incident_id"] = meta.incident_id.map(te_map)
    meta.drop(columns="split").sort_values("incident_id").to_csv(priv / "test_meta.csv", index=False)

    print(f"train: {len(train_ids)} incidents | test: {len(test_ids)} ({len(held)} normal + {len(test_ids) - len(held)} shifted)")
