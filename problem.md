# Find the first domino

A company runs a bunch of small services that call each other. Something breaks, and a few minutes later half of them are throwing errors. Your job: find the service that started it.

## Data

Every incident is a separate little system with 45 minutes of per-minute metrics. The trouble always starts after the first 10 minutes.

`train/` and `test/` have the same four files:

- `services.csv` - `incident_id, service, kind` (kind is gateway, service, db, cache, queue...)
- `calls.csv` - `incident_id, caller, callee, calls_per_request, timeout_ms, retries`
  - the caller depends on the callee. if the callee is slow or failing, the caller feels it
- `metrics.csv` - `incident_id, service, minute, rps, latency_ms, error_rate, cpu, log_errors`
  - `latency_ms` is p95. a blank row means no data came in that minute
- `deploys.csv` - `incident_id, service, minute` (deploys that happened during the window)

`train_labels.csv` - `incident_id, root_causes`. Usually one service, sometimes two (space separated).

Service names are just names. The same name in two incidents is not the same service.

## What to submit

One row per test incident, your top 3 guesses, best first:

```
incident_id,ranking
inc00012,orders-db checkout cart
```

## Score

MAP@3. If the root cause is your first guess you get 1, second guess 1/2, third 1/3, otherwise 0. When an incident has two root causes, both count.

## Things to know

- Services that call a broken service often look much worse than the broken one.
- There's normal noise in every incident: traffic going up and down, scheduled jobs, services that always throw a few errors, deploys that have nothing to do with the problem.
- The test set has bigger systems than train, and a few kinds of trouble that never show up in train.

## Rules

- `python solution.py <public_dir> <submission_out>` must train from scratch on every run, using only what's in `public/`
- no hand-written answers or rules aimed at specific test incidents
- no outside data, standard python ML libraries only
- fixed seeds, results should reproduce
