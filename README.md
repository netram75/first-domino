# first-domino

![ci](https://github.com/netram75/first-domino/actions/workflows/ci.yml/badge.svg)

Something broke in a system of 15-40 services, and now half of them are red. Find the one that started it.

Chaos engineering tools break services on purpose to see what happens. This is the other side of that: the breaking already happened, you only see the metrics, and you have to work out the cause.

I built it as a small ML challenge: a simulator that makes the incidents, a public/private split, a grader, two simple baselines, a reference solution and rubrics.

Also on Kaggle: the [dataset](https://www.kaggle.com/datasets/netramfaran/first-domino-root-cause-incidents) and a [notebook](https://www.kaggle.com/code/netramfaran/first-domino-finding-the-root-cause) that walks through the baselines, the model with and without the call graph, and where it still fails.

## The task

```
web -> checkout -> payments -> ledger-db
         |
         v
       cart -> cart-cache
```

Say `ledger-db` gets slow. `payments` waits on it and starts timing out, `checkout` waits on `payments`, and `web` shows errors to every user. On a dashboard `web` looks like a disaster and `ledger-db` looks a bit slow. The answer is `ledger-db`.

For each incident you get the call graph (with retries and timeouts), 45 minutes of per-minute metrics for every service, and a list of deploys. You return your top 3 guesses. Full details are in [problem.md](problem.md).

## Files

- `generate_data.py` - simulates the systems and breaks them (random + seeded, no real data)
- `prepare.py` - splits into `public/` and `private/`
- `grade.py` - scores a submission (MAP@3)
- `baseline.py` - two rules: "loudest service" and "first to look weird"
- `solution.py` - reference model, per-service anomaly features + call graph features
- `problem.md` - the task as a solver sees it
- `rubrics.md` - how to judge a solution beyond the score
- `tests/` - grader math, bad submissions, and checks on the simulator

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python generate_data.py --seed 11
.venv/bin/python prepare.py --seed 11
.venv/bin/python solution.py public submission.csv
.venv/bin/python grade.py --submission submission.csv --by-split --by-fault
```

Or just `./run_all.sh`, it does all of the above plus both baselines. Takes under a minute on a laptop. Tests: `.venv/bin/python -m pytest -q`

## Results

| | MAP@3 | normal | shifted |
|---|---|---|---|
| baseline: loudest service | 0.326 | 0.299 | 0.338 |
| baseline: first to look weird | 0.432 | 0.446 | 0.426 |
| solution.py --no-graph | 0.709 | 0.869 | 0.641 |
| solution.py | 0.844 | 0.958 | 0.794 |

The test set has 1000 incidents: 300 that look like train, and 700 shifted ones with bigger systems (23-38 services vs 11-21), deeper call chains, third party APIs, more noise, more incidents with two root causes, and two kinds of failure that never show up in train.

With a different seed (`--seed 21`) the reference gets 0.851 and the "first to look weird" baseline 0.421, so the gap isn't luck.

## Why it's hard

- The loudest service is usually a victim. Timeouts and retries turn a small slowdown at the bottom into a wall of errors at the top. Ranking by error rate gets 0.33.
- The broken service can look healthy. In a gray failure the network in front of a service drops requests. Its own metrics barely move (it never sees the lost requests), while everything calling it times out.
- Victims copy the root's symptoms. Waiting callers burn CPU, and services whose health check pings their dependencies get restarted too, so "high CPU" or "missing data" doesn't point at the root by itself.
- Timing alone isn't enough. The root is the first service to look weird in only 28% of test incidents. Noise makes other services look weird earlier, effects often land in the same minute as the cause, and about 10% of root causes never look weird at all.
- Lots of noise that isn't the problem: traffic going up and down, a traffic spike that moves every service at once, cron jobs, services that always throw a few errors, deploys that have nothing to do with it.

## Shortcuts that don't work

- "highest error rate" - 0.33
- "first to look weird" - 0.43
- a good model on per-service features only - 0.87 on normal incidents but 0.64 on shifted ones. Without the call graph it can't tell a victim from a cause once the systems get bigger.

## Where the reference still fails

From the shifted part of the test set:

- traffic spikes: 0.87 without one, 0.73 with one. A spike changes every service at the same minute and the onset features get confused.
- two root causes at once: 0.71 vs 0.82 for single ones.
- by failure type: gray failure 0.57, small partial failures 0.72, slow memory leaks 0.75 (the last two never show up in train).

So there's room above 0.79 on the shifted part.

## Rules if you try it

- train from scratch on every run, only on `public/`
- no rules written for specific test incidents
- no external data, standard python ML libs only
- fixed seed, results should reproduce

All data here is simulated. No real systems or companies.
