# first-domino

![ci](https://github.com/netram75/first-domino/actions/workflows/ci.yml/badge.svg)

Something broke in a system of 15-40 services, and now half of them are red. Find the one that started it.

Chaos engineering tools break services on purpose to see what happens. This is the other side of that: the breaking already happened, you only see the metrics, and you have to work out the cause.

I built it as a small ML challenge: a simulator that makes the incidents, a public/private split, a grader, two simple baselines, a reference solution and rubrics. This is version 2 - version 1 turned out to be too easy for an AI coding agent, more on that below.

Also on Kaggle: the [dataset](https://www.kaggle.com/datasets/netramfaran/first-domino-root-cause-incidents) and a [notebook](https://www.kaggle.com/code/netramfaran/first-domino-finding-the-root-cause) that walks through the baselines, the model with and without the call graph, and where it still fails.

## The task

```
web -> checkout -> payments -> ledger-db
         |
         v
       cart -> cart-cache
```

Say `ledger-db` gets slow. `payments` waits on it and starts timing out, `checkout` waits on `payments`, and `web` shows errors to every user. On a dashboard `web` looks like a disaster and `ledger-db` looks a bit slow. The answer is `ledger-db`.

For each incident you get the call graph (with retries and timeouts), 45 minutes of per-minute metrics for every service that sends them, and a list of deploys. You return your top 3 guesses. Full details are in [problem.md](problem.md).

## Files

- `generate_data.py` - simulates the systems and breaks them (random + seeded, no real data)
- `prepare.py` - splits into `public/` and `private/`
- `grade.py` - scores a submission (MAP@3)
- `baseline.py` - two rules: "loudest service" and "first to look weird"
- `solution.py` - reference model: lines up host clocks, per-service anomaly features, call graph features
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
| baseline: loudest service | 0.282 | 0.306 | 0.272 |
| baseline: first to look weird | 0.309 | 0.388 | 0.275 |
| solution.py --no-graph | 0.607 | 0.835 | 0.510 |
| solution.py | 0.784 | 0.912 | 0.729 |

The test set has 1000 incidents: 300 that look like train, and 700 shifted ones with bigger systems (23-38 services vs 11-21), deeper call chains, third party APIs, more services that send no metrics, more noise, more incidents with two root causes, and two kinds of failure that never show up in train.

With a different seed (`--seed 21`) the reference gets 0.810 and the baselines 0.294 and 0.330, so the gap isn't luck.

## Version 1 vs an AI agent

Once version 1 was done I gave a frontier AI coding agent exactly what a solver gets (`public/` and `problem.md`, no internet, about 30 minutes) and let it try three times on its own.

| | best try | all three | my reference |
|---|---|---|---|
| version 1 | 0.908 | 0.865, 0.669, 0.908 | 0.844 |
| version 2 | 0.789 | 0.722, 0.745, 0.789 | 0.784 |

On version 1 the best try beat my reference in under 4 minutes (two tries also had a broken python setup on my side and had to write everything without numpy or sklearn). It did the same thing I did - anomaly timing plus the call graph - just a bit better. So the problem was mostly "build the obvious features", and the agent is good at that.

What I changed for version 2, each one something that happens in real systems:

- every service's metrics are stamped by its own host and about half the clocks are 1-3 minutes off, so "who went weird first" can't be read off the timestamps anymore
- some services send no metrics at all (third party APIs, some managed databases and caches) and they can still be the root cause
- caches have a database behind them. when a cache dies its callers go straight to the database, which gets flooded and fails a few minutes later - and on a dashboard it looks like the cause

Then I made the reference handle that: line up each host's clock by matching the small traffic wiggles between a service and its callers (they hit both in the same real minute), rank services with no metrics from their callers, look at cpu per request so a traffic spike doesn't look like trouble, and only count trouble that's still there at the end. I didn't look at the agent's code for any of this.

Version 2 numbers: the agent's best try went from 0.908 to 0.789. On the shifted part it got 0.628, 0.666 and 0.726, the reference 0.729. So the best try and the reference are about even overall, and two of the three tries are clearly behind. Version 1 is still in the git history (tag `v1`).

## Why it's hard

- The loudest service is usually a victim. Timeouts and retries turn a small slowdown at the bottom into a wall of errors at the top. Ranking by error rate gets 0.28.
- The broken service can look healthy. In a gray failure the network in front of a service drops requests. Its own metrics barely move (it never sees the lost requests), while everything calling it times out.
- Victims copy the root's symptoms. Waiting callers burn CPU, and services whose health check pings their dependencies get restarted too, so "high CPU" or "missing data" doesn't point at the root by itself.
- Timing alone isn't enough. Going by the timestamps, the root is the first service to look weird in only 21% of test incidents. Noise makes other services look weird earlier, effects often land in the same minute as the cause, clocks are off, and some root causes never look weird at all.
- Lots of noise that isn't the problem: traffic going up and down, a traffic spike that moves every service at once, cron jobs, services that always throw a few errors, deploys that have nothing to do with it.
- Host clocks don't agree, so timestamps lie a bit (version 2).
- The root cause might send no metrics at all. That's 29% of test incidents, 34% of the shifted ones (version 2).
- A dead cache floods the database behind it, and the database ends up looking like the cause (version 2).

## Shortcuts that don't work

- "highest error rate" - 0.28
- "first to look weird" - 0.31. with clocks off, first in the data isn't first in real life
- a good model on per-service features only - 0.84 on normal incidents but 0.51 on shifted ones. Without the call graph it can't tell a victim from a cause, and it can't point at a service with no metrics at all.

## Where the reference still fails

From the shifted part of the test set:

- root cause sends no metrics: 0.58, vs 0.81 when it does
- two root causes at once: 0.59 vs 0.79 for single ones
- a dead cache flooded its database: 0.59 vs 0.79
- traffic spikes: 0.69 vs 0.76
- by failure type: slow services 0.53, small partial failures 0.66, gray failures 0.66

So there's room above 0.73 on the shifted part.

## Rules if you try it

- train from scratch on every run, only on `public/`
- no rules written for specific test incidents
- no external data, standard python ML libs only
- fixed seed, results should reproduce

All data here is simulated. No real systems or companies.
