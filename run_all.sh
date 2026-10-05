#!/bin/bash
# whole thing in one go: data -> split -> baselines -> reference -> scores
# usage: ./run_all.sh [seed]
set -e
SEED=${1:-11}
PY=${PY:-.venv/bin/python}

$PY generate_data.py --seed "$SEED"
$PY prepare.py --seed "$SEED"

for rule in loudest first; do
  $PY baseline.py public "submission_$rule.csv" --rule "$rule" > /dev/null
  echo "== baseline: $rule"
  $PY grade.py --submission "submission_$rule.csv" --by-split
done

$PY solution.py public submission.csv > /dev/null
echo "== solution.py"
$PY grade.py --submission submission.csv --by-split --by-fault
