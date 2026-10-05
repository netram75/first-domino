# rubrics

The score alone doesn't say if a solution actually gets the problem. Each of these is pass/fail.

1. **Uses the call graph** - the ranking depends on who calls who, not only on how bad each service looks on its own. Shows what the graph adds (with vs without is enough)
2. **Validates by incident** - any local validation splits by incident, never by row. Rows from one incident on both sides of a split is a fail
3. **Handles the shift** - reports normal vs shifted (or small vs big systems) separately, and the shifted score is at least 0.70
4. **Doesn't fall for the noise** - shows at least one red herring it handles (traffic spikes, cron jobs, always-flaky services, unrelated deploys) with numbers before and after
5. **No leakage** - trains only on `public/`, no rules written for specific test incidents, no service names hardcoded
6. **Honest error analysis** - lists the main failure cases left, with counts, including at least one kind of incident it still gets wrong
