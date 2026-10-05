import pandas as pd
import pytest

from grade import ap_at_3, check

SERVICES = {"inc1": {"web", "api", "db"}, "inc2": {"web", "cache"}}


def sub(rows):
    return pd.DataFrame(rows, columns=["incident_id", "ranking"])


def test_first_guess_right():
    assert ap_at_3(["db", "api", "web"], {"db"}) == 1


def test_second_and_third_guess():
    assert ap_at_3(["api", "db", "web"], {"db"}) == 0.5
    assert ap_at_3(["api", "web", "db"], {"db"}) == pytest.approx(1 / 3)


def test_miss():
    assert ap_at_3(["api", "web"], {"db"}) == 0


def test_two_roots():
    # found in 1st and 3rd place -> (1/1 + 2/3) / 2
    assert ap_at_3(["db", "api", "cache"], {"db", "cache"}) == pytest.approx((1 + 2 / 3) / 2)


def test_good_submission_passes():
    check(sub([["inc1", "db api web"], ["inc2", "cache"]]), SERVICES)


@pytest.mark.parametrize("rows", [
    [["inc1", "db"]],                                      # inc2 missing
    [["inc1", "db"], ["inc2", "cache"], ["inc9", "web"]],  # unknown incident
    [["inc1", "db"], ["inc1", "web"], ["inc2", "cache"]],  # same incident twice
    [["inc1", "db api web db"], ["inc2", "cache"]],        # more than 3
    [["inc1", "db db"], ["inc2", "cache"]],                # same name twice
    [["inc1", "orders-db"], ["inc2", "cache"]],            # not in that incident
])
def test_bad_submissions_get_rejected(rows):
    with pytest.raises(SystemExit):
        check(sub(rows), SERVICES)


def test_wrong_columns():
    with pytest.raises(SystemExit):
        check(pd.DataFrame({"incident_id": ["inc1", "inc2"], "guess": ["db", "web"]}), SERVICES)
