import numpy as np

from solution import best_lag


def test_best_lag_finds_a_clock_that_is_off():
    base = np.random.default_rng(0).normal(size=50)
    caller = base[3:47]
    callee = base[1:45]  # callee's minute m is the caller's minute m - 2
    assert best_lag(caller, callee)[0] == -2


def test_best_lag_no_shift():
    x = np.random.default_rng(1).normal(size=44)
    assert best_lag(x, x)[0] == 0
