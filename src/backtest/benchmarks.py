"""
Phase 3 benchmark strategies. All four implement the same Strategy
interface as any future predictive model, and all four get
eligible_universe from build_eligible_universe() rather than querying
index_membership or prices to determine eligibility themselves.
"""
import random
from src.backtest.execution import Strategy


class BuyAndHold(Strategy):
    """Equal-weight the eligible universe once at the first opportunity,
    then hold -- genuinely never rebalances again.

    Returning the same weights dict on every later call would not be
    equivalent: engine.run_backtest() calls rebalance_to() on whatever
    generate_signal() returns every period, so repeating the initial
    weights would nudge the portfolio back toward them each month as
    prices drift, silently turning this into monthly constant-weight
    rebalancing. (Caught this in an early validation run -- buy_and_hold
    showed ~39,600 trades over 108 monthly dates, which is a full-book
    rebalance every month, not a handful of initial trades.) Returning {}
    after the initial buy avoids that -- the engine treats a falsy signal
    as "no rebalance this period"."""

    name = "buy_and_hold"

    def __init__(self):
        self._initial_weights = None
        self._bought = False

    def generate_signal(self, data_access, as_of_date, eligible_universe):
        if self._bought:
            return {}  # already bought -- hold forever, no further trading
        if not eligible_universe:
            return {}  # no eligible universe yet -- keep retrying on later dates
        w = 1.0 / len(eligible_universe)
        self._initial_weights = {sec_id: w for sec_id in eligible_universe}
        self._bought = True
        return dict(self._initial_weights)


class EqualWeightSP500(Strategy):
    """Rebalance to equal weight across the full eligible universe at
    every call. The closest achievable proxy to "the index" without
    market-cap data -- not equivalent to the true cap-weighted S&P 500
    return."""

    name = "equal_weight_sp500"

    def generate_signal(self, data_access, as_of_date, eligible_universe):
        if not eligible_universe:
            return {}
        w = 1.0 / len(eligible_universe)
        return {sec_id: w for sec_id in eligible_universe}


class _TopBottomMomentum(Strategy):
    """Shared implementation for the top/bottom momentum benchmarks --
    ranks eligible_universe by trailing total-return momentum and
    equal-weights the top or bottom N. lookback_days is fixed; don't
    retune it just to make a future predictive model look better by
    comparison."""

    def __init__(self, n, lookback_days, direction):
        self.n = n
        self.lookback_days = lookback_days
        self.direction = direction  # "top" or "bottom"

    def generate_signal(self, data_access, as_of_date, eligible_universe):
        scored = []
        for sec_id in eligible_universe:
            r = data_access.trailing_return(sec_id, self.lookback_days)
            if r is not None:
                scored.append((sec_id, r))
        if not scored:
            return {}
        scored.sort(key=lambda x: x[1], reverse=(self.direction == "top"))
        selected = [sec_id for sec_id, _ in scored[: self.n]]
        if not selected:
            return {}
        w = 1.0 / len(selected)
        return {sec_id: w for sec_id in selected}


class TopNMomentum(_TopBottomMomentum):
    name = "top_n_momentum"

    def __init__(self, n, lookback_days):
        super().__init__(n, lookback_days, "top")


class BottomNMomentum(_TopBottomMomentum):
    name = "bottom_n_momentum"

    def __init__(self, n, lookback_days):
        super().__init__(n, lookback_days, "bottom")


class RandomSelection(Strategy):
    """A single seed is only meaningful as one draw from a distribution,
    so this is the per-seed unit meant to be run many times and
    aggregated. It should use the same eligible_universe, portfolio size,
    execution, and cost parameters as whatever it's being compared
    against."""

    name = "random_selection"

    def __init__(self, portfolio_size, seed):
        self.portfolio_size = portfolio_size
        self.seed = seed
        self._rng = random.Random(seed)

    def generate_signal(self, data_access, as_of_date, eligible_universe):
        if not eligible_universe:
            return {}
        n = min(self.portfolio_size, len(eligible_universe))
        selected = self._rng.sample(eligible_universe, n)
        w = 1.0 / len(selected)
        return {sec_id: w for sec_id in selected}
