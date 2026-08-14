"""
Phase 3 portfolio accounting. Tracks cash, positions, weights, shares,
portfolio value, realized/unrealized P&L, dividends, transaction costs,
and turnover.

Position values are computed from the `total_return` price series, which
already reflects dividends as if reinvested into the same security. So
portfolio_value compounds dividend income through price appreciation on
its own; crediting dividends to cash as well would double-count them.
The `dividends_received` field is therefore informational/attribution
only (pulled from corporate_actions for reporting), not a second
cash-flow mechanism.
"""
from src.backtest.costs import trade_cost


class Portfolio:
    def __init__(self, starting_cash, cost_config, accounting_tolerance=1e-6):
        self.cash = starting_cash
        self.starting_cash = starting_cash
        self.positions = {}  # security_id -> shares
        self.cost_config = cost_config
        self.accounting_tolerance = accounting_tolerance
        self.realized_pnl = 0.0
        self.total_costs_paid = 0.0
        self.total_dividends_received = 0.0
        self.trade_count = 0
        self.history = []  # list of snapshot dicts, one per rebalance

    def _price(self, conn, security_id, as_of_date, adj_type="total_return"):
        """Last known price on or before as_of_date, used for valuation
        (marking a holding to market, or closing out a position being
        fully exited). Falls back to stale prices on purpose -- that's
        fine for valuing an existing position, but don't use this to
        decide whether a new trade can be placed; see _price_exact."""
        row = conn.execute(
            "SELECT close FROM prices WHERE security_id=? AND adj_type=? AND date<=? ORDER BY date DESC LIMIT 1",
            (security_id, adj_type, as_of_date),
        ).fetchone()
        return row["close"] if row else None

    def _price_exact(self, conn, security_id, as_of_date, adj_type="total_return"):
        """Price on exactly as_of_date, no fallback to an earlier date.
        Used to decide whether a security can trade on the execution date
        resolved by execution.next_market_session(): if it has no price
        on that exact session it simply sits out this rebalance rather
        than trading at a stale substitute price."""
        row = conn.execute(
            "SELECT close FROM prices WHERE security_id=? AND adj_type=? AND date=?",
            (security_id, adj_type, as_of_date),
        ).fetchone()
        return row["close"] if row else None

    def _dividends_between(self, conn, security_id, start_date, end_date, shares_held):
        rows = conn.execute(
            """SELECT ratio_or_value FROM corporate_actions
               WHERE security_id=? AND action_type IN ('dividend','special_dividend')
               AND ex_date > ? AND ex_date <= ?""",
            (security_id, start_date, end_date),
        ).fetchall()
        return sum((r["ratio_or_value"] or 0) * shares_held for r in rows)

    def portfolio_value(self, conn, as_of_date, adj_type="total_return"):
        total = self.cash
        for sec_id, shares in self.positions.items():
            price = self._price(conn, sec_id, as_of_date, adj_type)
            if price is not None:
                total += shares * price
        return total

    def rebalance_to(self, conn, as_of_date, target_weights, adj_type="total_return"):
        """target_weights: dict[security_id, weight], weight fraction of
        portfolio_value, remainder held as cash. Executes trades at
        as_of_date's price -- the caller passes the execution date (the
        next market session, resolved by execution.py), this function
        doesn't figure out timing itself.

        A security named in target_weights only trades if it has a price
        exactly on as_of_date (_price_exact); otherwise its trade is
        skipped this period, its existing position is left untouched, and
        it's reconsidered at the next rebalance. This keeps one security's
        missing data from affecting anyone else's trade or from shifting
        the execution date for the whole rebalance (an earlier version of
        this engine picked the execution date from one security's price
        availability, which let a single data gap delay the entire
        portfolio by months -- see execution.next_market_session). A
        security that's simply absent from target_weights (a deliberate
        exit) is still force-closed at the last known price via _price."""
        self._assert_chronological_order(as_of_date)
        pv_before = self.portfolio_value(conn, as_of_date, adj_type)
        turnover = 0.0

        target_shares = {}
        skipped_no_execution_price = []
        for sec_id, w in target_weights.items():
            price = self._price_exact(conn, sec_id, as_of_date, adj_type)
            if price is None or price <= 0:
                skipped_no_execution_price.append(sec_id)
                continue
            target_shares[sec_id] = (w * pv_before) / price

        # Securities skipped above (no price on this exact execution date)
        # are excluded from all_secs entirely, not treated as target=0 --
        # that would force-sell them using a substitute price. What's left
        # is (a) securities we're targeting with a resolvable price and
        # (b) currently-held securities the strategy is genuinely exiting.
        all_secs = (set(self.positions.keys()) | set(target_weights.keys())) - set(skipped_no_execution_price)
        for sec_id in all_secs:
            current = self.positions.get(sec_id, 0.0)
            target = target_shares.get(sec_id, 0.0)
            delta_shares = target - current
            if abs(delta_shares) < 1e-9:
                continue
            price = self._price(conn, sec_id, as_of_date, adj_type)
            if price is None:
                # security has no current price (e.g. delisted) -- force-close
                # at last known price if we're holding it, otherwise skip
                if current != 0:
                    last_known = self._price(conn, sec_id, as_of_date, adj_type) or 0.0
                    proceeds = current * last_known
                    self.cash += proceeds
                    self.positions.pop(sec_id, None)
                continue

            trade_value = abs(delta_shares) * price
            is_sell = delta_shares < 0
            cost = trade_cost(trade_value, is_sell, self.cost_config)
            self.total_costs_paid += cost
            self.cash += -delta_shares * price  # buy: cash decreases; sell: cash increases
            self.cash -= cost
            turnover += trade_value
            self.trade_count += 1

            new_shares = current + delta_shares
            if abs(new_shares) < 1e-9:
                self.positions.pop(sec_id, None)
            else:
                self.positions[sec_id] = new_shares

        pv_after = self.portfolio_value(conn, as_of_date, adj_type)

        self.history.append({
            "as_of_date": as_of_date,
            "cash": self.cash,
            "positions": dict(self.positions),
            "portfolio_value": pv_after,
            "turnover": turnover,
            "turnover_pct": (turnover / pv_before) if pv_before else 0.0,
            "skipped_no_execution_price": list(skipped_no_execution_price),
        })
        self._assert_invariant(conn, as_of_date, adj_type)
        return pv_after

    def mark_to_market(self, conn, as_of_date, adj_type="total_return"):
        """Record a portfolio-value snapshot at as_of_date without trading.

        rebalance_to() is the only other thing that appends to
        self.history, so a strategy that legitimately skips a period
        (e.g. BuyAndHold after its initial purchase) would otherwise leave
        a gap in the value series -- in the worst case, a strategy that
        trades exactly once ends up with a single history point and a
        trivial 0% reported return. Call this on every rebalance date the
        engine visits, traded or not, so held positions get marked to
        market on the same cadence."""
        self._assert_chronological_order(as_of_date)
        pv = self.portfolio_value(conn, as_of_date, adj_type)
        self.history.append({
            "as_of_date": as_of_date,
            "cash": self.cash,
            "positions": dict(self.positions),
            "portfolio_value": pv,
            "turnover": 0.0,
            "turnover_pct": 0.0,
        })
        self._assert_invariant(conn, as_of_date, adj_type)
        return pv

    def _assert_chronological_order(self, as_of_date):
        """Every history entry, trade or mark-to-market, must land in
        non-decreasing as_of_date order. An old bug picked the execution
        date from one probe security's own price history, so a signal
        from an earlier month could resolve to a later execution date
        than a subsequent month's trade -- since the engine processes
        signals in signal-date order, that let a later trade mutate
        cash/positions before an earlier one was applied, corrupting
        state rather than just mistiming a value. next_market_session()
        now bounds execution to the very next session so this shouldn't
        trigger in practice; the check stays as a safety net."""
        if self.history and as_of_date < self.history[-1]["as_of_date"]:
            raise AssertionError(
                f"Chronological order violated: attempted to apply portfolio state at "
                f"as_of_date={as_of_date}, which is before the most recent recorded "
                f"as_of_date={self.history[-1]['as_of_date']}."
            )

    def _assert_invariant(self, conn, as_of_date, adj_type="total_return"):
        computed = self.cash + sum(
            (self._price(conn, sec_id, as_of_date, adj_type) or 0) * shares
            for sec_id, shares in self.positions.items()
        )
        reported = self.portfolio_value(conn, as_of_date, adj_type)
        rel_diff = abs(computed - reported) / max(abs(reported), 1e-9)
        if rel_diff > self.accounting_tolerance:
            raise AssertionError(
                f"Portfolio accounting invariant violated at {as_of_date}: "
                f"sum(positions)+cash={computed} != portfolio_value={reported} "
                f"(relative diff {rel_diff})"
            )
