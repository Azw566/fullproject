"""
backtest/event_driven.py — Event-driven backtester.

Processes one bar at a time through a four-event pipeline:

    MarketEvent → SignalEvent → OrderEvent → FillEvent

PUBLIC INTERFACE
────────────────
    run(bars, signals, fee_bps) → pd.DataFrame

"""

from __future__ import annotations

import logging
import math
import statistics
from collections import deque
from dataclasses import dataclass, field

import pandas as pd

_logger = logging.getLogger(__name__)


# ── Events 
@dataclass
class MarketEvent:
    """A new price bar has arrived from the data feed."""
    timestamp: pd.Timestamp
    close: float
    market_return: float  # close-to-close; NaN on the very first bar


@dataclass
class SignalEvent:
    """The strategy's desired target position for this bar."""
    timestamp: pd.Timestamp
    target_position: float  # 1.0 = long, 0.0 = flat


@dataclass
class OrderEvent:
    """Instruction to change position by `delta` units."""
    timestamp: pd.Timestamp
    delta: float  # positive = buy, negative = sell, 0 = hold


@dataclass
class FillEvent:
    """Confirmed execution of an order at the bar's closing price."""
    timestamp: pd.Timestamp
    delta:     float         # position change that was executed
    fee:       float         # exchange fee (fee_bps applied to |delta|)
    slippage:  float = 0.0   # half-spread cost (slippage_bps applied to |delta|)


# ── Handlers 

class _Portfolio:
    """
    Tracks position and equity; converts SignalEvents into OrderEvents.

    Phase 2: trivial pass-through (always trade to the exact target).
    Phase 5: optional vol targeting and drawdown circuit-breaker.

    All risk parameters default to disabled so Phase 2-4 behaviour is
    preserved when called without arguments.
    """

    def __init__(
        self,
        vol_target:       float | None = None,
        vol_lookback:     int          = 20,
        periods_per_year: float        = 252.0,
        max_drawdown:     float | None = None,
        cooldown_bars:    int          = 20,
    ) -> None:
        self.position: float = 0.0
        self.equity:   float = 1.0

        # Vol targeting
        self._vol_target       = vol_target
        self._vol_lookback     = vol_lookback
        self._periods_per_year = periods_per_year
        # Filled with market_return values in on_fill(); used to estimate vol.
        self._market_returns: deque[float] = deque(maxlen=vol_lookback)

        # Circuit breaker 
        self._max_drawdown    = max_drawdown
        self._cooldown_bars   = cooldown_bars
        self._peak_equity     = 1.0
        self._halted          = False
        self._bars_since_halt = 0

    def on_signal(self, event: SignalEvent) -> OrderEvent:
        """
        Convert a strategy signal into an order, applying risk constraints.

        Order of precedence:
          1. Circuit breaker (overrides everything — forces target to 0)
          2. Vol targeting   (scales the surviving target down toward vol_target)
        """
        target = event.target_position

        # Circuit breaker 
        if self._max_drawdown is not None:
            self._peak_equity = max(self._peak_equity, self.equity)
            drawdown = self.equity / self._peak_equity - 1.0

            if not self._halted and drawdown < -self._max_drawdown:
                self._halted          = True
                self._bars_since_halt = 0
                _logger.warning(
                    "Circuit breaker fired: drawdown=%.2f%%  threshold=%.0f%%.",
                    drawdown * 100, self._max_drawdown * 100,
                )
            elif self._halted:
                self._bars_since_halt += 1
                if self._bars_since_halt >= self._cooldown_bars:
                    self._halted = False
                    _logger.info(
                        "Circuit breaker reset after %d bars flat.",
                        self._cooldown_bars,
                    )

            if self._halted:
                target = 0.0

        #  Vol targeting (only when not halted and window is full) 
        if not self._halted and self._vol_target is not None:
            if len(self._market_returns) == self._vol_lookback:
                realized_vol = (
                    statistics.stdev(self._market_returns)
                    * math.sqrt(self._periods_per_year)
                )
                if realized_vol > 0.0:
                    target = target * min(self._vol_target / realized_vol, 1.0)

        return OrderEvent(timestamp=event.timestamp, delta=target - self.position)

    def on_fill(self, event: FillEvent, market_return: float) -> dict:
        """
        Apply the fill, compute returns, and return the row dict for this bar.

        The position changes to the target, then earns the bar's market return.
        Fee and slippage are both subtracted from the return. Equity compounds.
        Market return is appended to the rolling window for vol estimation.
        """
        self.position    += event.delta
        gross_return      = self.position * market_return
        net_return        = gross_return - event.fee - event.slippage
        self.equity      *= 1.0 + net_return
        self._market_returns.append(market_return)
        return {
            "position":      self.position,
            "market_return": market_return,
            "gross_return":  gross_return,
            "fee":           event.fee,
            "slippage":      event.slippage,
            "net_return":    net_return,
            "equity":        self.equity,
        }


class _Broker:
    """
    Simulated execution engine.

    Fills orders immediately at the closing price.  Two independent cost
    components are reported separately so attribution is unambiguous:
      fee      — exchange commission (fee_bps per side)
      slippage — half-spread / market-impact proxy (slippage_bps per side)
    """

    def __init__(self, fee_bps: float, slippage_bps: float = 0.0) -> None:
        self._fee_rate      = fee_bps      / 10_000.0
        self._slippage_rate = slippage_bps / 10_000.0

    def on_order(self, event: OrderEvent) -> FillEvent:
        size = abs(event.delta)
        return FillEvent(
            timestamp=event.timestamp,
            delta=event.delta,
            fee=size * self._fee_rate,
            slippage=size * self._slippage_rate,
        )


# Public run()
def run(
    bars:             pd.DataFrame,
    signals:          pd.Series,
    fee_bps:          float        = 10.0,
    slippage_bps:     float        = 0.0,
    vol_target:       float | None = None,
    vol_lookback:     int          = 20,
    periods_per_year: float        = 252.0,
    max_drawdown:     float | None = None,
    cooldown_bars:    int          = 20,
) -> pd.DataFrame:
    """
    Run an event-driven backtest and return a result DataFrame.

    Parameters
    ----------
    bars, signals, fee_bps : same as vectorized.run() — see that docstring.
    slippage_bps     : half-spread proxy per side (0 = disabled).
    vol_target       : annualised vol target for position scaling (None = disabled).
    vol_lookback     : bars for rolling realised-vol window.
    periods_per_year : annualisation factor (252 for daily, 8760 for 1h).
    max_drawdown     : circuit-breaker trigger level (None = disabled).
    cooldown_bars    : bars flat after circuit fires before attempting re-entry.

    Returns
    -------
    pd.DataFrame with columns: position, market_return, gross_return,
        fee, slippage, net_return, equity.
    """
    # Apply the 1-bar lag: lagged_signals[T] = signals[T-1].
    # The position held on bar T was decided by the signal from bar T-1.
    lagged_signals = signals.shift(1)

    mask           = lagged_signals.notna()
    lagged_signals = lagged_signals[mask].astype(float)
    bars_clean     = bars.loc[mask]

    portfolio = _Portfolio(
        vol_target=vol_target,
        vol_lookback=vol_lookback,
        periods_per_year=periods_per_year,
        max_drawdown=max_drawdown,
        cooldown_bars=cooldown_bars,
    )
    broker = _Broker(fee_bps, slippage_bps)

    rows: list[dict] = []
    prev_close: float | None = None

    for ts, bar in bars_clean.iterrows():
        close = float(bar["close"])

        if prev_close is None:
            # First bar: set the initial portfolio position without charging a fee.
            # No prior position to diff against — mirrors the vectorized engine
            # dropping its first row (NaN from pct_change and diff).
            portfolio.position = float(lagged_signals.loc[ts])
            prev_close = close
            continue

        # ── MarketEvent 
        market_return = close / prev_close - 1.0
        market_evt = MarketEvent(timestamp=ts, close=close, market_return=market_return)

        # ── SignalEvent
        signal_evt = SignalEvent(timestamp=ts, target_position=float(lagged_signals.loc[ts]))

        # ── OrderEvent 
        order_evt = portfolio.on_signal(signal_evt)

        # ── FillEvent 
        fill_evt = broker.on_order(order_evt)

        # ── Portfolio update 
        row = portfolio.on_fill(fill_evt, market_evt.market_return)
        rows.append(row)

        prev_close = close

    return pd.DataFrame(rows, index=bars_clean.index[1:])
