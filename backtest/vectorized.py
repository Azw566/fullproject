"""
backtest/vectorized.py — Vectorized (array-at-once) backtester.

Operates on the full price history simultaneously using pandas/numpy operations.
This is Phase 1's approach: fast to write and run, useful as a correctness
baseline, but structurally unable to prevent look-ahead bias on its own.

"""

import math

import numpy as np
import pandas as pd


def run(
    bars:             pd.DataFrame,
    signals:          pd.Series,
    fee_bps:          float        = 10.0,
    slippage_bps:     float        = 0.0,
    vol_target:       float | None = None,
    vol_lookback:     int          = 20,
    periods_per_year: float        = 252.0,
) -> pd.DataFrame:
    """
    Run a vectorized backtest and return a result DataFrame.

    Parameters
    ----------
    bars : pd.DataFrame
        OHLCV data with a DatetimeIndex. Must have a 'close' column.
    signals : pd.Series
        Raw position signal (1.0 = long, 0.0 = flat, NaN = warmup).
    fee_bps : float
        One-way exchange commission in basis points.
    slippage_bps : float
        Half-spread proxy per side in basis points (0 = disabled).
        Added on top of fee_bps; reported separately in the output.
    vol_target : float | None
        Annualised vol target. When set, position is scaled so realised
        portfolio vol tracks this level. None = no scaling (default).
    vol_lookback : int
        Rolling window (bars) for realised-vol estimate.
    periods_per_year : float
        Annualisation factor (252 for daily bars, 8760 for 1h bars).

    Returns
    -------
    pd.DataFrame with columns:
        position       — actual position held on this bar (after 1-bar lag)
        market_return  — raw close-to-close return (buy-and-hold reference)
        gross_return   — strategy return before costs (position × market_return)
        fee            — exchange commission paid on this bar
        slippage       — half-spread cost paid on this bar
        net_return     — strategy return after all costs
        equity         — cumulative equity curve starting at 1.0

    THE ANTI-LOOK-AHEAD MECHANISM
    ──────────────────────────────
    The single most important line in this file:

        position = signals.shift(1)

    VOL TARGETING
    ─────────────
    When vol_target is set, position is scaled by vol_target / realised_vol
    (capped at 1.0 so we never lever up).  The realised-vol estimate at bar t
    uses only returns through bar t-1, matching the event-driven engine's
    rolling deque which is filled one bar behind on_signal().

    Implementation: lagged_returns = market_return.shift(1), then
    rolling(vol_lookback).std().  The extra shift ensures the rolling window
    at bar t covers returns t-vol_lookback .. t-1, identical to the deque
    contents when on_signal(t) is called in the event-driven engine.
    """
    fee_rate      = fee_bps      / 10_000.0
    slippage_rate = slippage_bps / 10_000.0

    # Shift signal by 1 bar to avoid look-ahead.
    position = signals.shift(1).rename("position")

    # Drop warmup rows where position is NaN.
    mask       = position.notna()
    position   = position[mask].astype(float)
    bars_clean = bars.loc[mask]

    # Close-to-close return.
    market_return = bars_clean["close"].pct_change().rename("market_return")

    # ── Vol targeting ─────────────────────────────────────────────────────────
    # Uses lagged returns so at bar t we only see info through t-1.
    # This matches on_signal(t) in the event-driven engine, where the deque
    # contains returns appended in on_fill() calls for bars 0 .. t-1.
    if vol_target is not None and vol_target > 0.0:
        lagged_returns = market_return.shift(1)
        rolled_vol = (
            lagged_returns.rolling(vol_lookback).std(ddof=1)
            * math.sqrt(periods_per_year)
        )
        # NaN before the lookback window fills → fillna(1.0) = no scaling.
        # vol → 0 gives inf → clip to 1.0 = no scaling up.
        vol_scale = (vol_target / rolled_vol).clip(upper=1.0).fillna(1.0)
        position  = position * vol_scale

    # ── Cost and return series ────────────────────────────────────────────────
    gross_return = (position * market_return).rename("gross_return")

    trade    = position.diff().abs()
    fee      = (trade * fee_rate).rename("fee")
    slippage = (trade * slippage_rate).rename("slippage")

    net_return = (gross_return - fee - slippage).rename("net_return")

    result = pd.DataFrame({
        "position":      position,
        "market_return": market_return,
        "gross_return":  gross_return,
        "fee":           fee,
        "slippage":      slippage,
        "net_return":    net_return,
    })

    # Drop the first row (NaN from pct_change and diff).
    result = result.dropna()

    # Equity curve: cumulative product of (1 + net_return), starting at 1.
    result["equity"] = (1 + result["net_return"]).cumprod()

    return result
