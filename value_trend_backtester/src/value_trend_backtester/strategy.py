from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class Action(str, Enum):
    BUY = "BUY"
    ADD = "ADD"
    HOLD = "HOLD"
    TRIM = "TRIM"
    SELL = "SELL"


@dataclass(frozen=True)
class StrategyConfig:
    initial_cash: float = 1_000_000
    base_exposure: float = 0.2
    add_step: float = 0.2
    max_exposure: float = 1.0
    defensive_exposure: float = 0.2
    observe_days_below_ma10: int = 3
    hard_exit_days_below_ma10: int = 6
    max_price_drawdown: float = 0.12
    warning_price_drawdown: float = 0.08
    max_equity_drawdown: float = 0.10
    hard_equity_drawdown: float = 0.15
    atr_stop_multiple: float = 2.5
    lot_size: int = 1
    commission_bps: float = 2.0
    slippage_bps: float = 3.0


@dataclass(frozen=True)
class BacktestResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    config: StrategyConfig


def add_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices.copy()
    frame["MA5"] = frame["Close"].rolling(5).mean()
    frame["MA10"] = frame["Close"].rolling(10).mean()
    frame["MA20"] = frame["Close"].rolling(20).mean()
    frame["MA5Slope"] = frame["MA5"].diff()
    frame["MA10Slope"] = frame["MA10"].diff()
    high_low = frame["High"] - frame["Low"]
    high_close = (frame["High"] - frame["Close"].shift(1)).abs()
    low_close = (frame["Low"] - frame["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    frame["ATR14"] = true_range.rolling(14).mean()
    frame["RollingHigh20"] = frame["Close"].rolling(20).max()
    frame["RollingLow20"] = frame["Close"].rolling(20).min()
    frame["BelowMA10"] = frame["Close"] < frame["MA10"]
    frame["DaysBelowMA10"] = _consecutive_true(frame["BelowMA10"])
    return frame


def run_backtest(
    prices_with_lookback: pd.DataFrame,
    *,
    trade_start: str,
    config: StrategyConfig | None = None,
) -> BacktestResult:
    cfg = config or StrategyConfig()
    frame = add_indicators(prices_with_lookback)
    trade_start_ts = pd.Timestamp(trade_start)

    cash = float(cfg.initial_cash)
    shares = 0.0
    pending_target = 0.0
    highest_close_since_entry: float | None = None
    highest_equity = cfg.initial_cash

    rows: list[dict] = []
    trades: list[dict] = []

    for date, row in frame.iterrows():
        if date < trade_start_ts:
            continue

        open_price = float(row["Open"])
        close_price = float(row["Close"])

        if np.isfinite(open_price) and pending_target >= 0:
            cash, shares, trade = _rebalance_at_open(
                date=date,
                cash=cash,
                shares=shares,
                open_price=open_price,
                target_exposure=pending_target,
                cfg=cfg,
            )
            if trade is not None:
                trades.append(trade)

        equity = cash + shares * close_price
        highest_equity = max(highest_equity, equity)
        equity_drawdown = equity / highest_equity - 1 if highest_equity else 0.0

        if shares > 0:
            highest_close_since_entry = close_price if highest_close_since_entry is None else max(
                highest_close_since_entry,
                close_price,
            )
        else:
            highest_close_since_entry = None

        current_exposure = 0.0 if equity <= 0 else shares * close_price / equity
        target_exposure, action, reason = _decide_next_target(
            row=row,
            current_exposure=current_exposure,
            highest_close_since_entry=highest_close_since_entry,
            equity_drawdown=equity_drawdown,
            cfg=cfg,
        )
        pending_target = target_exposure

        rows.append(
            {
                "Date": date,
                "Open": open_price,
                "High": float(row["High"]),
                "Low": float(row["Low"]),
                "Close": close_price,
                "Volume": float(row["Volume"]),
                "MA5": row["MA5"],
                "MA10": row["MA10"],
                "MA20": row["MA20"],
                "ATR14": row["ATR14"],
                "Cash": cash,
                "Shares": shares,
                "Equity": equity,
                "Exposure": current_exposure,
                "TargetExposureNextOpen": target_exposure,
                "Action": action.value,
                "Reason": reason,
                "EquityDrawdown": equity_drawdown,
            }
        )

    daily = pd.DataFrame(rows).set_index("Date")
    trades_frame = pd.DataFrame(trades)
    if not trades_frame.empty:
        trades_frame = trades_frame.set_index("Date")
    return BacktestResult(daily=daily, trades=trades_frame, config=cfg)


def _decide_next_target(
    *,
    row: pd.Series,
    current_exposure: float,
    highest_close_since_entry: float | None,
    equity_drawdown: float,
    cfg: StrategyConfig,
) -> tuple[float, Action, str]:
    close = float(row["Close"])
    ma5 = float(row["MA5"]) if pd.notna(row["MA5"]) else np.nan
    ma10 = float(row["MA10"]) if pd.notna(row["MA10"]) else np.nan
    ma20 = float(row["MA20"]) if pd.notna(row["MA20"]) else np.nan
    atr14 = float(row["ATR14"]) if pd.notna(row["ATR14"]) else np.nan
    days_below_ma10 = int(row["DaysBelowMA10"]) if pd.notna(row["DaysBelowMA10"]) else 0
    ma5_slope = float(row["MA5Slope"]) if pd.notna(row["MA5Slope"]) else np.nan

    if any(np.isnan(value) for value in [ma5, ma10, ma20]):
        return 0.0, Action.HOLD, "waiting for moving-average warmup"

    price_drawdown = 0.0
    if highest_close_since_entry:
        price_drawdown = close / highest_close_since_entry - 1

    atr_stop_hit = False
    if highest_close_since_entry and np.isfinite(atr14):
        atr_stop_hit = close < highest_close_since_entry - cfg.atr_stop_multiple * atr14

    if equity_drawdown <= -cfg.hard_equity_drawdown:
        return 0.0, Action.SELL, "hard portfolio drawdown stop"

    if (
        days_below_ma10 >= cfg.hard_exit_days_below_ma10
        or close < ma20
        or price_drawdown <= -cfg.max_price_drawdown
        or atr_stop_hit
    ):
        return 0.0, Action.SELL, "trend break or hard stop"

    if equity_drawdown <= -cfg.max_equity_drawdown:
        return min(current_exposure, cfg.defensive_exposure), Action.TRIM, "portfolio drawdown control"

    healthy = close >= ma5 and ma5 >= ma10 and ma5_slope > 0
    constructive = close >= ma10 and ma5 >= ma10 * 0.995
    warning = days_below_ma10 > cfg.observe_days_below_ma10 or price_drawdown <= -cfg.warning_price_drawdown

    if current_exposure <= 0:
        if healthy or constructive:
            return cfg.base_exposure, Action.BUY, "re-entry: price back above short-term trend"
        return 0.0, Action.HOLD, "no position: waiting for trend confirmation"

    if warning:
        target = max(cfg.defensive_exposure, current_exposure - cfg.add_step)
        return target, Action.TRIM, "warning: MA10 break did not recover quickly"

    if healthy:
        target = min(cfg.max_exposure, current_exposure + cfg.add_step)
        action = Action.ADD if target > current_exposure + 1e-6 else Action.HOLD
        return target, action, "healthy MA5 uptrend"

    if constructive:
        return current_exposure, Action.HOLD, "constructive pullback above MA10"

    return max(cfg.defensive_exposure, current_exposure - cfg.add_step), Action.TRIM, "weak short-term structure"


def _rebalance_at_open(
    *,
    date: pd.Timestamp,
    cash: float,
    shares: float,
    open_price: float,
    target_exposure: float,
    cfg: StrategyConfig,
) -> tuple[float, float, dict | None]:
    equity_at_open = cash + shares * open_price
    if equity_at_open <= 0 or open_price <= 0:
        return cash, shares, None

    target_value = equity_at_open * min(max(target_exposure, 0.0), cfg.max_exposure)
    current_value = shares * open_price
    diff_value = target_value - current_value
    if abs(diff_value) < max(10.0, equity_at_open * 0.001):
        return cash, shares, None

    raw_shares = diff_value / open_price
    trade_shares = _round_to_lot(raw_shares, cfg.lot_size)
    if trade_shares == 0:
        return cash, shares, None

    side = "BUY" if trade_shares > 0 else "SELL"
    slippage = cfg.slippage_bps / 10_000
    commission = cfg.commission_bps / 10_000
    execution_price = open_price * (1 + slippage if trade_shares > 0 else 1 - slippage)
    gross = trade_shares * execution_price
    fee = abs(gross) * commission

    if trade_shares > 0:
        affordable = max(cash - fee, 0) / execution_price
        if trade_shares > affordable:
            trade_shares = _round_to_lot(affordable, cfg.lot_size)
            gross = trade_shares * execution_price
            fee = abs(gross) * commission
        if trade_shares <= 0:
            return cash, shares, None
        cash -= gross + fee
    else:
        trade_shares = max(trade_shares, -shares)
        gross = trade_shares * execution_price
        fee = abs(gross) * commission
        cash -= gross
        cash -= fee

    shares += trade_shares
    return (
        cash,
        shares,
        {
            "Date": date,
            "Side": side,
            "Shares": trade_shares,
            "Price": execution_price,
            "GrossValue": gross,
            "Fee": fee,
            "CashAfter": cash,
            "SharesAfter": shares,
            "TargetExposure": target_exposure,
        },
    )


def _round_to_lot(shares: float, lot_size: int) -> float:
    lot = max(int(lot_size), 1)
    if lot == 1:
        return float(shares)
    if shares > 0:
        return float(int(shares // lot) * lot)
    return float(-int(abs(shares) // lot) * lot)


def _consecutive_true(series: pd.Series) -> pd.Series:
    groups = (series != series.shift()).cumsum()
    counts = series.groupby(groups).cumcount() + 1
    return counts.where(series, 0)
