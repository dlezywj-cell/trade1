from __future__ import annotations

import pandas as pd

from value_trend_backtester.strategy import StrategyConfig, run_backtest


def _prices(closes: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=len(closes), freq="B")
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c * 1.01 for c in closes],
            "Low": [c * 0.99 for c in closes],
            "Close": closes,
            "Volume": [1_000_000] * len(closes),
        },
        index=dates,
    )


def test_strategy_adds_in_smooth_uptrend() -> None:
    closes = [100 + i * 0.5 for i in range(80)]
    result = run_backtest(
        _prices(closes),
        trade_start="2024-02-01",
        config=StrategyConfig(initial_cash=100_000, commission_bps=0, slippage_bps=0),
    )

    assert result.daily["Exposure"].max() > 0.8
    assert (result.trades["Side"] == "BUY").any()


def test_strategy_exits_after_trend_break() -> None:
    closes = [100 + i * 0.6 for i in range(45)] + [125, 121, 116, 111, 106, 101, 96, 92, 90, 88]
    result = run_backtest(
        _prices(closes),
        trade_start="2024-02-01",
        config=StrategyConfig(initial_cash=100_000, commission_bps=0, slippage_bps=0),
    )

    assert "SELL" in set(result.trades["Side"])
    assert result.daily["Exposure"].iloc[-1] < 0.1
