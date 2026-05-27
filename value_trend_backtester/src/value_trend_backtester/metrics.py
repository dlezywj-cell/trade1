from __future__ import annotations

import numpy as np
import pandas as pd


def summarize(daily: pd.DataFrame) -> dict[str, float]:
    if daily.empty:
        return {}

    equity = daily["Equity"].astype(float)
    returns = equity.pct_change().dropna()
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    annual_return = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    max_drawdown = (equity / equity.cummax() - 1).min()
    volatility = returns.std() * np.sqrt(252) if not returns.empty else 0.0
    sharpe = annual_return / volatility if volatility else np.nan
    win_rate = (returns > 0).mean() if not returns.empty else np.nan
    exposure = daily["Exposure"].mean()

    return {
        "start_equity": float(equity.iloc[0]),
        "end_equity": float(equity.iloc[-1]),
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "max_drawdown": float(max_drawdown),
        "annual_volatility": float(volatility),
        "sharpe_like": float(sharpe),
        "daily_win_rate": float(win_rate),
        "average_exposure": float(exposure),
    }


def format_summary(summary: dict[str, float]) -> str:
    if not summary:
        return "No summary available."

    percent_keys = {
        "total_return",
        "annual_return",
        "max_drawdown",
        "annual_volatility",
        "daily_win_rate",
        "average_exposure",
    }
    labels = {
        "start_equity": "Start equity",
        "end_equity": "End equity",
        "total_return": "Total return",
        "annual_return": "Annual return",
        "max_drawdown": "Max drawdown",
        "annual_volatility": "Annual volatility",
        "sharpe_like": "Sharpe-like",
        "daily_win_rate": "Daily win rate",
        "average_exposure": "Average exposure",
    }
    lines = []
    for key, value in summary.items():
        label = labels.get(key, key)
        if key in percent_keys:
            lines.append(f"{label}: {value:.2%}")
        elif key == "sharpe_like":
            lines.append(f"{label}: {value:.2f}")
        else:
            lines.append(f"{label}: {value:,.2f}")
    return "\n".join(lines)
