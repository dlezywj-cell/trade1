from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd


def save_report_chart(daily: pd.DataFrame, output_path: Path, title: str) -> None:
    if daily.empty:
        return

    _configure_fonts()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(12, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.4, 1.1]},
    )

    price_ax, equity_ax, exposure_ax = axes
    daily[["Close", "MA5", "MA10", "MA20"]].plot(ax=price_ax, linewidth=1.2)
    price_ax.set_title(title)
    price_ax.set_ylabel("Price")
    price_ax.grid(True, alpha=0.25)

    daily["Equity"].plot(ax=equity_ax, color="#1f77b4", linewidth=1.4)
    equity_ax.set_ylabel("Equity")
    equity_ax.grid(True, alpha=0.25)

    (daily["Exposure"] * 100).plot(ax=exposure_ax, color="#2ca02c", linewidth=1.2)
    exposure_ax.set_ylabel("Exposure %")
    exposure_ax.set_ylim(-5, 105)
    exposure_ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _configure_fonts() -> None:
    candidates = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Heiti SC",
        "Songti SC",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "SimHei",
    ]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False
