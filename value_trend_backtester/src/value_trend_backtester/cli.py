from __future__ import annotations

import argparse
import json
from pathlib import Path

from value_trend_backtester.data import (
    ResolvedTicker,
    fetch_price_history,
    load_price_csv,
    normalize_end_date,
    resolve_ticker,
    trim_to_trade_window,
)
from value_trend_backtester.metrics import format_summary, summarize
from value_trend_backtester.plot import save_report_chart
from value_trend_backtester.strategy import StrategyConfig, run_backtest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="value-trend-backtest",
        description="Backtest a value-investor trend overlay strategy.",
    )
    parser.add_argument("stock", help="Stock name or ticker, e.g. 贵州茅台, 600519, 0700.HK, AAPL")
    parser.add_argument("--start", required=True, help="Start date, YYYY-MM-DD")
    parser.add_argument("--end", default="today", help="End date, YYYY-MM-DD or today")
    parser.add_argument("--market", choices=["auto", "cn", "hk", "us"], default="auto")
    parser.add_argument("--cash", type=float, default=1_000_000)
    parser.add_argument("--base-exposure", type=float, default=0.2)
    parser.add_argument("--add-step", type=float, default=0.2)
    parser.add_argument("--max-exposure", type=float, default=1.0)
    parser.add_argument("--observe-days", type=int, default=3)
    parser.add_argument("--hard-exit-days", type=int, default=6)
    parser.add_argument("--max-price-drawdown", type=float, default=0.12)
    parser.add_argument("--max-equity-drawdown", type=float, default=0.10)
    parser.add_argument("--hard-equity-drawdown", type=float, default=0.15)
    parser.add_argument("--lot-size", type=int, default=1)
    parser.add_argument("--commission-bps", type=float, default=2.0)
    parser.add_argument("--slippage-bps", type=float, default=3.0)
    parser.add_argument("--no-realtime", action="store_true", help="Do not try to refresh today's intraday bar")
    parser.add_argument(
        "--prices-csv",
        help="Use local OHLCV data instead of downloading prices. Columns: Date, Open, High, Low, Close, Volume.",
    )
    parser.add_argument("--output-dir", default="outputs")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    end_date = normalize_end_date(args.end)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.prices_csv:
        try:
            resolved = resolve_ticker(args.stock, args.market)
        except ValueError:
            resolved = ResolvedTicker(
                query=args.stock,
                ticker=args.stock,
                market=args.market,
                source="local_csv",
            )
        prices = load_price_csv(args.prices_csv)
    else:
        resolved = resolve_ticker(args.stock, args.market)
        prices = fetch_price_history(
            resolved.ticker,
            args.start,
            end_date,
            include_realtime=not args.no_realtime,
        )
    trade_prices = trim_to_trade_window(prices, args.start, end_date)
    if trade_prices.empty:
        raise SystemExit("No rows inside the requested trading window.")

    config = StrategyConfig(
        initial_cash=args.cash,
        base_exposure=args.base_exposure,
        add_step=args.add_step,
        max_exposure=args.max_exposure,
        observe_days_below_ma10=args.observe_days,
        hard_exit_days_below_ma10=args.hard_exit_days,
        max_price_drawdown=args.max_price_drawdown,
        max_equity_drawdown=args.max_equity_drawdown,
        hard_equity_drawdown=args.hard_equity_drawdown,
        lot_size=args.lot_size,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
    )
    result = run_backtest(prices, trade_start=args.start, config=config)
    summary = summarize(result.daily)

    safe_ticker = resolved.ticker.replace(".", "_").replace("/", "_")
    daily_path = output_dir / f"{safe_ticker}_daily.csv"
    trades_path = output_dir / f"{safe_ticker}_trades.csv"
    summary_path = output_dir / f"{safe_ticker}_summary.json"
    chart_path = output_dir / f"{safe_ticker}_chart.png"

    result.daily.to_csv(daily_path, encoding="utf-8-sig")
    result.trades.to_csv(trades_path, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    save_report_chart(result.daily, chart_path, f"{resolved.ticker} Value Trend Backtest")

    print(f"Resolved: {args.stock} -> {resolved.ticker} ({resolved.source})")
    if resolved.name:
        print(f"Name: {resolved.name}")
    print(format_summary(summary))
    print(f"Daily file: {daily_path}")
    print(f"Trades file: {trades_path}")
    print(f"Summary file: {summary_path}")
    print(f"Chart file: {chart_path}")


if __name__ == "__main__":
    main()
