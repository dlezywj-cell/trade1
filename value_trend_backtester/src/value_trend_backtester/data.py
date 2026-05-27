from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Literal

import pandas as pd
import yfinance as yf

Market = Literal["auto", "cn", "hk", "us"]


@dataclass(frozen=True)
class ResolvedTicker:
    query: str
    ticker: str
    name: str | None = None
    market: str | None = None
    source: str = "rule"


def normalize_end_date(end: str | None) -> date:
    if end is None or end.strip().lower() in {"", "today", "now"}:
        return date.today()
    return datetime.strptime(end, "%Y-%m-%d").date()


def resolve_ticker(query: str, market: Market = "auto") -> ResolvedTicker:
    """Resolve a code or name to a Yahoo Finance compatible ticker.

    The function is intentionally conservative:
    - explicit tickers are preserved;
    - common A-share suffixes are converted to Yahoo suffixes;
    - Chinese A-share names are resolved through optional AKShare if installed;
    - otherwise Yahoo Finance search is used as a global fallback.
    """

    raw = query.strip()
    upper = raw.upper()
    explicit = _resolve_explicit_code(upper, market)
    if explicit is not None:
        return explicit

    if market in {"auto", "cn"}:
        a_share = _resolve_a_share_name(raw)
        if a_share is not None:
            return a_share

    yahoo = _resolve_with_yahoo_search(raw, market)
    if yahoo is not None:
        return yahoo

    raise ValueError(
        f"Could not resolve '{query}'. Try an explicit ticker such as 600519.SS, "
        "000001.SZ, 0700.HK, or AAPL."
    )


def fetch_price_history(
    ticker: str,
    start: str,
    end: date,
    *,
    include_realtime: bool = True,
    lookback_days: int = 90,
) -> pd.DataFrame:
    start_dt = datetime.strptime(start, "%Y-%m-%d").date()
    download_start = start_dt - timedelta(days=lookback_days)
    download_end = end + timedelta(days=1)

    frame = yf.download(
        ticker,
        start=download_start.isoformat(),
        end=download_end.isoformat(),
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if frame.empty:
        raise ValueError(f"No price data returned for {ticker}.")

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    frame = frame.rename(columns=str.title)
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns from price data: {missing}")

    frame = frame[required].dropna(subset=["Open", "High", "Low", "Close"])
    frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
    frame = frame[~frame.index.duplicated(keep="last")]

    if include_realtime and end == date.today():
        frame = _append_latest_intraday_bar(frame, ticker)

    frame = frame.loc[frame.index.date <= end].copy()
    frame.index.name = "Date"
    return frame


def trim_to_trade_window(frame: pd.DataFrame, start: str, end: date) -> pd.DataFrame:
    start_dt = datetime.strptime(start, "%Y-%m-%d").date()
    return frame.loc[(frame.index.date >= start_dt) & (frame.index.date <= end)].copy()


def load_price_csv(path: str) -> pd.DataFrame:
    """Load OHLCV data from a CSV file.

    Required columns are Date, Open, High, Low, Close, and Volume.
    """

    frame = pd.read_csv(path)
    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"CSV file is missing required columns: {missing}")

    frame["Date"] = pd.to_datetime(frame["Date"]).dt.tz_localize(None).dt.normalize()
    frame = frame.set_index("Date").sort_index()
    frame = frame[required[1:]].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=["Open", "High", "Low", "Close"])
    frame.index.name = "Date"
    return frame


def _resolve_explicit_code(raw: str, market: Market) -> ResolvedTicker | None:
    if re.fullmatch(r"[A-Z][A-Z0-9.\-]+", raw) and not raw.isdigit():
        ticker = raw.replace(".SH", ".SS")
        market_name = _infer_market(ticker)
        return ResolvedTicker(query=raw, ticker=ticker, market=market_name, source="explicit")

    if re.fullmatch(r"\d{6}", raw) and market in {"auto", "cn"}:
        suffix = ".SS" if raw.startswith(("5", "6", "9")) else ".SZ"
        return ResolvedTicker(query=raw, ticker=f"{raw}{suffix}", market="cn", source="rule")

    if raw.isdigit() and market == "hk":
        return ResolvedTicker(
            query=raw,
            ticker=f"{raw.zfill(4)}.HK",
            market="hk",
            source="rule",
        )

    return None


def _resolve_a_share_name(query: str) -> ResolvedTicker | None:
    try:
        import akshare as ak  # type: ignore
    except Exception:
        return None

    try:
        codes = ak.stock_info_a_code_name()
    except Exception:
        return None

    if not {"code", "name"}.issubset(codes.columns):
        return None

    exact = codes[codes["name"].astype(str) == query]
    candidates = exact if not exact.empty else codes[codes["name"].astype(str).str.contains(query, na=False)]
    if candidates.empty:
        return None

    row = candidates.iloc[0]
    code = str(row["code"]).zfill(6)
    suffix = ".SS" if code.startswith(("5", "6", "9")) else ".SZ"
    return ResolvedTicker(
        query=query,
        ticker=f"{code}{suffix}",
        name=str(row["name"]),
        market="cn",
        source="akshare",
    )


def _resolve_with_yahoo_search(query: str, market: Market) -> ResolvedTicker | None:
    try:
        search = yf.Search(query, max_results=10)
        quotes = getattr(search, "quotes", None) or []
    except Exception:
        return None

    preferred = [_market_filter(market)]
    for quote in quotes:
        symbol = quote.get("symbol")
        quote_type = quote.get("quoteType")
        if not symbol or quote_type not in {None, "EQUITY", "ETF"}:
            continue
        if preferred[0] and not preferred[0](symbol):
            continue
        return ResolvedTicker(
            query=query,
            ticker=symbol,
            name=quote.get("shortname") or quote.get("longname"),
            market=_infer_market(symbol),
            source="yahoo_search",
        )
    return None


def _market_filter(market: Market):
    if market == "cn":
        return lambda symbol: symbol.endswith((".SS", ".SZ", ".BJ"))
    if market == "hk":
        return lambda symbol: symbol.endswith(".HK")
    if market == "us":
        return lambda symbol: "." not in symbol
    return None


def _infer_market(ticker: str) -> str:
    if ticker.endswith((".SS", ".SZ", ".BJ")):
        return "cn"
    if ticker.endswith(".HK"):
        return "hk"
    return "us/global"


def _append_latest_intraday_bar(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Append or refresh today's bar with the latest intraday price when available."""

    try:
        intraday = yf.download(
            ticker,
            period="5d",
            interval="1m",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception:
        return frame

    if intraday.empty:
        return frame
    if isinstance(intraday.columns, pd.MultiIndex):
        intraday.columns = intraday.columns.get_level_values(0)

    intraday = intraday.rename(columns=str.title)
    required = ["Open", "High", "Low", "Close", "Volume"]
    if any(col not in intraday.columns for col in required):
        return frame

    intraday.index = pd.to_datetime(intraday.index).tz_localize(None)
    latest_day = intraday.index.max().normalize()
    day_rows = intraday[intraday.index.normalize() == latest_day]
    if day_rows.empty:
        return frame

    bar = pd.DataFrame(
        {
            "Open": [float(day_rows["Open"].iloc[0])],
            "High": [float(day_rows["High"].max())],
            "Low": [float(day_rows["Low"].min())],
            "Close": [float(day_rows["Close"].iloc[-1])],
            "Volume": [float(day_rows["Volume"].sum())],
        },
        index=[latest_day],
    )
    combined = pd.concat([frame, bar])
    return combined[~combined.index.duplicated(keep="last")].sort_index()
