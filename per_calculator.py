#!/usr/bin/env python3
"""
PER Calculator (Monthly using Quarterly EPS) — v3.2 (with plotting)
-------------------------------------------------------------------
- Robust datetime handling
- P/E (quarterly EPS, annualized x4) + P/E (TTM)
- 5-period moving averages: price_ma5, pe_ma5, pe_ttm_ma5
- Plotting support (--plot, --savefig)

Install:
    pip install yfinance pandas numpy matplotlib
"""

from __future__ import annotations
import argparse
from typing import Optional
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# ---------------- Utilities ----------------

def to_dt_index(idx) -> pd.DatetimeIndex:
    """Coerce any index/array-like to tz-naive DatetimeIndex; invalid -> NaT."""
    if isinstance(idx, pd.PeriodIndex):
        dt = idx.to_timestamp(how='end')
    else:
        dt = pd.to_datetime(idx, errors='coerce')
    dt = pd.DatetimeIndex(dt)
    try:
        dt = dt.tz_localize(None)
    except Exception:
        pass
    return dt


# ---------------- Prices ----------------

def fetch_daily_prices(ticker: str, start: Optional[str], end: Optional[str], years: Optional[int], debug: bool=False) -> pd.DataFrame:
    if years is not None and (start or end):
        raise ValueError("Provide either --years OR (--start and --end), not both.")
    if years is not None:
        df = yf.download(ticker, period=f"{years}y", interval="1d", auto_adjust=False, progress=False)
    else:
        if not start:
            raise ValueError("When --years is not provided, --start must be given.")
        df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError("No price data returned. Check ticker or date range.")
    df = df.copy()
    df.index = to_dt_index(df.index)
    df = df[~df.index.isna()]
    if debug:
        print("Prices fetched:", df.shape, type(df.index), df.index.dtype)
    return df


def monthly_avg_close(daily: pd.DataFrame, debug: bool=False) -> pd.Series:
    daily = daily.copy()
    if not isinstance(daily.index, pd.DatetimeIndex):
        daily.index = to_dt_index(daily.index)
    daily = daily[~daily.index.isna()]
    m = daily['Close'].resample('ME').mean()
    if isinstance(m, pd.DataFrame):
        m = m.squeeze("columns")
    if not isinstance(m, pd.Series):
        m = pd.Series(np.asarray(m).ravel(), index=daily['Close'].resample('ME').mean().index)
    m.name = "monthly_avg_close"
    if debug:
        print("Monthly avg computed:", type(m), m.shape)
    return m


# ---------------- EPS (longer history merge) ----------------

EPS_LABELS = [
    'Diluted EPS', 'Basic EPS', 'BasicEPS', 'DilutedEPS',
    'Diluted EPS (USD)', 'Basic EPS (USD)'
]

def _eps_from_earnings_dates(t: yf.Ticker, limit: int = 60, debug: bool = False) -> Optional[pd.Series]:
    """
    Use earnings dates (EPS Actual) to get a longer EPS series by announcement date.
    """
    try:
        try:
            ed = t.get_earnings_dates(limit=limit)
        except AttributeError:
            ed = getattr(t, "earnings_dates", None)
        if ed is None or ed.empty:
            return None
        df = ed.copy()
        date_col = None
        for c in ["Earnings Date", "EarningsDate", "Date", "earningsDate"]:
            if c in df.columns:
                date_col = c
                break
        eps_col = None
        for c in ["EPS Actual", "EPS", "epsActual", "EpsActual"]:
            if c in df.columns:
                eps_col = c
                break
        if date_col is None or eps_col is None:
            if debug: print("earnings_dates: required columns not found")
            return None
        s = df[[date_col, eps_col]].dropna()
        s[date_col] = to_dt_index(s[date_col])
        s = s.dropna(subset=[date_col]).set_index(date_col)[eps_col].astype(float)
        s.index = to_dt_index(s.index)
        s = s[~s.index.isna()].sort_index()
        s = s[~s.index.duplicated(keep="last")]
        if debug:
            print(f"earnings_dates EPS pulled: {len(s)} points, {s.index.min()} → {s.index.max()}")
        return s
    except Exception as e:
        if debug: print("eps_from_earnings_dates error:", e)
        return None


def get_quarterly_eps_series(ticker: str, debug: bool=False) -> pd.Series:
    """
    Merge multiple sources to get a longer quarterly EPS series:
      1) quarterly_financials EPS (most accurate; short)
      2) quarterly_income_stmt EPS (if available)
      3) earnings dates EPS Actual (long history; announcement date)
      4) fallback: quarterly_earnings['Earnings'] / shares_outstanding (approx)
    Priority: (1)>(2)>(3)>(4)
    """
    t = yf.Ticker(ticker)

    eps_fin = None
    try:
        qf = t.quarterly_financials
        if qf is not None and not qf.empty:
            qf = qf.copy()
            qf.columns = to_dt_index(qf.columns)
            qf = qf.loc[:, ~qf.columns.isna()]
            for lab in EPS_LABELS:
                if lab in qf.index:
                    eps_fin = qf.loc[lab].dropna()
                    eps_fin.index = to_dt_index(eps_fin.index)
                    eps_fin = eps_fin[~eps_fin.index.isna()].astype(float).sort_index()
                    break
    except Exception as e:
        if debug: print("quarterly_financials error:", e)

    eps_is = None
    try:
        qis = getattr(t, "quarterly_income_stmt", None)
        if qis is not None and not qis.empty:
            qis = qis.copy()
            qis.columns = to_dt_index(qis.columns)
            qis = qis.loc[:, ~qis.columns.isna()]
            for lab in EPS_LABELS:
                if lab in qis.index:
                    eps_is = qis.loc[lab].dropna()
                    eps_is.index = to_dt_index(eps_is.index)
                    eps_is = eps_is[~eps_is.index.isna()].astype(float).sort_index()
                    break
    except Exception as e:
        if debug: print("quarterly_income_stmt error:", e)

    eps_ed = _eps_from_earnings_dates(t, limit=60, debug=debug)

    eps_est = None
    try:
        qe = t.quarterly_earnings
        if qe is not None and not qe.empty:
            shares = None
            try:
                info = t.fast_info or {}
                shares = info.get('shares_outstanding', None)
            except Exception:
                shares = None
            if not shares:
                try:
                    info = t.info or {}
                    shares = info.get('sharesOutstanding', None)
                except Exception:
                    shares = None
            if shares and shares > 0:
                eps_est = (qe['Earnings'] / shares).dropna()
                eps_est.index = to_dt_index(eps_est.index)
                eps_est = eps_est[~eps_est.index.isna()].astype(float).sort_index()
    except Exception as e:
        if debug: print("quarterly_earnings fallback error:", e)

    s = None
    for cand in [eps_fin, eps_is, eps_ed, eps_est]:
        if cand is not None and not cand.empty:
            if s is None:
                s = cand.copy()
            else:
                s = s.combine_first(cand)

    if s is None or s.empty:
        raise RuntimeError("No quarterly EPS available from yfinance for this ticker.")

    s = s.sort_index()
    if debug: print("Merged EPS series:", len(s), "points,", s.index.min(), "→", s.index.max())
    return s


# ---------------- P/E compute ----------------

def compute_monthly_pe(monthly_price, eps_quarterly: pd.Series, annualize: bool, debug: bool=False) -> pd.DataFrame:
    """
    Returns monthly DataFrame with:
      monthly_avg_close, eps_quarter, eps_basis, pe, quarter_ref,
      eps_ttm, pe_ttm, price_ma5, pe_ma5, pe_ttm_ma5
    """
    if isinstance(monthly_price, pd.DataFrame):
        if 'monthly_avg_close' in monthly_price.columns and monthly_price.shape[1] == 1:
            monthly_price = monthly_price['monthly_avg_close']
        else:
            monthly_price = monthly_price.squeeze("columns")
    if not isinstance(monthly_price, pd.Series):
        monthly_price = pd.Series(np.asarray(monthly_price).ravel(), index=getattr(monthly_price, 'index', None))
    monthly_price.index = to_dt_index(monthly_price.index)
    monthly_price = monthly_price[~monthly_price.index.isna()].sort_index()
    monthly_price.name = "monthly_avg_close"

    m = monthly_price.to_frame('monthly_avg_close').reset_index().rename(columns={'index': 'date', monthly_price.index.name or 'index': 'date'})
    if 'date' not in m.columns:
        m = m.rename(columns={m.columns[0]: 'date'})
    m['date'] = to_dt_index(m['date'])
    m = m.dropna(subset=['date']).sort_values('date')

    q = eps_quarterly.copy().to_frame('eps_quarter')
    q.index = to_dt_index(q.index)
    q = q[~q.index.isna()].sort_index()
    q = q.reset_index().rename(columns={'index': 'quarter_ref'})
    if 'quarter_ref' not in q.columns:
        q = q.rename(columns={q.columns[0]: 'quarter_ref'})
    q['quarter_ref'] = to_dt_index(q['quarter_ref'])
    q = q.dropna(subset=['quarter_ref']).sort_values('quarter_ref')

    if debug:
        print("Monthly df head:\n", m.head())
        print("Quarterly EPS df head:\n", q.head())

    aligned = pd.merge_asof(
        m, q,
        left_on='date', right_on='quarter_ref',
        direction='backward',
        allow_exact_matches=True
    )

    factor = 1.0 if not annualize else 4.0
    aligned['eps_basis'] = aligned['eps_quarter'] * factor

    aligned['pe'] = np.where(
        (aligned['eps_basis'].notna()) & (aligned['eps_basis'] > 0),
        aligned['monthly_avg_close'] / aligned['eps_basis'],
        np.nan
    )

    # TTM EPS/P-E
    q_ttm = eps_quarterly.sort_index().rolling(window=4).sum().dropna()
    qttm = q_ttm.to_frame('eps_ttm').reset_index().rename(columns={'index': 'quarter_ref'})
    qttm['quarter_ref'] = to_dt_index(qttm['quarter_ref'])

    aligned_ttm = pd.merge_asof(
        aligned[['date', 'monthly_avg_close', 'quarter_ref']],
        qttm.sort_values('quarter_ref'),
        left_on='date', right_on='quarter_ref',
        direction='backward', allow_exact_matches=True
    )

    out = aligned.set_index('date')[['monthly_avg_close', 'eps_quarter', 'eps_basis', 'pe', 'quarter_ref']]
    out = out.join(aligned_ttm.set_index('date')[['eps_ttm']], how='left')
    out['pe_ttm'] = np.where(
        (out['eps_ttm'].notna()) & (out['eps_ttm'] > 0),
        out['monthly_avg_close'] / out['eps_ttm'],
        np.nan
    )

    # 5-period moving averages
    out = out.sort_index()
    out['price_ma5']  = out['monthly_avg_close'].rolling(window=5, min_periods=1).mean()
    out['pe_ma5']     = out['pe'].rolling(window=5, min_periods=1).mean()
    out['pe_ttm_ma5'] = out['pe_ttm'].rolling(window=5, min_periods=1).mean()

    if debug:
        print("Output shape:", out.shape)
        print(out.tail(5))

    return out


# ---------------- Plotting ----------------

def plot_monthly(res: pd.DataFrame, ticker: str, savefig: Optional[str] = None):
    """
    Draw 4 lines as requested:
      - Left y-axis: Price (monthly_avg_close), Price MA5
      - Right y-axis: P/E (pe), P/E TTM (pe_ttm)
    """
    res = res.sort_index()
    fig, ax_price = plt.subplots(figsize=(12, 6))

    # Left axis: Price
    ax_price.plot(res.index, res['monthly_avg_close'], label="Price")
    ax_price.plot(res.index, res['price_ma5'], linestyle="--", label="Price MA5")
    ax_price.set_xlabel("Date")
    ax_price.set_ylabel("Price")

    # Right axis: P/E
    ax_pe = ax_price.twinx()
    ax_pe.plot(res.index, res['pe'], label="P/E (Annualized EPS)")
    ax_pe.plot(res.index, res['pe_ttm'], label="P/E (TTM)")
    ax_pe.set_ylabel("P/E")

    # Legends
    lines_left, labels_left = ax_price.get_legend_handles_labels()
    lines_right, labels_right = ax_pe.get_legend_handles_labels()
    ax_price.legend(lines_left + lines_right, labels_left + labels_right, loc="upper left")

    ax_price.grid(True)
    plt.title(f"{ticker} — Monthly Price & P/E (with MA5)")
    plt.tight_layout()

    if savefig:
        plt.savefig(savefig, dpi=150, bbox_inches="tight")
        print(f"Saved figure to: {savefig}")
    else:
        plt.show()


# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(description="Compute monthly P/E using quarterly EPS (v3.2 with plotting).")
    ap.add_argument("--ticker", required=True, help="Ticker symbol, e.g., NVDA")
    ap.add_argument("--years", type=int, default=5, help="Lookback in years (default: 5)")
    ap.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    ap.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    ap.add_argument("--no_annualize", action="store_true", help="Use raw quarterly EPS (no x4).")
    ap.add_argument("--csv", type=str, default=None, help="Optional CSV output path")
    ap.add_argument("--plot", action="store_true", help="Show plot with 4 lines (Price/MA5, P/E, P/E TTM)")
    ap.add_argument("--savefig", type=str, default=None, help="Path to save the figure (e.g., nvda_pe.png)")
    ap.add_argument("--debug", action="store_true", help="Print debug info")
    args = ap.parse_args()

    daily = fetch_daily_prices(args.ticker, args.start, args.end, args.years if not args.start else None, debug=args.debug)
    monthly = monthly_avg_close(daily, debug=args.debug)
    eps_q = get_quarterly_eps_series(args.ticker, debug=args.debug)
    res = compute_monthly_pe(monthly, eps_q, annualize=not args.no_annualize, debug=args.debug)

    print(f"\nMonthly P/E for {args.ticker} (using quarterly EPS{' (annualized x4)' if not args.no_annualize else ''}):\n")
    print(res.tail(24))
    print("\nColumns:", ", ".join(res.columns), "\n")

    if args.csv:
        res.to_csv(args.csv, index=True)
        print(f"Saved to CSV: {args.csv}")

    if args.plot:
        plot_monthly(res, args.ticker, savefig=args.savefig)


if __name__ == "__main__":
    main()
