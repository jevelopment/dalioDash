"""
Macro Dashboard — Ray Dalio / late-cycle monitor.
- Real S&P 500 earnings yield (multpl.com scrape, falls back to flat-PE with warning)
- Single fetch per ticker (reused for daily change + charts)
- Yield curve slope (10Y − 3M) as a recession indicator
- HY OAS credit spread from FRED (BAMLH0A0HYM2) with historical endnote
- Credit-risk ratio uses HYG/IEF (duration-matched) with HYG/LQD overlay for reference
- Atomic writes for index.html and macro_dashboard.png
- Structured logging to dashboard.log
"""

import os
import sys
import tempfile
import traceback
from datetime import datetime
from io import StringIO

import matplotlib
matplotlib.use("Agg")  # headless — required under cron
import matplotlib.pyplot as plt
import pandas as pd
import requests
import yfinance as tck

from logging_setup import get_logger

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
log = get_logger()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PERIOD = "6mo"          # chart window
LOOKBACK_DAYS = 400     # extra days for EPS forward-fill at chart start
MULTPL_URL = "https://www.multpl.com/s-p-500-earnings/table/by-quarter"
FRED_HY_OAS_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAMLH0A0HYM2"
HTTP_TIMEOUT = 15
USER_AGENT = "Mozilla/5.0 (compatible; ray_dalio_dashboard/1.0)"

# Endnote text for the HY OAS panel — kept out of the plot to preserve layout
HY_OAS_ENDNOTE = (
    "* HY OAS = ICE BofA US High Yield Index Option-Adjusted Spread: the extra yield "
    "junk bonds must pay over Treasuries. Historical anchors — long-run median ≈ 400 bps; "
    "pre-GFC bubble low ≈ 240 bps; COVID peak ≈ 1,100 bps; 2022 peak ≈ 600 bps. "
    "Below ~300 bps = late-cycle complacency. Above ~700 bps = genuine credit stress."
)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def fetch_close(ticker: str, period: str = PERIOD) -> pd.Series:
    """Fetch close prices for a ticker. Returns empty Series on failure."""
    try:
        df = tck.Ticker(ticker).history(period=period, auto_adjust=False)
    except Exception as e:
        log.error(f"fetch_close({ticker}) raised: {e}")
        return pd.Series(dtype=float)
    if df is None or df.empty or "Close" not in df.columns:
        log.warning(f"fetch_close({ticker}) returned empty data")
        return pd.Series(dtype=float)
    return df["Close"].dropna()


def get_change(series: pd.Series) -> tuple[float, float]:
    """Return (latest, pct_change) or (nan, nan) if insufficient data."""
    if series is None or len(series) < 2:
        return float("nan"), float("nan")
    latest, prev = series.iloc[-1], series.iloc[-2]
    if prev == 0 or pd.isna(prev):
        return latest, float("nan")
    return latest, ((latest - prev) / prev) * 100


def scrape_multpl_eps() -> pd.Series:
    """
    Scrape S&P 500 TTM earnings from multpl.com.
    Returns a Series indexed by date, quarterly frequency.
    Raises on any failure so the caller can fall back.
    """
    resp = requests.get(MULTPL_URL, headers={"User-Agent": USER_AGENT}, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    if not tables:
        raise ValueError("multpl returned no tables")
    df = tables[0].copy()
    if df.shape[1] < 2:
        raise ValueError(f"unexpected multpl table shape: {df.shape}")
    # First two columns are date and value
    df = df.iloc[:, :2]
    df.columns = ["Date", "EPS"]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    # Strip $, commas, whitespace; handle possible "estimate" suffixes
    df["EPS"] = (
        df["EPS"].astype(str)
        .str.replace(r"[$,]", "", regex=True)
        .str.replace(r"[^\d.\-]", "", regex=True)
    )
    df["EPS"] = pd.to_numeric(df["EPS"], errors="coerce")
    df = df.dropna().sort_values("Date").set_index("Date")["EPS"]
    # Ensure tz-naive — pandas 3.x is strict about mixed-tz comparisons
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    if df.empty:
        raise ValueError("multpl EPS series parsed to empty")
    log.info(f"multpl EPS: {len(df)} quarterly points, latest {df.index[-1].date()} = {df.iloc[-1]:.2f}")
    return df


def fetch_fred_hy_oas() -> pd.Series:
    """
    Fetch ICE BofA US High Yield OAS from FRED's public CSV endpoint.
    Returns a daily Series (tz-naive) trimmed to roughly 6 months.
    Raises on any failure so the caller can fall back gracefully.
    """
    resp = requests.get(FRED_HY_OAS_URL,
                        headers={"User-Agent": USER_AGENT},
                        timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    # FRED CSV: first column is observation date, second is the series value
    if df.shape[1] < 2:
        raise ValueError(f"unexpected FRED CSV shape: {df.shape}")
    df.columns = ["Date", "OAS"]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["OAS"] = pd.to_numeric(df["OAS"], errors="coerce")
    df = df.dropna().sort_values("Date").set_index("Date")["OAS"]
    if df.empty:
        raise ValueError("FRED HY OAS series parsed to empty")
    # Trim to ~6 months — enough for the chart window
    cutoff = df.index[-1] - pd.Timedelta(days=200)
    df = df[df.index >= cutoff]
    log.info(f"FRED HY OAS: {len(df)} points, latest {df.index[-1].date()} = {df.iloc[-1]:.2f}%")
    return df


def get_sp500_earnings_yield(index_prices: pd.Series) -> tuple[pd.Series, str, bool]:
    """
    Returns (earnings_yield_series, source_label, is_fallback).

    IMPORTANT: `index_prices` must be the S&P 500 *index level* (^GSPC), not SPY.
    multpl.com publishes index-scale TTM EPS (~$298); SPY trades at ~1/10 the
    index level, so dividing index EPS by SPY price inflates the yield ~10x.

    Primary: multpl.com TTM EPS forward-filled to daily, divided by ^GSPC.
    Fallback: flat trailingPE from yfinance (scaled inverse-price proxy).

    Note: yfinance returns tz-aware indexes (America/New_York);
    multpl returns tz-naive. We normalize both to tz-naive for alignment,
    then restore the original tz on the output so the plot shares an axis
    with the TNX series.
    """
    if index_prices.empty:
        return pd.Series(dtype=float), "UNAVAILABLE", True

    original_tz = index_prices.index.tz
    spy_naive = index_prices.copy()  # keeps the name to minimize churn below
    spy_naive.index = spy_naive.index.tz_localize(None) if original_tz is not None else spy_naive.index

    # --- primary: multpl ---
    try:
        eps_q = scrape_multpl_eps()  # tz-naive DatetimeIndex
        start = spy_naive.index[0] - pd.Timedelta(days=LOOKBACK_DAYS)
        daily_idx = pd.date_range(start=start, end=spy_naive.index[-1], freq="D")
        eps_daily = eps_q.reindex(daily_idx, method="ffill")
        eps_on_spy = eps_daily.reindex(spy_naive.index, method="ffill")
        if eps_on_spy.isna().all():
            raise ValueError("multpl EPS did not align to SPY index")
        ey = (eps_on_spy / spy_naive) * 100
        # Restore original tz so it plots on the same axis as TNX
        if original_tz is not None:
            ey.index = ey.index.tz_localize(original_tz)
        log.info(f"Earnings yield: using real TTM EPS from multpl "
                 f"(latest EPS={eps_q.iloc[-1]:.2f}, index=GSPC, tz={original_tz})")
        return ey, "S&P 500 Earnings Yield (TTM, multpl.com)", False
    except Exception as e:
        log.warning(f"multpl earnings scrape failed: {e}")

    # --- fallback: yfinance trailingPE held flat ---
    try:
        info = tck.Ticker("SPY").info
    except Exception as e:
        log.error(f"yfinance .info also failed: {e}")
        return pd.Series(dtype=float), "UNAVAILABLE", True

    pe = info.get("trailingPE")
    if pe is None or pd.isna(pe) or pe <= 0:
        log.error("trailingPE unavailable — cannot compute any earnings yield.")
        return pd.Series(dtype=float), "UNAVAILABLE", True

    latest = spy_naive.iloc[-1]
    ttm_eps = latest / pe
    log.warning(f"FALLBACK: flat trailingPE={pe:.2f} — earnings yield is a scaled "
                f"inverse-price proxy, NOT historical earnings.")
    ey = (ttm_eps / spy_naive) * 100
    if original_tz is not None:
        ey.index = ey.index.tz_localize(original_tz)
    return ey, f"S&P 500 Earnings Yield (FLAT-PE PROXY, PE={pe:.1f} — UNRELIABLE)", True


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------

def write_atomic(path: str, content: str) -> None:
    """Write text to path via a temp file + os.replace (atomic on POSIX).
    Sets mode 0644 so the file is readable by the HTTP server user."""
    dir_ = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=dir_, text=True)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def savefig_atomic(fig, path: str) -> None:
    """Save a matplotlib figure atomically, mode 0644."""
    dir_ = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".png")
    os.close(fd)
    try:
        fig.savefig(tmp)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run() -> None:
    os.chdir(BASE_DIR)
    log.info("=== Dashboard run starting ===")

    # 1. Fetch each ticker exactly once
    log.info("Fetching market data...")
    spy_6m  = fetch_close("SPY")        # used for the price card only
    gspc_6m = fetch_close("^GSPC")      # S&P 500 index — pairs with multpl's index-scale EPS
    tnx_6m  = fetch_close("^TNX")
    irx_6m  = fetch_close("^IRX")       # 13-week T-bill yield, for curve slope
    hyg_6m  = fetch_close("HYG")
    lqd_6m  = fetch_close("LQD")
    ief_6m  = fetch_close("IEF")

    if spy_6m.empty:
        raise RuntimeError("SPY data unavailable — cannot build dashboard")

    spy_val, spy_chg = get_change(spy_6m)
    tnx_val, tnx_chg_pct = get_change(tnx_6m)
    hyg_val, hyg_chg = get_change(hyg_6m)
    lqd_val, lqd_chg = get_change(lqd_6m)
    ief_val, ief_chg = get_change(ief_6m)

    # ^TNX is a yield — show bps change, not percent-of-percent
    if len(tnx_6m) >= 2:
        tnx_bp = (tnx_6m.iloc[-1] - tnx_6m.iloc[-2]) * 100
    else:
        tnx_bp = float("nan")

    # 2. FRED: HY OAS (credit spread)
    log.info("Fetching HY OAS from FRED...")
    try:
        hy_oas = fetch_fred_hy_oas()
        hy_oas_ok = True
    except Exception as e:
        log.warning(f"FRED HY OAS fetch failed: {e}")
        hy_oas = pd.Series(dtype=float)
        hy_oas_ok = False

    # 3. Earnings yield — use ^GSPC (index level), NOT SPY.
    # multpl publishes index-scale EPS (~$298); SPY trades at ~1/10 the index level.
    # Dividing index EPS by SPY price inflates the yield ~10x. ^GSPC is the correct pairing.
    log.info("Computing S&P 500 earnings yield...")
    earnings_yield, ey_label, ey_is_fallback = get_sp500_earnings_yield(gspc_6m)

    # 4. Derived series
    # Primary credit-risk ratio: HYG / IEF (duration-matched, credit-isolated)
    # Secondary for comparison:  HYG / LQD (legacy — has duration contamination)
    if not hyg_6m.empty and not ief_6m.empty:
        credit_ratio = (hyg_6m / ief_6m).dropna()
        credit_ratio_label = "HYG / IEF (credit-isolated)"
        credit_ratio_source = "ief"
    elif not hyg_6m.empty and not lqd_6m.empty:
        credit_ratio = (hyg_6m / lqd_6m).dropna()
        credit_ratio_label = "HYG / LQD (legacy — IEF unavailable)"
        credit_ratio_source = "lqd-fallback"
        log.warning("IEF unavailable — falling back to HYG/LQD ratio for Panel 4")
    else:
        credit_ratio = pd.Series(dtype=float)
        credit_ratio_label = "UNAVAILABLE"
        credit_ratio_source = "none"

    legacy_ratio = (
        (hyg_6m / lqd_6m).dropna()
        if not hyg_6m.empty and not lqd_6m.empty
        else pd.Series(dtype=float)
    )

    # Yield curve slope: 10Y minus 3M, both tz-aware from yfinance
    if not tnx_6m.empty and not irx_6m.empty:
        yield_curve = (tnx_6m - irx_6m).dropna()
        if len(yield_curve) >= 1:
            curve_val = float(yield_curve.iloc[-1])
            curve_bp = curve_val * 100
        else:
            curve_val = float("nan")
            curve_bp = float("nan")
    else:
        yield_curve = pd.Series(dtype=float)
        curve_val = float("nan")
        curve_bp = float("nan")
        log.warning("^TNX or ^IRX unavailable — skipping yield-curve panel")

    # 5. Chart — 5 panels, with reserved space at the bottom for the endnote
    log.info("Rendering chart...")
    # sharex=False: each panel draws its own x-axis date labels.
    # (With sharex=True, matplotlib hides all but the bottom row's labels,
    # and fig.autofmt_xdate() re-hides them even after tick_params overrides.)
    fig, axes = plt.subplots(5, 1, figsize=(10, 22), sharex=False)
    # Reserve ~7% of the figure height at the bottom for the footnote
    fig.subplots_adjust(bottom=0.08)

    # Panel 1: Valuation squeeze
    axes[0].plot(tnx_6m.index, tnx_6m, label="10Y Treasury Yield (^TNX)", color="crimson")
    if not earnings_yield.empty:
        style = dict(linestyle="--", color="purple")
        if ey_is_fallback:
            style = dict(linestyle=":", color="grey", linewidth=2)
        axes[0].plot(earnings_yield.index, earnings_yield, label=ey_label, **style)
    axes[0].set_title(
        "1. Valuation Squeeze (10Y Yield vs. S&P Earnings Yield)\n"
        "Generic Guide: When the 10Y (red) exceeds the S&P earnings yield (purple), "
        "risk-free cash competes with equities.",
        fontsize=10,
    )
    axes[0].set_ylabel("Yield (%)")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="upper left", fontsize=8)

    # Panel 2: Yield curve slope (10Y − 3M)
    if not yield_curve.empty:
        axes[1].plot(yield_curve.index, yield_curve,
                     label="10Y − 3M slope", color="navy", linewidth=2)
        axes[1].axhline(0, color="black", linewidth=0.8, alpha=0.6)
        axes[1].fill_between(yield_curve.index, yield_curve, 0,
                             where=(yield_curve < 0), color="red", alpha=0.15,
                             label="Inverted (recession signal)")
    axes[1].set_title(
        "2. Yield Curve Slope (10Y − 3M)\n"
        "Positive = normal. Below zero (shaded red) = inverted, historically a "
        "recession precursor.",
        fontsize=10,
    )
    axes[1].set_ylabel("Spread (pp)")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="upper left", fontsize=8)

    # Panel 3: Corporate debt prices
    if not hyg_6m.empty:
        axes[2].plot(hyg_6m.index, hyg_6m, label="High-Yield Bonds (HYG)", color="darkorange")
    if not lqd_6m.empty:
        axes[2].plot(lqd_6m.index, lqd_6m, label="Investment-Grade (LQD)", color="royalblue")
    axes[2].set_title(
        "3. Corporate Debt Prices (HYG vs LQD)\n"
        "LQD (safe, long-duration) responds to rates; HYG (risky, short-duration) "
        "responds to credit stress.",
        fontsize=9,
    )
    axes[2].set_ylabel("Price ($)")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend(loc="upper left", fontsize=8)

    # Panel 4: Credit risk appetite
    if not credit_ratio.empty:
        axes[3].plot(credit_ratio.index, credit_ratio,
                     label=credit_ratio_label, color="teal", linewidth=2)
    if credit_ratio_source == "ief" and not legacy_ratio.empty:
        axes[3].plot(legacy_ratio.index, legacy_ratio,
                     label="HYG / LQD (legacy, duration-contaminated)",
                     color="grey", linestyle=":", linewidth=1, alpha=0.7)
    axes[3].set_title(
        "4. Credit Risk Appetite (HYG / IEF)\n"
        "HYG and IEF have matched duration, so the ratio moves on credit quality alone.\n"
        "Rising = Risk-On (junk outperforming duration-matched Treasuries). "
        "Falling = Flight to safety.",
        fontsize=10,
    )
    axes[3].set_ylabel("Ratio")
    axes[3].grid(True, linestyle="--", alpha=0.6)
    axes[3].legend(loc="upper left", fontsize=8)

    # Panel 5: HY OAS from FRED — with asterisk pointing to endnote
    if hy_oas_ok and not hy_oas.empty:
        axes[4].plot(hy_oas.index, hy_oas,
                     label="ICE BofA US High Yield OAS (BAMLH0A0HYM2)",
                     color="firebrick", linewidth=2)
    axes[4].set_title(
        "5. High-Yield Credit Spread (HY OAS, from FRED)*\n"
        "Widening = stress rising. Pure spread, so no duration contamination "
        "like the price ratios above.",
        fontsize=10,
    )
    axes[4].set_ylabel("Spread (%)")
    axes[4].set_xlabel("Date")
    axes[4].grid(True, linestyle="--", alpha=0.6)
    axes[4].legend(loc="upper left", fontsize=8)

    # Rotate x-axis tick labels on every panel.
    # NOTE: do NOT use fig.autofmt_xdate() — it is designed to hide
    # all but the bottom panel's labels and ignores sharex=False.
    for ax in axes:
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha("right")

    # Footnote at the bottom of the figure — only if HY OAS loaded
    if hy_oas_ok and not hy_oas.empty:
        fig.text(
            0.05, 0.015, HY_OAS_ENDNOTE,
            fontsize=7.5, color="#444444",
            ha="left", va="bottom", wrap=True,
            bbox=dict(boxstyle="round,pad=0.5",
                      facecolor="#f9f9f9",
                      edgecolor="#dddddd",
                      linewidth=0.8),
        )

    # Explicit layout instead of tight_layout — the latter collapses
    # the spacing we need for per-panel date labels + the footnote box.
    fig.subplots_adjust(
        left=0.08, right=0.97,
        top=0.97, bottom=0.08,   # bottom reserves room for the endnote
        hspace=0.55,             # vertical gap between panels
    )
    savefig_atomic(fig, os.path.join(BASE_DIR, "macro_dashboard.png"))
    plt.close(fig)

    # 6. HTML
    log.info("Writing index.html...")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data_date = spy_6m.index[-1].strftime("%Y-%m-%d")

    warning_banner = ""
    if ey_is_fallback:
        warning_banner = (
            '<div style="background:#fff3cd;border:1px solid #ffc107;color:#856404;'
            'padding:12px;border-radius:6px;margin-bottom:20px;text-align:left;">'
            '<strong>⚠ Data warning:</strong> The S&P 500 earnings yield is using a '
            '<em>flat trailing-PE proxy</em>, not real historical earnings. '
            'The purple line in Panel 1 is a scaled inverse-price curve and should '
            'not be read as actual earnings yield. Check <code>dashboard.log</code> '
            'for the multpl.com scrape failure reason.</div>'
        )

    def card(title: str, value: str, change: str, change_positive: bool) -> str:
        cls = "positive" if change_positive else "negative"
        return (
            f'<div class="card"><h3>{title}</h3>'
            f'<div class="val">{value}</div>'
            f'<div class="val {cls}">{change}</div></div>'
        )

    def fmt_pct(x: float) -> str:
        return f"{x:+.2f}%" if not pd.isna(x) else "—"

    def fmt_bp(x: float) -> str:
        return f"{x:+.1f} bps" if not pd.isna(x) else "—"

    def fmt_price(x: float) -> str:
        return f"${x:.2f}" if not pd.isna(x) else "—"

    cards_html = "".join([
        card("S&P 500 (SPY)", fmt_price(spy_val), fmt_pct(spy_chg), spy_chg >= 0),
        card("10Y Treasury (^TNX)",
             f"{tnx_val:.2f}%" if not pd.isna(tnx_val) else "—",
             fmt_bp(tnx_bp), tnx_bp >= 0),
        card("High Yield (HYG)", fmt_price(hyg_val), fmt_pct(hyg_chg), hyg_chg >= 0),
        card("7-10Y Treasury (IEF)", fmt_price(ief_val), fmt_pct(ief_chg), ief_chg >= 0),
        card("10Y − 3M Curve",
             f"{curve_val:+.2f} pp" if not pd.isna(curve_val) else "—",
             fmt_bp(curve_bp), curve_bp >= 0),
    ])

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Late-Cycle Macro Monitor</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta http-equiv="refresh" content="1800">
    <style>
        body {{ font-family: Arial, sans-serif; background: #f4f6f9; color: #333; margin: 0; padding: 20px; text-align: center; }}
        .container {{ max-width: 1000px; margin: auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
        h1 {{ color: #111; margin-bottom: 5px; }}
        .timestamp {{ color: #666; font-size: 0.9em; margin-bottom: 5px; }}
        .data-date {{ color: #888; font-size: 0.85em; margin-bottom: 25px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 15px; margin-bottom: 30px; }}
        .card {{ background: #fafbfc; border: 1px solid #e1e4e8; padding: 15px; border-radius: 6px; text-align: center; }}
        .card h3 {{ margin: 0 0 10px 0; font-size: 1em; color: #555; }}
        .card .val {{ font-size: 1.2em; font-weight: bold; }}
        .positive {{ color: #28a745; }}
        .negative {{ color: #dc3545; }}
        img {{ max-width: 100%; height: auto; border-radius: 6px; border: 1px solid #ddd; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Late-Cycle Macro Monitor</h1>
        <div class="timestamp">Rendered: {timestamp}</div>
        <div class="data-date">Market data as of {data_date} close</div>
        {warning_banner}
        <div class="metrics-grid">{cards_html}</div>
        <div><img src="macro_dashboard.png" alt="Macro Dashboard"></div>
    </div>
</body>
</html>
"""

    write_atomic(os.path.join(BASE_DIR, "index.html"), html_content)
    log.info("=== Dashboard run complete ===")


def main() -> None:
    try:
        run()
    except Exception:
        log.error("Dashboard generation failed:\n" + traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
