# Late-Cycle Macro Monitor

A single-page macro dashboard modeled on Ray Dalio's late-cycle framework.
Runs on a schedule, fetches market data from free public sources, and produces
a static `index.html` + `macro_dashboard.png` served by a lightweight HTTP server.

## What it shows

Five panels, plus a metrics strip:

1. **Valuation Squeeze** — 10Y Treasury yield vs. S&P 500 trailing earnings yield.
   When the 10Y exceeds the earnings yield, risk-free cash competes with equities.
2. **Yield Curve Slope (10Y − 3M)** — Negative (inverted) is a historical recession precursor.
3. **Corporate Debt Prices (HYG vs LQD)** — Context for how rate moves and credit stress
   pull on different parts of the corporate credit market.
4. **Credit Risk Appetite (HYG / IEF)** — Duration-matched ratio; moves on credit quality
   alone. Legacy HYG/LQD overlay shown for reference.
5. **High-Yield Credit Spread (HY OAS, FRED)** — Direct measure of credit stress.
   Footnote at the bottom explains historical anchors.

Metrics strip cards: SPY, 10Y (^TNX), HYG, IEF, 10Y − 3M curve.

## Data sources

| Source | What | Notes |
|--------|------|-------|
| yfinance | SPY, ^TNX, ^IRX, HYG, LQD, IEF | Free, no key. Occasionally flaky. |
| multpl.com | S&P 500 TTM EPS | Scraped. Falls back to flat-PE with warning banner. |
| FRED CSV | HY OAS (`BAMLH0A0HYM2`) | Free, no key. Rolling ~3-year window. |

## Files

- `app.py` — main script
- `logging_setup.py` — rotates `dashboard.log` at 1 MB, keeps 5 backups
- `index.html` — generated output, served via HTTP
- `macro_dashboard.png` — generated chart
- `dashboard.log` — application log
- `cron.log` — captures cron stdout/stderr
- `requirements.txt` — pinned dependencies from `pip freeze`
- `venv/` — virtualenv
- `app.py.bak.*` / `app.py.pre-*` — historical backups

## Running manually

```bash
cd /home/devuan/websites/ray_dalio
source venv/bin/activate
python3 app.py
