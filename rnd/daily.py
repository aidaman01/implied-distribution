"""Daily 30-day snapshot of what live ^SPX options imply for a drop: one row per trading day in
data/daily_crash_prob.csv. Usage: python -m rnd.daily  (GitHub Actions: .github/workflows/daily.yml)

Constant 30-day maturity, as the VIX does it: fit SVI to the last expiry within 30 days ("near") and
the first one after it ("next"), interpolate total implied variance w(k) = vol(k)^2 T linearly in T
to 30 days at each log-moneyness k = ln(K/F), and read the 30-day density off that smile.
The probabilities are for the index level on day 30, not for touching a level along the way."""
import csv
import logging
import os
import subprocess
import sys
import time as clock
from datetime import date, datetime, time, timezone
from typing import Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .core import analyze, density_w, lognormal_cdf, moments, svi
from .data import MarketClosed, load_chain

log = logging.getLogger("rnd.daily")

TICKER = "^SPX"
NY = ZoneInfo("America/New_York")
WINDOW = (time(15, 30), time(16, 15))   # New York time; start inclusive, end exclusive
HORIZON_DAYS = 30
YEAR_SECONDS = 365.25 * 86400
T30 = HORIZON_DAYS * 86400 / YEAR_SECONDS
DROP = 0.10                             # fixed threshold: S_30 below 90% of today's spot
MASS_TOL = 0.005                        # |probability mass - 1|
DRIFT_TOL_BP = 30.0                     # |mean of density / forward - 1|, basis points
RMSE_TOL = 0.01                         # SVI fit error, 1 vol point
RATE_RANGE = (0.0, 0.10)                # implied rate from put-call parity
NEG_MASS_TOL = 1e-5                     # negative probability after interpolation
RETRY_WAITS = (30, 60)                  # seconds before the 2nd and the 3rd attempt
MAX_SPOT_AGE = pd.Timedelta(hours=1)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
DATA_FILE = "daily_crash_prob.csv"
SKIPPED_FILE = "skipped_days.csv"
SKIPPED_COLUMNS = ["date", "run_utc", "reason"]
SLICE_FIELDS = ["expiry", "T", "F", "DF", "rate", "a", "b", "rho", "m", "s",
                "fit_rmse", "mass", "mean_vs_fwd_bp", "n_svi", "n_parity"]
COLUMNS = (["date", "run_utc", "quotes_asof_ny", "spot", "spot_time_ny",
            "F_30d", "DF_30d", "weight_near", "atm_vol_30d", "skew_30d", "ex_kurt_30d",
            "mass_30d", "mean_vs_fwd_bp_30d", "neg_mass_30d",
            "level_2sd", "p_2sd_market", "p_2sd_lognormal",
            "level_drop10", "p_drop10_market", "p_drop10_lognormal"]
           + [f"{side}_{f}" for side in ("near", "next") for f in SLICE_FIELDS]
           + ["quality_ok", "quality_flags", "code_sha"])


def in_window(t: datetime) -> bool:
    """Is t (any time zone) inside the 15:30-16:15 New York snapshot window?"""
    t_ny = t.astimezone(NY).time()
    return WINDOW[0] <= t_ny < WINDOW[1]


def years_to(expiry: str, t: datetime) -> float:
    """Year fraction from t to the 16:00 New York close on the expiry date."""
    close = datetime.combine(date.fromisoformat(expiry), time(16), NY)
    return (close - t).total_seconds() / YEAR_SECONDS


def bracket(expiries: list[str], now: datetime) -> tuple[str, str]:
    """Cboe's 30 Day Bracket: near = the last expiry at most 30 days away (or the first expiry if
    none is), next = the first expiry after near. Time is measured to the minute, like the VIX."""
    future = sorted(e for e in expiries if years_to(e, now) > 0)
    within = [e for e in future if years_to(e, now) <= T30]
    near = within[-1] if within else future[0]
    later = [e for e in future if e > near]
    if not later:
        raise ValueError(f"no expiry after {near} among {len(future)} listed")
    return near, later[0]


def interpolate(near: dict, nxt: dict, T: float = T30) -> tuple[float, float, Callable, float]:
    """VIX-style constant maturity: w_T(k) = a w_near(k) + (1 - a) w_next(k) with
    a = (T_next - T) / (T_next - T_near); ln F and ln DF are interpolated with the same weight.
    Returns (F, DF, w, a)."""
    a = (nxt["T"] - T) / (nxt["T"] - near["T"])
    F = np.exp(a * np.log(near["F"]) + (1 - a) * np.log(nxt["F"]))
    DF = np.exp(a * np.log(near["DF"]) + (1 - a) * np.log(nxt["DF"]))
    w = lambda k: a * svi(k, *near["params"]) + (1 - a) * svi(k, *nxt["params"])
    return F, DF, w, a


def thirty_day(near: dict, nxt: dict, spot: float) -> dict:
    """Density and tail probabilities of the index on day 30, from the interpolated smile."""
    F, DF, w, a = interpolate(near, nxt)
    K, q = density_w(F, T30, w)
    h = K[1] - K[0]
    s = moments(K, q, F)
    atm = np.sqrt(w(0.0) / T30)
    level_2sd = F * np.exp(-2 * atm * np.sqrt(T30))
    level_drop = (1 - DROP) * spot
    return {
        "F_30d": F, "DF_30d": DF, "weight_near": a, "atm_vol_30d": atm,
        "skew_30d": s["skew"], "ex_kurt_30d": s["ex_kurt"], "mass_30d": s["mass"],
        "mean_vs_fwd_bp_30d": (s["mean"] / F - 1) * 1e4, "neg_mass_30d": -q[q < 0].sum() * h,
        "level_2sd": level_2sd, "p_2sd_market": q[K < level_2sd].sum() * h,
        "p_2sd_lognormal": lognormal_cdf(level_2sd, F, atm, T30),
        "level_drop10": level_drop, "p_drop10_market": q[K < level_drop].sum() * h,
        "p_drop10_lognormal": lognormal_cdf(level_drop, F, atm, T30),
    }


def rebuild_30d(row: dict) -> tuple[np.ndarray, np.ndarray]:
    """The 30-day density from a stored CSV row alone (no quotes needed): (K, q)."""
    side = lambda p: {"T": float(row[f"{p}_T"]), "F": float(row[f"{p}_F"]), "DF": float(row[f"{p}_DF"]),
                      "params": tuple(float(row[f"{p}_{x}"]) for x in ("a", "b", "rho", "m", "s"))}
    F, _, w, _ = interpolate(side("near"), side("next"))
    return density_w(F, T30, w)


def summarize_slice(expiry: str, chain: pd.DataFrame, r: dict) -> dict:
    """What is stored per expiry: SVI parameters, forward, discount factor and diagnostics."""
    n_parity = int((chain["call_mid"].notna() & chain["put_mid"].notna()).sum())
    return {
        "expiry": expiry, "T": r["T"], "F": r["F"], "DF": r["DF"], "rate": -np.log(r["DF"]) / r["T"],
        **dict(zip(("a", "b", "rho", "m", "s"), r["params"])), "params": r["params"],
        "fit_rmse": r["fit_rmse"], "mass": r["stats"]["mass"],
        "mean_vs_fwd_bp": (r["stats"]["mean"] / r["F"] - 1) * 1e4,
        "n_svi": len(r["smile"][0]), "n_parity": n_parity,
    }


def quality_flags(slices: dict, r30: dict) -> list[str]:
    """Names of the failed checks; an empty list means the row passed them all. Every check is
    written as "not (value within limit)" so that a NaN, which fails every comparison, is flagged."""
    flags = []
    for side in ("near", "next"):
        s = slices[side]
        if not abs(s["mass"] - 1) < MASS_TOL:
            flags.append(f"{side}: mass {s['mass']:.4f}")
        if not abs(s["mean_vs_fwd_bp"]) < DRIFT_TOL_BP:
            flags.append(f"{side}: mean vs forward {s['mean_vs_fwd_bp']:.1f} bp")
        if not s["fit_rmse"] < RMSE_TOL:
            flags.append(f"{side}: SVI error {s['fit_rmse'] * 100:.2f} vol pts")
        if not RATE_RANGE[0] <= s["rate"] <= RATE_RANGE[1]:
            flags.append(f"{side}: implied rate {s['rate']:.2%}")
    if not abs(r30["mass"] - 1) < MASS_TOL:
        flags.append(f"30d: mass {r30['mass']:.4f}")
    if not abs(r30["mean_vs_fwd_bp"]) < DRIFT_TOL_BP:
        flags.append(f"30d: mean vs forward {r30['mean_vs_fwd_bp']:.1f} bp")
    if not r30["neg_mass"] < NEG_MASS_TOL:
        flags.append(f"30d: negative mass {r30['neg_mass']:.1e}")
    if not 0 <= r30["weight_near"] <= 1:
        flags.append(f"30d: extrapolated, weight_near {r30['weight_near']:.3f}")
    return flags


def spot_level(tk, now: pd.Timestamp) -> tuple[float, pd.Timestamp]:
    """Latest SPX level from Yahoo's 1-minute bars (delayed ~15 min); refuses a closed session's."""
    bars = tk.history(period="1d", interval="1m")
    if bars.empty:
        raise RuntimeError(f"Yahoo returned no {TICKER} price bars")
    at = pd.Timestamp(bars.index[-1]).tz_convert(NY)
    if now - at > MAX_SPOT_AGE:
        raise MarketClosed(f"latest {TICKER} level is from {at:%a %Y-%m-%d %H:%M} New York")
    return float(bars["Close"].iloc[-1]), at


def fetch(now: datetime) -> dict:
    """One attempt at everything the row needs from Yahoo."""
    import yfinance as yf

    stamp = pd.Timestamp(now).tz_convert(NY)
    tk = yf.Ticker(TICKER)
    near, nxt = bracket(list(tk.options), now)
    spot, spot_time = spot_level(tk, stamp)
    chains = {side: load_chain(TICKER, e, now=stamp) for side, e in (("near", near), ("next", nxt))}
    return {"spot": spot, "spot_time": spot_time, "chains": chains}


def fetch_with_retries(now: datetime, sleep: Callable) -> dict:
    """Retry Yahoo failures (not a closed market) twice, then give up with the last error."""
    for attempt, wait in enumerate((*RETRY_WAITS, None), start=1):
        try:
            return fetch(now)
        except MarketClosed:
            raise
        except Exception as e:  # network, rate limit, malformed reply: all worth a retry
            log.warning("Yahoo attempt %d failed: %s: %s", attempt, type(e).__name__, e)
            if wait is None:
                raise RuntimeError(f"{type(e).__name__}: {e} (after {attempt} attempts)") from e
            sleep(wait)


def build_row(now: datetime, got: dict) -> dict:
    """Fit both expiries (next with a calendar penalty against near), interpolate, check."""
    (c1, T1, e1, asof1), (c2, T2, e2, _) = got["chains"]["near"], got["chains"]["next"]
    r1 = analyze(c1, T1)
    k1 = np.log(r1["smile"][0] / r1["F"])
    r2 = analyze(c2, T2, prev=(k1.min(), k1.max(), r1["params"]))
    slices = {"near": summarize_slice(e1, c1, r1), "next": summarize_slice(e2, c2, r2)}
    r30 = thirty_day(slices["near"], slices["next"], got["spot"])
    flags = quality_flags(slices, {"mass": r30["mass_30d"], "mean_vs_fwd_bp": r30["mean_vs_fwd_bp_30d"],
                                   "neg_mass": r30["neg_mass_30d"], "weight_near": r30["weight_near"]})
    return {
        "date": now.astimezone(NY).date().isoformat(),
        "run_utc": now.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "quotes_asof_ny": asof1.isoformat(timespec="seconds"),
        "spot": got["spot"], "spot_time_ny": got["spot_time"].isoformat(timespec="seconds"),
        **r30,
        **{f"{side}_{f}": slices[side][f] for side in ("near", "next") for f in SLICE_FIELDS},
        "quality_ok": not flags, "quality_flags": "; ".join(flags), "code_sha": code_sha(),
    }


def code_sha() -> str:
    """The commit the numbers were computed with."""
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def cell(v) -> str:
    """CSV text: floats to 10 significant digits (and -0.0 as 0), everything else as str()."""
    return f"{float(v) + 0.0:.10g}" if isinstance(v, (float, np.floating)) else str(v)


def read_rows(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: str, columns: list[str], rows: list[dict]) -> None:
    """Rewrite a small CSV atomically (temp file, then rename)."""
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=columns)
        out.writeheader()
        out.writerows(rows)
    os.replace(tmp, path)


def append_row(path: str, row: dict) -> None:
    """Append one row; refuse to append to a file written with different columns."""
    new = not os.path.exists(path)
    if not new:
        with open(path, newline="") as f:
            header = next(csv.reader(f), [])
        if header != COLUMNS:
            raise RuntimeError(f"{path} has other columns than this code writes; refusing to append")
    with open(path, "a", newline="") as f:
        out = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            out.writeheader()
        out.writerow({c: cell(row[c]) for c in COLUMNS})


def record_skip(path: str, day: str, now: datetime, reason: str) -> None:
    """One line per skipped day, keeping the latest reason. An unchanged reason leaves the file
    untouched, so the later crons on a holiday make no new commit."""
    rows = read_rows(path)
    reason = " ".join(reason.split())[:300]
    if any(r["date"] == day and r["reason"] == reason for r in rows):
        return
    kept = [r for r in rows if r["date"] != day]
    run_utc = now.astimezone(timezone.utc).isoformat(timespec="seconds")
    write_rows(path, SKIPPED_COLUMNS, kept + [{"date": day, "run_utc": run_utc, "reason": reason}])


def clear_skip(path: str, day: str) -> None:
    rows = read_rows(path)
    kept = [r for r in rows if r["date"] != day]
    if len(kept) != len(rows):
        write_rows(path, SKIPPED_COLUMNS, kept)


def run(now: datetime | None = None, data_dir: str = DATA_DIR,
        sleep: Callable = clock.sleep) -> tuple[str, str, int]:
    """One scheduled run. Returns (status, message, exit code). Status: outside window, duplicate,
    skipped (logged in skipped_days.csv) or written. Only Yahoo or fit failures exit non-zero."""
    now = now or datetime.now(timezone.utc)
    day = now.astimezone(NY).date().isoformat()
    if not in_window(now):
        return "outside window", f"{now.astimezone(NY):%H:%M} New York is outside 15:30-16:15: nothing to do", 0
    data_path, skip_path = os.path.join(data_dir, DATA_FILE), os.path.join(data_dir, SKIPPED_FILE)
    if any(r["date"] == day for r in read_rows(data_path)):
        return "duplicate", f"{day} is already in {DATA_FILE}: nothing to do", 0
    os.makedirs(data_dir, exist_ok=True)
    try:
        got = fetch_with_retries(now, sleep)
    except MarketClosed as e:
        record_skip(skip_path, day, now, f"market closed: {e}")
        return "skipped", f"{day} skipped, market closed: {e}", 0
    except RuntimeError as e:
        record_skip(skip_path, day, now, f"Yahoo error: {e}")
        return "skipped", f"{day} skipped, Yahoo error: {e}", 1
    try:
        row = build_row(now, got)
    except Exception as e:  # a fit that cannot be computed: recorded, then reported as a failed run
        record_skip(skip_path, day, now, f"fit failed: {type(e).__name__}: {e}")
        return "skipped", f"{day} skipped, fit failed: {type(e).__name__}: {e}", 1
    append_row(data_path, row)
    clear_skip(skip_path, day)
    flags = row["quality_flags"] or "all checks passed"
    return "written", (f"{day}: P(drop > 10% by day 30) market {row['p_drop10_market']:.2%}, "
                       f"lognormal {row['p_drop10_lognormal']:.2%}; {flags}"), 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    now = datetime.now(timezone.utc)
    status, message, code = run(now)
    log.info("%s: %s", status, message)
    out = os.environ.get("GITHUB_OUTPUT")
    if out and status in ("written", "skipped"):
        day = now.astimezone(NY).date().isoformat()
        kind = "crash prob" if status == "written" else "skipped"
        with open(out, "a") as f:
            f.write(f"message=data: {day} {kind}\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
