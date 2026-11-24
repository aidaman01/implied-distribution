"""Tests for the daily 30-day snapshot (rnd/daily.py). No network: Yahoo is simulated.
Run: python tests/test_daily.py"""
import csv
import os
import re
import sys
import tempfile
import types
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from rnd import daily  # noqa: E402
from rnd.data import heston_pdf, synthetic_chain  # noqa: E402

NY = ZoneInfo("America/New_York")
WINTER_RUN = datetime(2026, 12, 16, 15, 40, tzinfo=NY)  # a Wednesday, inside the window


def tv(K, q, truth):
    return 0.5 * np.abs(q - truth).sum() * (K[1] - K[0])


def fake_yahoo(now, days=(20, 28, 31, 45), quotes_age=timedelta(minutes=15), spot_age=None, fail=0,
               empty_quotes=False, spx_only=()):
    """A yfinance stand-in: Heston quotes (forward 100) for expiries `days` calendar days out, all last
    traded `quotes_age` before `now` (the SPX level `spot_age` before, default the same). The first
    `fail` Ticker() calls raise ConnectionError; `empty_quotes` blanks every bid and ask. Contracts are
    SPXW, except for the expiries `spx_only` days out, which list AM-settled SPX only."""
    asof = pd.Timestamp(now - quotes_age)
    spot_at = pd.Timestamp(now - (quotes_age if spot_age is None else spot_age))
    expiries = tuple((now + timedelta(days=d)).strftime("%Y-%m-%d") for d in days)
    am_only = {(now + timedelta(days=d)).strftime("%Y-%m-%d") for d in spx_only}
    calls = {"n": 0}

    def chain(expiry):
        T = (pd.Timestamp(expiry + " 16:00", tz=NY) - asof).total_seconds() / (365.25 * 86400)
        c, _, _ = synthetic_chain(T=T)
        blank = 0.0 if empty_quotes else 1.0
        root = "SPX" if expiry in am_only else "SPXW"
        side = lambda mid: pd.DataFrame({"contractSymbol": [f"{root}000000C{int(k * 1000):08d}" for k in c["strike"]],
                                         "strike": c["strike"], "bid": blank * (mid - 0.005),
                                         "ask": blank * (mid + 0.005), "lastTradeDate": [asof] * len(c)})
        return types.SimpleNamespace(calls=side(c["call_mid"]), puts=side(c["put_mid"]))

    def ticker(symbol):
        calls["n"] += 1
        if calls["n"] <= fail:
            raise ConnectionError("Yahoo did not answer")
        spot = pd.DataFrame({"Close": [99.5]}, index=pd.DatetimeIndex([spot_at]))
        return types.SimpleNamespace(options=expiries, option_chain=chain,
                                     history=lambda period, interval: spot)

    return types.SimpleNamespace(Ticker=ticker), calls


def run(now, yahoo, data_dir):
    real = sys.modules.get("yfinance")
    sys.modules["yfinance"] = yahoo
    try:
        return daily.run(now=now, data_dir=data_dir, sleep=lambda s: None)
    finally:
        if real is None:
            del sys.modules["yfinance"]
        else:
            sys.modules["yfinance"] = real


def rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_interpolation_uses_vix_weights_on_total_variance():
    """w30 = w_near (T2 - T30)/(T2 - T1) + w_next (T30 - T1)/(T2 - T1): with 28 and 31 days the
    weights are 1/3 and 2/3. Forward and discount factor are interpolated in log, linearly in T."""
    T1, T2, T30 = 28 / 365.25, 31 / 365.25, 30 / 365.25
    flat = lambda w: (w, 0.0, 0.0, 0.0, 0.1)                           # SVI with b = 0: w(k) = a
    near = {"T": T1, "F": 100.0, "DF": np.exp(-0.04 * T1), "params": flat(0.002)}
    nxt = {"T": T2, "F": 101.0, "DF": np.exp(-0.04 * T2), "params": flat(0.005)}
    F30, DF30, w30, weight = daily.interpolate(near, nxt, T30)
    assert abs(weight - 1 / 3) < 1e-12
    assert abs(w30(np.array([-0.3, 0.0, 0.2])) - (0.002 / 3 + 0.005 * 2 / 3)).max() < 1e-15
    assert abs(np.log(F30) - (np.log(100) / 3 + np.log(101) * 2 / 3)) < 1e-12
    assert abs(DF30 - np.exp(-0.04 * T30)) < 1e-12


def test_flat_smile_gives_lognormal_probabilities():
    """Both expiries at 20% vol everywhere: the 30-day density is lognormal, so market and lognormal
    tail probabilities must agree."""
    T1, T2 = 28 / 365.25, 31 / 365.25
    flat = lambda T: (0.04 * T, 0.0, 0.0, 0.0, 0.1)
    near = {"T": T1, "F": 100.0, "DF": 0.997, "params": flat(T1)}
    nxt = {"T": T2, "F": 100.0, "DF": 0.996, "params": flat(T2)}
    s = daily.thirty_day(near, nxt, spot=100.0)
    assert abs(s["atm_vol_30d"] - 0.2) < 1e-12
    assert abs(s["p_2sd_market"] - s["p_2sd_lognormal"]) < 1e-3
    assert abs(s["p_drop10_market"] - s["p_drop10_lognormal"]) < 1e-3
    assert abs(s["mass_30d"] - 1) < 1e-4 and abs(s["mean_vs_fwd_bp_30d"]) < 1


def test_bracket_is_the_vix_30_day_bracket():
    """Near: the last expiry within 30 days; next: the first one after it (Cboe's 30 Day Bracket)."""
    now = WINTER_RUN
    e = lambda d: (now + timedelta(days=d)).strftime("%Y-%m-%d")
    assert daily.bracket([e(d) for d in (1, 7, 28, 29, 31, 45)], now) == (e(29), e(31))
    assert daily.bracket([e(d) for d in (31, 45, 60)], now) == (e(31), e(45))      # none within 30 days
    # Time runs to the 16:00 expiry: from 15:40, "30 days" is 30 days 20 minutes, past the horizon.
    assert daily.bracket([e(d) for d in (20, 30, 33)], now) == (e(20), e(30))


def test_window_in_winter_and_summer():
    """15:30-16:15 New York, whatever the UTC offset (EST is UTC-5, EDT is UTC-4)."""
    utc = lambda *a: datetime(*a, tzinfo=timezone.utc)
    assert daily.in_window(utc(2026, 12, 16, 20, 35))          # 15:35 EST
    assert not daily.in_window(utc(2026, 12, 16, 19, 50))      # 14:50 EST: the summer run, too early
    assert daily.in_window(utc(2027, 7, 14, 19, 35))           # 15:35 EDT
    assert not daily.in_window(utc(2027, 7, 14, 20, 35))       # 16:35 EDT: the winter run, too late
    assert not daily.in_window(utc(2026, 12, 16, 21, 15))      # 16:15 EST: window end is exclusive


def test_workflow_crons_hit_the_window_three_times_each_season():
    """The six UTC crons in daily.yml: exactly 15:35, 15:50 and 16:05 New York both in winter and summer."""
    with open(os.path.join(ROOT, ".github", "workflows", "daily.yml")) as f:
        crons = re.findall(r'cron: "(\d+) (\d+) \* \* 1-5"', f.read())
    assert len(crons) == 6
    for day in (datetime(2026, 12, 16, tzinfo=timezone.utc), datetime(2027, 7, 14, tzinfo=timezone.utc)):
        hits = sorted(t.astimezone(NY).strftime("%H:%M") for t in
                      (day.replace(hour=int(h), minute=int(m)) for m, h in crons) if daily.in_window(t))
        assert hits == ["15:35", "15:50", "16:05"], (day, hits)


def test_collects_one_row_close_to_the_true_density():
    """End to end on simulated Heston quotes: one row, quality ok, and the 30-day density
    interpolated from the 28- and 31-day expiries is close to the exact 30-day Heston density."""
    yahoo, _ = fake_yahoo(WINTER_RUN)
    with tempfile.TemporaryDirectory() as d:
        status, _, code = run(WINTER_RUN, yahoo, d)
        data = rows(os.path.join(d, daily.DATA_FILE))
        assert (status, code, len(data)) == ("written", 0, 1), (status, data)
        r = data[0]
        assert r["date"] == "2026-12-16" and r["quality_ok"] == "True", r["quality_flags"]
        assert (r["near_expiry"], r["next_expiry"]) == ("2027-01-13", "2027-01-16")
        assert not os.path.exists(os.path.join(d, daily.SKIPPED_FILE))
    Kq, q = daily.rebuild_30d(r)                       # from the stored numbers alone, no quotes
    assert tv(Kq, q, heston_pdf(Kq, 100.0, 30 / 365.25)) < 0.04
    assert abs(float(r["F_30d"]) / 100 - 1) < 1e-3 and float(r["spot"]) == 99.5


def test_one_row_per_trading_day():
    """A second run on the same New York date writes nothing, logs nothing and never calls Yahoo."""
    yahoo, calls = fake_yahoo(WINTER_RUN)
    with tempfile.TemporaryDirectory() as d:
        run(WINTER_RUN, yahoo, d)
        n = calls["n"]
        status, _, code = run(WINTER_RUN + timedelta(minutes=15), yahoo, d)
        assert (status, code, calls["n"]) == ("duplicate", 0, n)
        assert len(rows(os.path.join(d, daily.DATA_FILE))) == 1
        assert not os.path.exists(os.path.join(d, daily.SKIPPED_FILE))


def test_outside_window_does_nothing():
    yahoo, calls = fake_yahoo(WINTER_RUN)
    with tempfile.TemporaryDirectory() as d:
        status, _, code = run(WINTER_RUN - timedelta(hours=1), yahoo, d)
        assert (status, code, calls["n"], os.listdir(d)) == ("outside window", 0, 0, [])


def test_closed_market_is_a_logged_skip_not_an_error():
    """Holiday or early close: inside the window the newest quote is hours old. No row; the skip is
    recorded in skipped_days.csv with its reason; the run itself succeeds (exit code 0)."""
    yahoo, _ = fake_yahoo(WINTER_RUN, quotes_age=timedelta(hours=2, minutes=25))   # 13:15 close
    with tempfile.TemporaryDirectory() as d:
        status, _, code = run(WINTER_RUN, yahoo, d)
        assert (status, code) == ("skipped", 0)
        assert rows(os.path.join(d, daily.DATA_FILE)) == []
        skipped = rows(os.path.join(d, daily.SKIPPED_FILE))
        assert len(skipped) == 1 and skipped[0]["date"] == "2026-12-16"
        assert "market closed" in skipped[0]["reason"], skipped[0]["reason"]


def test_yahoo_errors_are_retried_then_reported():
    """Two failures then success: one row. Persistent failure: no row, a logged skip, exit code 1."""
    with tempfile.TemporaryDirectory() as d:
        status, _, code = run(WINTER_RUN, fake_yahoo(WINTER_RUN, fail=2)[0], d)
        assert (status, code) == ("written", 0)
    with tempfile.TemporaryDirectory() as d:
        status, message, code = run(WINTER_RUN, fake_yahoo(WINTER_RUN, fail=99)[0], d)
        assert (status, code) == ("skipped", 1) and "Yahoo" in message
        assert "Yahoo error" in rows(os.path.join(d, daily.SKIPPED_FILE))[0]["reason"]


def test_success_later_in_the_day_clears_that_days_skip():
    """skipped_days.csv lists days with no row. A 15:40 Yahoo failure followed by a 15:55 success
    leaves the day with a row and no skip entry."""
    later = WINTER_RUN + timedelta(minutes=15)
    with tempfile.TemporaryDirectory() as d:
        run(WINTER_RUN, fake_yahoo(WINTER_RUN, fail=99)[0], d)
        assert len(rows(os.path.join(d, daily.SKIPPED_FILE))) == 1
        status, _, _ = run(later, fake_yahoo(later)[0], d)
        assert status == "written" and rows(os.path.join(d, daily.SKIPPED_FILE)) == []


def test_quality_flags_name_the_failed_check():
    """Rows are always written; failed checks are flagged by name, never silently dropped."""
    good = {"mass": 1.0, "mean_vs_fwd_bp": 2.0, "fit_rmse": 0.004, "rate": 0.04}
    ok30 = {"mass": 1.0, "mean_vs_fwd_bp": 1.0, "neg_mass": 0.0, "weight_near": 0.4}
    assert daily.quality_flags({"near": good, "next": good}, ok30) == []
    flags = daily.quality_flags({"near": {**good, "mass": 0.99}, "next": {**good, "rate": 0.15}},
                                {**ok30, "mean_vs_fwd_bp": 45.0, "neg_mass": 2e-5})
    assert flags == ["near: mass 0.9900", "next: implied rate 15.00%",
                     "30d: mean vs forward 45.0 bp", "30d: negative mass 2.0e-05"], flags


def test_nan_and_extrapolation_are_flagged():
    """A NaN fails every comparison, so it must be flagged explicitly; so must a 30-day point outside
    the two expiries (interpolation weight outside [0, 1])."""
    good = {"mass": 1.0, "mean_vs_fwd_bp": 2.0, "fit_rmse": 0.004, "rate": 0.04}
    flags = daily.quality_flags({"near": {**good, "fit_rmse": float("nan")}, "next": good},
                                {"mass": float("nan"), "mean_vs_fwd_bp": 1.0, "neg_mass": 0.0, "weight_near": 1.2})
    assert flags == ["near: SVI error nan vol pts", "30d: mass nan", "30d: extrapolated, weight_near 1.200"], flags


def test_quotes_stale_but_spot_fresh_is_still_a_closed_market():
    """The option chain alone can show a closed session (load_chain's own check), not only the SPX bars."""
    yahoo, _ = fake_yahoo(WINTER_RUN, quotes_age=timedelta(hours=3), spot_age=timedelta(minutes=15))
    with tempfile.TemporaryDirectory() as d:
        status, message, code = run(WINTER_RUN, yahoo, d)
        assert (status, code) == ("skipped", 0) and "market is closed" in message, message


def test_empty_quotes_on_a_trading_day_are_retried_as_a_yahoo_error():
    """Fresh timestamps but blank bid/ask: a Yahoo glitch, not a holiday. Retried, then exit code 1."""
    yahoo, calls = fake_yahoo(WINTER_RUN, empty_quotes=True)
    with tempfile.TemporaryDirectory() as d:
        status, message, code = run(WINTER_RUN, yahoo, d)
        assert (status, code) == ("skipped", 1) and "Yahoo error" in message, message
        assert calls["n"] >= 3                                         # three attempts


def test_fit_failure_is_recorded_and_fails_the_run():
    yahoo, _ = fake_yahoo(WINTER_RUN)
    real = daily.analyze

    def broken(*a, **k):
        raise FloatingPointError("SVI did not converge")

    daily.analyze = broken
    try:
        with tempfile.TemporaryDirectory() as d:
            status, message, code = run(WINTER_RUN, yahoo, d)
            assert (status, code) == ("skipped", 1) and "fit failed" in message, message
            assert "SVI did not converge" in rows(os.path.join(d, daily.SKIPPED_FILE))[0]["reason"]
    finally:
        daily.analyze = real


def test_bracket_expiry_without_spxw_moves_to_the_nearest_spxw_and_is_flagged():
    """Only PM-settled SPXW is used. If the near expiry (28 days) lists AM-settled SPX only, the next
    SPXW expiry inward (20 days) is used instead, never the SPX monthly, and the row says so."""
    yahoo, _ = fake_yahoo(WINTER_RUN, spx_only=(28,))
    with tempfile.TemporaryDirectory() as d:
        status, _, code = run(WINTER_RUN, yahoo, d)
        r = rows(os.path.join(d, daily.DATA_FILE))[0]
    assert (status, code, r["near_expiry"], r["next_expiry"]) == ("written", 0, "2027-01-05", "2027-01-16")
    # The note comes first; the 20-day fit may add its own flags (its parity rate is noisy at 20 days).
    assert r["quality_ok"] == "False" and r["quality_flags"].startswith(
        "near: 2027-01-13 has no SPXW contracts, used 2027-01-05"), r["quality_flags"]


def test_no_spxw_on_either_side_is_a_failed_run():
    yahoo, _ = fake_yahoo(WINTER_RUN, days=(28, 31), spx_only=(28, 31))
    with tempfile.TemporaryDirectory() as d:
        status, message, code = run(WINTER_RUN, yahoo, d)
        assert (status, code) == ("skipped", 1) and "no SPXW" in message, message
        assert rows(os.path.join(d, daily.DATA_FILE)) == []


def commit_step_script() -> str:
    """The shell script of the workflow's "Commit the data" step, exactly as GitHub runs it."""
    with open(os.path.join(ROOT, ".github", "workflows", "daily.yml")) as f:
        block = f.read().split("- name: Commit the data", 1)[1].split("run: |\n", 1)[1]
    lines = []
    for line in block.splitlines():
        if line.strip() and not line.startswith(" " * 10):
            break
        lines.append(line[10:])
    return "\n".join(lines)


def test_workflow_commit_step_without_and_with_data():
    """On 2026-11-23 every run was outside the window, data/ did not exist yet, and `git add data/`
    failed the step (6 red runs). Without data/ the step must succeed and commit nothing; with a new
    row it must commit as github-actions[bot] and push. Runs the real step against a local remote."""
    import subprocess
    git = lambda *a, cwd: subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True).stdout
    with tempfile.TemporaryDirectory() as tmp:
        remote, work = os.path.join(tmp, "remote.git"), os.path.join(tmp, "work")
        git("init", "-q", "--bare", "-b", "main", remote, cwd=tmp)
        git("clone", "-q", remote, work, cwd=tmp)
        git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "init", cwd=work)
        git("push", "-q", "origin", "HEAD:main", cwd=work)
        step = lambda message: subprocess.run(["bash", "-e", "-c", commit_step_script()], cwd=work,
                                              env={**os.environ, "MESSAGE": message}, capture_output=True, text=True)
        none = step("")
        assert none.returncode == 0, none.stderr
        assert git("log", "--format=%s", "origin/main", cwd=work).split() == ["init"]
        os.makedirs(os.path.join(work, "data"))
        with open(os.path.join(work, "data", daily.DATA_FILE), "w") as f:
            f.write("date\n2026-11-24\n")
        done = step("data: 2026-11-24 crash prob")
        assert done.returncode == 0, done.stderr
        assert git("log", "-1", "--format=%an|%s", "main", cwd=remote).strip() == \
            "github-actions[bot]|data: 2026-11-24 crash prob"


def test_repeated_skips_with_the_same_reason_change_nothing():
    """Three crons on a holiday give one skip line and no file change after the first (no commit churn)."""
    yahoo, _ = fake_yahoo(WINTER_RUN, quotes_age=timedelta(hours=20))
    with tempfile.TemporaryDirectory() as d:
        run(WINTER_RUN, yahoo, d)
        path = os.path.join(d, daily.SKIPPED_FILE)
        before = open(path).read()
        run(WINTER_RUN + timedelta(minutes=15), yahoo, d)
        assert open(path).read() == before and len(rows(path)) == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
