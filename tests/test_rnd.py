"""Every test checks against a known answer. Run: python -m pytest  (or python tests/test_rnd.py)"""
import os
import sys
import types

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
from rnd.core import analyze, black, calendar_violation, density, implied_vol, lognormal_pdf, moments, parity_forward
from rnd.__main__ import fit_surface
from download import spxw_2019
from rnd.data import HESTON, heston_call, heston_cf, heston_pdf, load_cboe_csv, load_chain, synthetic_chain


def tv(K, q, truth):
    """Total variation distance between two densities on a uniform grid (0 = identical, 1 = disjoint)."""
    return 0.5 * np.abs(q - truth).sum() * (K[1] - K[0])


def test_implied_vol_round_trip():
    # Out-of-the-money options, as the engine uses (deep ITM time value can sit below float precision).
    K = np.linspace(50, 200, 61)
    call = K >= 100
    for vol in (0.05, 0.2, 0.8):
        price = black(100, K, 0.75, vol, call)
        iv = implied_vol(price, 100, K, 0.75, call)
        quotable = price > 1e-6  # far below any real tick
        assert np.isfinite(iv[quotable]).all()
        assert np.abs(iv[quotable] - vol).max() < 1e-8


def test_implied_vol_rejects_arbitrage_prices():
    assert np.isnan(implied_vol(5.0, 100, 90, 1.0, True))     # below intrinsic 10
    assert np.isnan(implied_vol(101.0, 100, 90, 1.0, True))   # above forward


def test_heston_reference_implementation():
    T = 0.5
    assert abs(heston_cf(-1j, T, **HESTON) - 1) < 1e-12               # martingale
    flat = dict(v0=0.04, kappa=1.0, theta=0.04, xi=1e-4, rho=0.0)     # vol of vol -> 0
    K = np.array([80.0, 100.0, 120.0])
    assert np.abs(heston_call(100, K, T, flat) - black(100, K, T, 0.2)).max() < 1e-6
    Kc = np.linspace(60, 140, 401)
    C = heston_call(100, Kc, T)
    d2 = (C[:-2] - 2 * C[1:-1] + C[2:]) / (Kc[1] - Kc[0]) ** 2       # B-L identity on exact prices
    assert np.abs(d2 - heston_pdf(Kc[1:-1], 100, T)).max() < 1e-5


def test_parity_forward():
    F, DF, T = 100.0, np.exp(-0.04 * 0.5), 0.5
    K = np.arange(60.0, 141.0)
    C = DF * heston_call(F, K, T)
    Fh, DFh = parity_forward(K, C, C - DF * (F - K))
    assert abs(Fh - F) < 1e-8 and abs(DFh - DF) < 1e-10
    chain, T, DF = synthetic_chain()
    Fh, DFh = parity_forward(chain["strike"].to_numpy(), chain["call_mid"].to_numpy(), chain["put_mid"].to_numpy())
    assert abs(Fh / 100 - 1) < 1e-3 and abs(DFh / DF - 1) < 1e-3


def test_flat_vol_recovers_lognormal():
    F, T, DF, vol = 100.0, 0.5, 0.98, 0.2
    K = np.arange(50.0, 161.0)
    C = DF * black(F, K, T, vol)
    r = analyze(pd.DataFrame({"strike": K, "call_mid": C, "put_mid": C - DF * (F - K)}), T)
    Kq, q = r["density"]
    assert r["fit_rmse"] < 1e-4
    assert tv(Kq, q, lognormal_pdf(Kq, F, vol, T)) < 0.005


def test_heston_density_recovered_from_noisy_quotes():
    for seed in range(5):
        chain, T, _ = synthetic_chain(seed=seed)
        r = analyze(chain, T)
        Kq, q = r["density"]
        truth = heston_pdf(Kq, 100.0, T)
        s = r["stats"]
        assert tv(Kq, q, truth) < 0.03, seed
        assert abs(s["mass"] - 1) < 0.005 and abs(s["mean"] / 100 - 1) < 0.001
        assert s["min_q"] >= 0                                          # no negative probabilities
        p_true = truth[Kq < r["crash_level"]].sum() * (Kq[1] - Kq[0])
        assert abs(r["p_crash"] / p_true - 1) < 0.15, seed
        assert r["p_crash"] > 1.5 * r["p_crash_lognormal"]             # fat left tail detected


def test_robust_to_bad_wing_quotes():
    chain, T, _ = synthetic_chain()
    put_wing = chain.index[chain["put_mid"].notna()][:15]
    call_wing = chain.index[chain["call_mid"].notna()][-15:]
    chain.loc[put_wing[::5], "put_mid"] *= 1.5    # 3 stale quotes per wing, off by 50%
    chain.loc[call_wing[::5], "call_mid"] *= 0.5
    Kq, q = analyze(chain, T)["density"]
    assert tv(Kq, q, heston_pdf(Kq, 100.0, T)) < 0.03


def test_real_spx_quotes_all_expiries():
    """Real Cboe SPXW snapshot, 2019-06-26 15:45 (fetched by tests/data/download.py). No true density exists for real data,
    so check what must hold: probabilities, forward, rates, calendar no-arbitrage, equity skew."""
    path = spxw_2019()
    spot = (2917.80 + 2918.42) / 2  # SPX bid/ask in the same file
    atm_var = []
    for e in sorted(pd.read_csv(path)["expiration"].unique()):
        chain, T = load_cboe_csv(path, e)
        if T * 365.25 < 7:
            continue
        r = analyze(chain, T)
        Kq, q = r["density"]
        s = r["stats"]
        assert abs(s["mass"] - 1) < 0.005, e                            # probabilities sum to 1
        assert -q[q < 0].sum() * (Kq[1] - Kq[0]) < 1e-5, e              # no material negative mass
        assert abs(s["mean"] / r["F"] - 1) < 0.003, e                   # martingale: mean = forward
        assert abs(r["F"] / spot - 1) < 0.005, e                        # forward close to spot
        assert 0.015 < -np.log(r["DF"]) / T < 0.035, e                  # mid-2019 rates (Fed funds ~2.4%)
        assert r["fit_rmse"] < 0.01, e                                  # SVI within 1 vol pt
        assert s["skew"] < 0 and r["p_crash"] > 1.5 * r["p_crash_lognormal"], e
        atm_var.append(r["atm_vol"] ** 2 * T)
    assert len(atm_var) == 27 and np.all(np.diff(atm_var) > 0)        # calendar no-arbitrage


def test_naive_breeden_litzenberger_fails_on_noise():
    chain, T, _ = synthetic_chain()
    assert (analyze(chain, T)["naive"][1] < 0).mean() > 0.10


def test_live_loader_drops_stale_quotes():
    """Yahoo keeps showing the last bid/ask of contracts that stopped trading. On live SPX (2026-10-14)
    such quotes pushed the parity rate to 38%. Feed load_chain a fake yfinance chain with the same flaw."""
    now = pd.Timestamp.now(tz="America/New_York")
    expiry = (now + pd.Timedelta(days=182)).strftime("%Y-%m-%d")
    T = (pd.Timestamp(expiry + " 16:00", tz="America/New_York") - now).total_seconds() / (365.25 * 86400)
    chain, _, DF = synthetic_chain(T=T, r=0.04)
    stale = chain["strike"] < 80                                       # deep in-the-money calls, two months old,
    call = chain["call_mid"] + np.where(stale, DF * 10, 0)            # quoted when the forward was 110

    def side(mid, old):
        return pd.DataFrame({"strike": chain["strike"], "bid": mid - 0.005, "ask": mid + 0.005,
                             "lastTradeDate": now - pd.to_timedelta(np.where(old, 60, 0), unit="D")})

    calls, puts = side(call, stale), side(chain["put_mid"], False)
    yahoo = types.SimpleNamespace(options=(expiry,), option_chain=lambda e: types.SimpleNamespace(calls=calls, puts=puts))
    real = sys.modules.get("yfinance")
    sys.modules["yfinance"] = types.SimpleNamespace(Ticker=lambda ticker: yahoo)
    try:
        live, T_live, _, _ = load_chain("^SPX", expiry)
    finally:
        sys.modules["yfinance"] = real
    r = analyze(live, T_live)
    assert abs(-np.log(r["DF"]) / T_live - 0.04) < 0.003 and abs(r["F"] / 100 - 1) < 1e-3


def test_density_grid_holds_fat_left_tail():
    """SVI fitted to live SPX (2026-10-14), 6-month expiry. A +-8 sd grid lost 0.56% of the probability
    off its left edge, putting the mean 40 bp above the forward. All of it must be on the grid."""
    F, T = 7906.26, 0.50434
    s = moments(*density(F, T, (-0.074913, 0.163516, -0.302061, 0.015965, 0.522605)), F)
    assert abs(s["mass"] - 1) < 0.002 and abs(s["mean"] / F - 1) < 0.0015


def test_surface_2019_calendar_arbitrage_free():
    """Whole-surface calendar no-arbitrage on the 2019 snapshot: total variance must not fall with
    maturity at any strike two adjacent expiries both quote. Independent fits fell by up to 2.9e-3."""
    path = spxw_2019()
    slices, skipped = fit_surface(lambda e: load_cboe_csv(path, e), pd.read_csv(path)["expiration"].unique())
    assert len(slices) == 27 and [why.endswith("days < 7") for _, why in skipped] == [True] * 3
    drop, k, T1, T2 = calendar_violation(slices)
    assert drop <= 0, (drop, k, T1, T2)


def test_snapshot_download_is_verified():
    """The Cboe snapshot is not committed; download.py fetches it and must replace a corrupt copy."""
    import download
    tmp = os.path.join(os.path.dirname(download.PATH), "_check.csv")
    with open(tmp, "w") as f:
        f.write("corrupt")
    real, download.PATH = download.PATH, tmp
    try:
        path = download.spxw_2019()
        d = pd.read_csv(path)
    finally:
        download.PATH = real
        os.remove(tmp)
    assert path == tmp and d.shape == (10384, 12) and (d["quote_date"] == "2019-06-26").all()


def test_live_quotes_timed_and_closed_session_refused():
    """Yahoo delays quotes ~15 min, and after the close it keeps the last session's quotes (seen on
    Saturday 2026-10-17, discount factor 1.018). T runs from the newest trade; a closed session stops."""
    now = pd.Timestamp.now(tz="America/New_York")
    expiry = (now + pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    chain, _, _ = synthetic_chain(T=60 / 365.25)
    real = sys.modules.get("yfinance")

    def load(newest):
        side = lambda mid: pd.DataFrame({"strike": chain["strike"], "bid": mid - 0.005, "ask": mid + 0.005,
                                         "lastTradeDate": [newest - pd.Timedelta(minutes=m) for m in range(len(chain))]})
        calls, puts = side(chain["call_mid"]), side(chain["put_mid"])
        yahoo = types.SimpleNamespace(options=(expiry,), option_chain=lambda e: types.SimpleNamespace(calls=calls, puts=puts))
        sys.modules["yfinance"] = types.SimpleNamespace(Ticker=lambda ticker: yahoo)
        try:
            return load_chain("^SPX", expiry)
        finally:
            sys.modules["yfinance"] = real

    delayed = now - pd.Timedelta(minutes=15)
    _, T, _, asof = load(delayed)
    close = pd.Timestamp(expiry + " 16:00", tz="America/New_York")
    assert asof == delayed and abs(T - (close - delayed).total_seconds() / (365.25 * 86400)) < 1e-12
    try:
        load(now - pd.Timedelta(hours=37))                              # Friday close, read on Saturday
    except RuntimeError as e:
        assert "market is closed" in str(e)
    else:
        raise AssertionError("a closed session's quotes were used")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
