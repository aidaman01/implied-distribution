"""python -m rnd [TICKER] [--expiry YYYY-MM-DD] | python -m rnd --demo | python -m rnd [TICKER | --csv FILE] --surface"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core import analyze, calendar_violation, svi
from .data import heston_pdf, load_cboe_csv, load_chain, synthetic_chain

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"


def report(r, title):
    s = r["stats"]
    print(f"\n{title}")
    print(f"  forward {r['F']:.2f}   discount factor {r['DF']:.5f}   T {r['T']:.3f}y")
    print(f"  ATM vol {r['atm_vol']:.1%}   SVI fit RMSE {r['fit_rmse'] * 100:.2f} vol pts")
    print(f"  density mass {s['mass']:.4f}   mean {s['mean']:.2f}   min q {s['min_q']:.2e}")
    print(f"  log-return vol {s['vol']:.1%}   skew {s['skew']:.2f}   excess kurtosis {s['ex_kurt']:.2f}")
    print(f"  P(2-sd crash, S_T < {r['crash_level']:.2f}): market {r['p_crash']:.2%}"
          f"  vs lognormal {r['p_crash_lognormal']:.2%}")
    neg = (r["naive"][1] < 0).mean()
    print(f"  naive finite-difference density negative at {neg:.0%} of strikes")


def plot(r, title, out, truth=None):
    F, T = r["F"], r["T"]
    K, iv = r["smile"]
    Kq, q = r["density"]
    Kn, qn = r["naive"]
    lo, hi = np.interp([0.002, 0.998], np.cumsum(q) / q.sum(), Kq)

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for a in (a1, a2):
        a.grid(color=GRID, lw=0.8)
        a.spines[["top", "right"]].set_visible(False)

    kk = np.linspace(K.min(), K.max(), 400)
    a1.plot(K, iv * 100, "o", ms=4, color=BLUE, alpha=0.45, label="market (OTM mids)")
    a1.plot(kk, np.sqrt(svi(np.log(kk / F), *r["params"]) / T) * 100, color=BLUE, lw=2, label="SVI fit")
    a1.axvline(F, color=GRAY, lw=1, ls=":")
    a1.set(xlabel="strike", ylabel="implied vol (%)", title="Volatility smile")

    a2.plot(Kn, qn, "o", ms=4, color=ORANGE, alpha=0.7, label="naive finite differences")
    a2.plot(Kq, r["lognormal"], "--", color=GRAY, lw=1.5, label="lognormal at ATM vol")
    if truth is not None:
        a2.plot(Kq, truth, color=INK, lw=1, label="true density")
    a2.plot(Kq, q, color=BLUE, lw=2, label="implied (SVI + Breeden-Litzenberger)")
    a2.axhline(0, color=MUTED, lw=0.8)
    a2.set(xlim=(lo, hi), ylim=(min(0, np.percentile(qn, 5)) * 1.2, q.max() * 1.6),
           xlabel="price at expiry", ylabel="risk-neutral density", title="Market-implied distribution")
    for a in (a1, a2):
        a.legend(frameon=False, fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"  chart -> {out}")


def fit_surface(load, expiries):
    """SVI on every expiry in order of maturity; load(e) -> (chain, T). Each fit is kept from falling
    below the previous one (calendar penalty). Skips expiries under 7 days and fits that fail the
    usual checks. Returns slices (T, k_min, k_max, params, expiry) sorted by T, and [(expiry, why)]."""
    slices, skipped = [], []
    for e in sorted(expiries):
        try:
            chain, T = load(e)
            if T * 365.25 < 7:
                skipped.append((e, f"{T * 365.25:.1f} days < 7"))
                continue
            r = analyze(chain, T, slices[-1][1:4] if slices else None)
        except Exception as x:
            skipped.append((e, f"{type(x).__name__}: {x}"))
            continue
        mass, drift = r["stats"]["mass"], r["stats"]["mean"] / r["F"] - 1
        if not (abs(mass - 1) < 0.005 and abs(drift) < 0.003):
            skipped.append((e, f"mass {mass:.4f}, mean vs forward {drift * 1e4:.0f} bp"))
            continue
        k = np.log(r["smile"][0] / r["F"])
        slices.append((T, k.min(), k.max(), r["params"], e))
    return slices, skipped


def plot_surface(slices, title, out, max_T=1.0):
    """Implied vol over a regular (ln K/F, T) grid: expiries up to max_T years, strikes quoted on every
    one of them (no extrapolation). Total variance is linear in T between expiries."""
    shown = [s for s in slices if s[0] <= max_T]
    Ts = np.array([s[0] for s in shown])
    lo, hi = max(s[1] for s in shown), min(s[2] for s in shown)
    print(f"  surface: {len(shown)} expiries, T {Ts[0]:.3f}-{Ts[-1]:.3f}y, ln(K/F) {lo:.3f} to {hi:.3f}")
    k, T = np.linspace(lo, hi, 80), np.linspace(Ts[0], Ts[-1], 80)
    w = np.array([svi(k, *s[3]) for s in shown])
    w = np.array([np.interp(T, Ts, w[:, j]) for j in range(k.size)]).T
    kk, TT = np.meshgrid(k, T)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(kk, TT, np.sqrt(w / TT) * 100, cmap="viridis", linewidth=0, antialiased=True)
    ax.set(xlabel="log-moneyness ln(K/F)", ylabel="time to expiry (years)", zlabel="implied vol (%)")
    ax.view_init(elev=25, azim=-125)
    ax.set_title(title, loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"  chart -> {out}")


def main():
    ap = argparse.ArgumentParser(description="Market-implied probability distribution from option prices.")
    ap.add_argument("ticker", nargs="?", default="^SPX")
    ap.add_argument("--expiry", help="YYYY-MM-DD; default first expiry >= 30 days out")
    ap.add_argument("--demo", action="store_true", help="offline: noisy Heston quotes with known truth")
    ap.add_argument("--csv", help="offline: Cboe end-of-day quote file (needs --expiry), e.g. tests/data/spxw_20190626.csv")
    ap.add_argument("--surface", action="store_true", help="SVI on every expiry -> 3D implied vol surface")
    ap.add_argument("--out", help="chart file (default rnd.png, or surface.png with --surface)")
    a = ap.parse_args()

    out = a.out or ("surface.png" if a.surface else "rnd.png")
    if a.surface:
        if a.csv:
            expiries = sorted(pd.read_csv(a.csv)["expiration"].unique())
            load = lambda e: load_cboe_csv(a.csv, e)
            title = f"SPX implied vol surface, Cboe quotes {pd.read_csv(a.csv, nrows=1)['quote_date'].iloc[0]} 15:45 New York"
        else:
            import yfinance as yf
            expiries = yf.Ticker(a.ticker).options
            load = lambda e: load_chain(a.ticker, e)[:2]
            title = f"{a.ticker} implied vol surface, {pd.Timestamp.now(tz='America/New_York'):%Y-%m-%d %H:%M} New York"
        slices, skipped = fit_surface(load, expiries)
        print(f"\n{title}\n  {len(slices)} expiries fitted, {len(skipped)} skipped")
        for e, why in skipped:
            print(f"    skipped {e}: {why}")
        drop, k, T1, T2 = calendar_violation(slices)
        if drop > 0:
            print(f"  calendar arbitrage: total variance falls by {drop:.2e} at ln(K/F) = {k:.3f}"
                  f" between T = {T1:.3f}y and {T2:.3f}y")
        else:
            print(f"  calendar no-arbitrage holds on every adjacent pair (closest: w rises by {-drop:.2e})")
        plot_surface(slices, title, out)
        return
    if a.csv:
        if not a.expiry:
            ap.error("--csv needs --expiry")
        chain, T = load_cboe_csv(a.csv, a.expiry)
        r = analyze(chain, T)
        quoted = pd.read_csv(a.csv, nrows=1)["quote_date"].iloc[0]
        title, truth = f"SPX options expiring {a.expiry}, real quotes {quoted} 15:45 New York", None
        report(r, title)
    elif a.demo:
        chain, T, _ = synthetic_chain()
        r = analyze(chain, T)
        title = "Demo: noisy Heston quotes (true density known)"
        truth = heston_pdf(r["density"][0], 100.0, T)
        report(r, title)
        Kq, q = r["density"]
        print(f"  error vs truth: total variation {0.5 * np.abs(q - truth).sum() * (Kq[1] - Kq[0]):.2%}")
    else:
        chain, T, expiry, asof = load_chain(a.ticker, a.expiry)
        r = analyze(chain, T)
        title, truth = f"{a.ticker} options expiring {expiry}, quotes as of {asof:%Y-%m-%d %H:%M} New York", None
        report(r, title)
    plot(r, title, out, truth)


if __name__ == "__main__":
    main()
