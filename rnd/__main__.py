"""python -m rnd [TICKER] [--expiry YYYY-MM-DD] | python -m rnd --demo"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .core import analyze, svi
from .data import heston_pdf, load_chain, synthetic_chain

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


def main():
    ap = argparse.ArgumentParser(description="Market-implied probability distribution from option prices.")
    ap.add_argument("ticker", nargs="?", default="^SPX")
    ap.add_argument("--expiry", help="YYYY-MM-DD; default first expiry >= 30 days out")
    ap.add_argument("--demo", action="store_true", help="offline: noisy Heston quotes with known truth")
    ap.add_argument("--out", default="rnd.png")
    a = ap.parse_args()

    if a.demo:
        chain, T, _ = synthetic_chain()
        r = analyze(chain, T)
        title = "Demo: noisy Heston quotes (true density known)"
        truth = heston_pdf(r["density"][0], 100.0, T)
        report(r, title)
        Kq, q = r["density"]
        print(f"  error vs truth: total variation {0.5 * np.abs(q - truth).sum() * (Kq[1] - Kq[0]):.2%}")
    else:
        chain, T, expiry = load_chain(a.ticker, a.expiry)
        r = analyze(chain, T)
        title, truth = f"{a.ticker} options expiring {expiry}", None
        report(r, title)
    plot(r, title, a.out, truth)


if __name__ == "__main__":
    main()
