"""GitHub social preview image, 1280x640: python docs/social_preview.py -> docs/social_preview.png
Every number on it is computed here from the 2019 Cboe snapshot."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tests", "data")]

from download import spxw_2019  # noqa: E402
from rnd.__main__ import BLUE, GRAY, GRID, INK, MUTED, plt  # noqa: E402
from rnd.core import analyze  # noqa: E402
from rnd.data import load_cboe_csv  # noqa: E402

EXPIRY = "2019-07-26"
chain, T = load_cboe_csv(spxw_2019(), EXPIRY)
r = analyze(chain, T)
Kq, q = r["density"]
F, crash = r["F"], r["crash_level"]
lo, hi = F * 0.86, F * 1.07
view = (Kq > lo) & (Kq < hi)

fig = plt.figure(figsize=(12.8, 6.4), dpi=100, facecolor="white")
fig.text(0.05, 0.84, "What the options market\nimplies for the S&P 500", fontsize=34, weight="bold",
         color=INK, va="top")
fig.text(0.05, 0.50, f"Chance of a drop of more than {1 - crash / F:.1%}\nwithin {T * 365.25:.0f} days:",
         fontsize=22, color=MUTED, va="top")
fig.text(0.05, 0.33, f"options market {r['p_crash']:.1%}", fontsize=30, weight="bold", color=BLUE, va="top")
fig.text(0.05, 0.22, f"bell curve {r['p_crash_lognormal']:.1%}", fontsize=30, weight="bold", color=GRAY, va="top")
fig.text(0.05, 0.06, "github.com/aidaman01/implied-distribution", fontsize=18, color=INK)

ax = fig.add_axes([0.58, 0.20, 0.38, 0.62])
ax.plot(Kq[view], r["lognormal"][view], "--", color=GRAY, lw=2.5, label="bell curve (lognormal)")
ax.plot(Kq[view], q[view], color=BLUE, lw=3.5, label="implied by option prices")
tail = view & (Kq <= crash)
ax.fill_between(Kq[tail], q[tail], color=BLUE, alpha=0.3, lw=0)
ax.set_xlim(lo, hi)
ax.set_ylim(0, q[view].max() * 1.25)
ax.set_yticks([])
ax.set_xlabel("S&P 500 on expiry", fontsize=14, color=MUTED)
ax.tick_params(labelsize=12, colors=MUTED)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.legend(frameon=False, fontsize=13, loc="upper left")
fig.text(0.58, 0.06, f"Real Cboe SPX quotes, 26 June 2019; expiry {EXPIRY}", fontsize=13, color=MUTED)

out = os.path.join(ROOT, "docs", "social_preview.png")
fig.savefig(out, dpi=100, facecolor="white")
print(f"P(drop > {1 - crash / F:.2%}): market {r['p_crash']:.2%}, lognormal {r['p_crash_lognormal']:.2%} -> {out}")
