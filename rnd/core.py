"""Option prices -> market-implied probability distribution.

Pipeline: put-call parity forward -> Black-76 implied vols -> SVI smile fit
-> Breeden-Litzenberger density  q(K) = d^2 C / dK^2  (undiscounted call).
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.special import ndtr


def black(F, K, T, vol, call=True):
    """Undiscounted Black-76 price (multiply by the discount factor for the market price)."""
    F, K, vol, call = np.broadcast_arrays(np.asarray(F, float), np.asarray(K, float),
                                          np.asarray(vol, float), np.asarray(call, bool))
    sd = vol * np.sqrt(T)
    d1 = np.log(F / K) / sd + sd / 2
    c = F * ndtr(d1) - K * ndtr(d1 - sd)
    return np.where(call, c, c - (F - K))


def implied_vol(price, F, K, T, call, lo=1e-4, hi=5.0, iters=100):
    """Vectorised bisection on Black-76. NaN where the price breaks no-arbitrage bounds."""
    price, K, call = np.broadcast_arrays(np.asarray(price, float), np.asarray(K, float),
                                         np.asarray(call, bool))
    intrinsic = np.where(call, np.maximum(F - K, 0), np.maximum(K - F, 0))
    ok = (price > intrinsic) & (price < np.where(call, F, K))
    a, b = np.full(price.shape, lo), np.full(price.shape, hi)
    for _ in range(iters):
        mid = (a + b) / 2
        high = black(F, K, T, mid, call) > price
        a, b = np.where(high, a, mid), np.where(high, mid, b)
    iv = (a + b) / 2
    return np.where(ok & (iv > lo * 1.01) & (iv < hi * 0.99), iv, np.nan)


def parity_forward(K, call_mid, put_mid):
    """C - P = DF*F - DF*K. Weighted regression over strikes quoted on both sides -> (F, DF).
    Quote error grows with price, so weight by 1/(C+P): near-the-money strikes dominate.
    No rate or dividend assumption needed."""
    m = np.isfinite(call_mid) & np.isfinite(put_mid)
    slope, intercept = np.polyfit(K[m], (call_mid - put_mid)[m], 1, w=1 / (call_mid + put_mid)[m])
    DF = -slope
    return intercept / DF, DF


def svi(k, a, b, rho, m, s):
    """Gatheral raw SVI total implied variance w(k), k = ln(K/F)."""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s ** 2))


def fit_svi(k, iv, T):
    """Robust least squares in implied-vol space, multi-start. Returns (a, b, rho, m, s)."""
    w = iv ** 2 * T
    lb = [-w.max(), 1e-6, -0.999, k.min() - 0.5, 1e-4]
    ub = [w.max(), 2.0, 0.999, k.max() + 0.5, 2.0]

    def resid(p):
        return np.sqrt(np.maximum(svi(k, *p), 1e-12) / T) - iv

    best = None
    for rho0 in (-0.7, 0.0):
        for m0 in (-0.1, 0.0, 0.1):
            for s0 in (0.05, 0.2):
                x0 = np.clip([w.min() / 2, 0.1, rho0, m0, s0], lb, ub)
                # soft_l1: residuals beyond ~0.5 vol pt count linearly, so stale wing quotes can't dominate
                r = least_squares(resid, x0, bounds=(lb, ub), loss="soft_l1", f_scale=0.005)
                if best is None or r.cost < best.cost:
                    best = r
    return tuple(best.x)


def density(F, T, params, n=4001, width=8.0):
    """Breeden-Litzenberger on the smooth SVI call curve. Returns strikes K and density q(K)."""
    sd = np.sqrt(svi(0.0, *params))  # ATM total std dev
    K = np.linspace(F * np.exp(-width * sd), F * np.exp(width * sd), n)
    C = black(F, K, T, np.sqrt(svi(np.log(K / F), *params) / T))
    h = K[1] - K[0]
    q = (C[:-2] - 2 * C[1:-1] + C[2:]) / h ** 2
    return K[1:-1], q


def naive_density(K, C, DF):
    """Finite-difference B-L on raw quotes (non-uniform grid). Shows why smoothing is needed."""
    h1, h2 = np.diff(K)[:-1], np.diff(K)[1:]
    d2 = 2 * ((C[2:] - C[1:-1]) / h2 - (C[1:-1] - C[:-2]) / h1) / (h1 + h2)
    return K[1:-1], d2 / DF


def lognormal_pdf(K, F, vol, T):
    s = vol * np.sqrt(T)
    return np.exp(-(np.log(K / F) + s ** 2 / 2) ** 2 / (2 * s ** 2)) / (K * s * np.sqrt(2 * np.pi))


def moments(K, q, F):
    """Mass, mean, and shape of the log-return ln(S_T/F) under q."""
    h = K[1] - K[0]
    p = q * h
    mass = p.sum()
    p = p / mass
    x = np.log(K / F)
    mu = (p * x).sum()
    var = (p * (x - mu) ** 2).sum()
    return {
        "mass": mass,
        "mean": (p * K).sum(),
        "vol": np.sqrt(var),
        "skew": (p * (x - mu) ** 3).sum() / var ** 1.5,
        "ex_kurt": (p * (x - mu) ** 4).sum() / var ** 2 - 3,
        "min_q": q.min(),
    }


def analyze(chain, T):
    """chain: DataFrame with columns strike, call_mid, put_mid (discounted market mids)."""
    chain = chain.sort_values("strike")
    K = chain["strike"].to_numpy(float)
    cm, pm = chain["call_mid"].to_numpy(float), chain["put_mid"].to_numpy(float)
    F, DF = parity_forward(K, cm, pm)

    # Out-of-the-money side only: most liquid, and parity makes call/put IV identical.
    use_call = K >= F
    otm = np.where(use_call, cm, pm)
    iv = implied_vol(otm / DF, F, K, T, use_call)
    ok = np.isfinite(iv)
    k, iv = np.log(K[ok] / F), iv[ok]

    params = fit_svi(k, iv, T)
    Kq, q = density(F, T, params)

    # Raw call curve (OTM puts converted by parity) for the naive comparison.
    C_raw = np.where(use_call, cm, pm + DF * (F - K))[ok]
    Kn, qn = naive_density(K[ok], C_raw, DF)

    atm_vol = np.sqrt(svi(0.0, *params) / T)
    crash = F * np.exp(-2 * atm_vol * np.sqrt(T))  # a 2-standard-deviation drop at ATM vol
    return {
        "F": F, "DF": DF, "T": T, "params": params, "atm_vol": atm_vol,
        "smile": (K[ok], iv), "density": (Kq, q), "naive": (Kn, qn),
        "lognormal": lognormal_pdf(Kq, F, atm_vol, T),
        "fit_rmse": np.sqrt(np.mean((np.sqrt(svi(k, *params) / T) - iv) ** 2)),
        "stats": moments(Kq, q, F),
        "crash_level": crash,
        "p_crash": q[Kq < crash].sum() * (Kq[1] - Kq[0]),
        "p_crash_lognormal": ndtr(-2 + atm_vol * np.sqrt(T) / 2),
    }
