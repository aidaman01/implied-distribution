"""Option chains: live (yfinance) or synthetic Heston with an exactly known density."""
import numpy as np
import pandas as pd
from scipy.integrate import quad_vec


def _mid(df):
    good = (df["bid"] > 0) & (df["ask"] >= df["bid"])
    return np.where(good, (df["bid"] + df["ask"]) / 2, np.nan), good


def load_chain(ticker="^SPX", expiry=None, min_days=30):
    """Live chain from Yahoo. Returns (chain, T, expiry). ^SPX is European: the right
    instrument for Breeden-Litzenberger. SPY works but is American (small early-exercise bias)."""
    import yfinance as yf

    tk = yf.Ticker(ticker)
    now = pd.Timestamp.now(tz="America/New_York")
    if expiry is None:
        later = [e for e in tk.options if (pd.Timestamp(e, tz="America/New_York") - now).days >= min_days]
        if not later:
            raise RuntimeError(f"No {ticker} expiry at least {min_days} days out (got {tk.options!r}).")
        expiry = later[0]
    elif expiry not in tk.options:
        raise RuntimeError(f"{expiry} is not a listed {ticker} expiry. Listed: {', '.join(tk.options)}")
    ch = tk.option_chain(expiry)

    c, p = ch.calls.copy(), ch.puts.copy()
    c["call_mid"], c_ok = _mid(c)
    p["put_mid"], p_ok = _mid(p)
    if c_ok.mean() < 0.5 or p_ok.mean() < 0.5:
        raise RuntimeError("Most bid/ask quotes are empty (US market closed?). Run during "
                           "US hours, 9:30-16:00 New York time, for real two-sided quotes.")
    # SPX monthly and SPXW weekly contracts can share an expiry date and strike: one row per strike.
    c = c.groupby("strike", as_index=False)["call_mid"].mean()
    p = p.groupby("strike", as_index=False)["put_mid"].mean()
    chain = c.merge(p, on="strike", how="outer")

    close = pd.Timestamp(expiry + " 16:00", tz="America/New_York")
    T = (close - now).total_seconds() / (365.25 * 86400)
    return chain, T, expiry


def load_cboe_csv(path, expiry):
    """Cboe end-of-day file (columns quote_date, expiration, strike, option_type, bid_1545,
    ask_1545, ...), one row per strike and C/P. Quotes are a 15:45 New York snapshot.
    Returns (chain, T)."""
    d = pd.read_csv(path, encoding="utf-8-sig")
    d = d[d["expiration"] == expiry].rename(columns={"bid_1545": "bid", "ask_1545": "ask"})
    if d.empty:
        raise ValueError(f"no rows for expiry {expiry}; available: {sorted(pd.read_csv(path)['expiration'].unique())}")
    d["mid"] = _mid(d)[0]
    wide = d.pivot(index="strike", columns="option_type", values="mid")
    chain = pd.DataFrame({"strike": wide.index.to_numpy(float), "call_mid": wide["C"].to_numpy(),
                          "put_mid": wide["P"].to_numpy()})
    quote = pd.Timestamp(d["quote_date"].iloc[0] + " 15:45", tz="America/New_York")
    close = pd.Timestamp(expiry + " 16:00", tz="America/New_York")
    return chain, (close - quote).total_seconds() / (365.25 * 86400)


# Ground truth: Heston stochastic volatility, equity-index-like (negative spot/vol correlation).
HESTON = dict(v0=0.04, kappa=1.5, theta=0.05, xi=0.6, rho=-0.7)


def heston_cf(u, T, v0, kappa, theta, xi, rho):
    """E[exp(iu ln(S_T/F))], 'little trap' form (Albrecher et al. 2007)."""
    b = kappa - rho * xi * 1j * u
    d = np.sqrt(b ** 2 + xi ** 2 * (1j * u + u ** 2))
    g = (b - d) / (b + d)
    e = np.exp(-d * T)
    C = kappa * theta / xi ** 2 * ((b - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
    D = (b - d) / xi ** 2 * (1 - e) / (1 - g * e)
    return np.exp(C + D * v0)


def heston_call(F, K, T, p=HESTON):
    """Undiscounted call via Gil-Pelaez: C = F*P1 - K*P2."""
    k = np.log(np.asarray(K, float) / F)

    def P(shift):  # shift=-1j gives the share-measure probability P1
        f = lambda u: np.real(np.exp(-1j * u * k) * heston_cf(u + shift, T, **p) / (1j * u))
        return 0.5 + quad_vec(f, 1e-10, np.inf, epsabs=1e-12)[0] / np.pi

    return F * P(-1j) - K * P(0)


def heston_pdf(K, F, T, p=HESTON):
    """Exact density of S_T by Fourier inversion of the log-price characteristic function."""
    x = np.log(np.asarray(K, float) / F)
    f = lambda u: np.real(np.exp(-1j * u * x) * heston_cf(u, T, **p))
    return quad_vec(f, 0, np.inf, epsabs=1e-12)[0] / np.pi / K


def synthetic_chain(F=100.0, T=0.5, r=0.04, step=1.0, noise=0.005, tick=0.01, seed=0):
    """Heston quotes with relative noise and tick rounding, like a real screen.
    Quotes that round to zero are dropped (no bid). Returns (chain, T, DF)."""
    rng = np.random.default_rng(seed)
    K = np.arange(0.5 * F, 1.6 * F + step / 2, step)
    DF = np.exp(-r * T)
    call = heston_call(F, K, T)
    put = call - (F - K)

    def quote(px):
        px = DF * px * (1 + noise * rng.standard_normal(K.size))
        px = np.round(px / tick) * tick
        return np.where(px >= tick, px, np.nan)

    return pd.DataFrame({"strike": K, "call_mid": quote(call), "put_mid": quote(put)}), T, DF
