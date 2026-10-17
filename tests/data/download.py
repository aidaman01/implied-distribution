"""Download the 2019-06-26 Cboe SPXW quote snapshot used by the tests and the --csv examples.

The file is Cboe data, so it is not committed here. It is fetched from the open-source volkit
project at a pinned commit and checked against a SHA-256 hash.  Usage: python tests/data/download.py
"""
import hashlib
import os
import ssl
import urllib.request

URL = "https://raw.githubusercontent.com/sitmo/volkit/eb6c501/volkit/datasets/data/spxw20190626.csv"
SHA256 = "23c90a0dfe534b9467d19de2601abe7776e12ba462a1f668bed31fe8a03aa0d0"
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spxw_20190626.csv")


def spxw_2019():
    """Path to the snapshot; downloads it if it is missing or does not match the hash."""
    if os.path.exists(PATH):
        with open(PATH, "rb") as f:
            if hashlib.sha256(f.read()).hexdigest() == SHA256:
                return PATH
    try:  # python.org builds on macOS ship without CA certificates; certifi comes with yfinance
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = None
    data = urllib.request.urlopen(URL, timeout=60, context=context).read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise RuntimeError(f"SHA-256 mismatch for {URL}; refusing to use it")
    with open(PATH, "wb") as f:
        f.write(data)
    return PATH


if __name__ == "__main__":
    print(spxw_2019())
