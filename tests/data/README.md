`spxw_20190626.csv` is not committed. `python tests/data/download.py` fetches it (the tests do this
automatically) and checks its SHA-256.

What it is: Cboe SPXW (S&P 500 weekly, European, PM-settled) option quotes, a 15:45 New York snapshot
on 2019-06-26 covering every expiry, in Cboe DataShop's "Option EOD Summary" format.

Where it comes from: the open-source volkit project, https://github.com/sitmo/volkit, file
`volkit/datasets/data/spxw20190626.csv` at commit eb6c501. volkit's code is BSD 3-Clause; the
underlying quotes are Cboe market data, sold by Cboe DataShop. volkit states no Cboe licence for the
file, so this repository downloads it rather than redistributing it.
