"""
german-credit-risk — a methodology-first credit risk analysis & scoring pipeline.

IMPORTANT: the source dataset shipped with this project has no ground-truth
default/repayment label. See `config.yaml` -> `target` and `src/proxy_scoring.py`
for how this package handles that honestly. Do not quote any metric produced
under `target.mode: proxy` as real-world predictive performance.
"""

__version__ = "0.1.0"
