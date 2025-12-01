# risk_metrics.py
"""
Risk metrics and backtesting utilities:
- compute_var_es: Monte Carlo and historical VaR/ES
- bootstrap_ci: bootstrap confidence intervals for VaR/ES
- backtest_var: Kupiec POF test (unconditional coverage)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import math
from typing import Tuple, Dict, Any

def compute_var_es(returns: np.ndarray, alpha: float = 0.95, method: str = "mc") -> Dict[str, float]:
    """
    returns: 1-d array of portfolio returns (NOT prices) ordered per simulation path
    alpha: confidence level (e.g., 0.95 for 95% VaR), VaR is loss at alpha quantile
    method: 'mc' or 'historical' (mc and hist same for a returns array)
    Returns dict with var, es (expected shortfall)
    Note: returns should be *losses* if you want positive VaR; here we compute losses = -returns
    """
    r = np.asarray(returns)
    losses = -r  # positive loss
    # VaR at alpha (quantile)
    q = np.quantile(losses, alpha)
    # ES mean of tail beyond q
    tail = losses[losses >= q]
    es = tail.mean() if tail.size > 0 else q
    return {"VaR": float(q), "ES": float(es)}

def bootstrap_ci(returns: np.ndarray, alpha: float = 0.95, stat: str = "VaR", n_boot: int = 2000, seed: int = 2025) -> Tuple[float, float]:
    """
    Bootstrap CI for VaR or ES.
    """
    rng = np.random.default_rng(seed)
    n = len(returns)
    stats = []
    for _ in range(n_boot):
        sample = rng.choice(returns, size=n, replace=True)
        out = compute_var_es(sample, alpha=alpha)
        stats.append(out[stat])
    lower = np.percentile(stats, 2.5)
    upper = np.percentile(stats, 97.5)
    return lower, upper

def kupiec_pof(violations: int, n_obs: int, alpha: float = 0.95) -> Dict[str, float]:
    """
    Kupiec proportion of failures (POF) test
    violations: number of times loss > VaR in out-of-sample
    n_obs: number of observations in backtest
    alpha: VaR level (e.g., 0.95)
    Returns LR statistic and p-value
    """
    from math import log, exp
    p_hat = violations / n_obs
    if p_hat == 0:
        # adjust slightly to avoid log(0)
        p_hat = 1e-12
    # Likelihood ratio
    LR = -2 * ( (n_obs - violations)*math.log((1-alpha)/(1-p_hat)) + violations*math.log(alpha/p_hat) )
    # p-value from chi-sq(1)
    from scipy.stats import chi2
    pval = 1 - chi2.cdf(LR, df=1)
    return {"LR": LR, "pvalue": pval, "p_hat": p_hat}

# Quick helper to convert portfolio *prices* to returns
def prices_to_returns(prices: np.ndarray) -> np.ndarray:
    # simple total return from initial price to terminal
    # if terminal prices array: returns = (P_T - P_0)/P_0
    # Here assume portfolio started at P0 = mean? better to pass P0 explicitly
    P0 = np.mean(prices) if prices.size>0 else 1.0
    returns = (prices - P0) / (P0 + 1e-16)
    return returns

# Quick test
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--quick-test", action="store_true")
    args = p.parse_args()
    if args.quick_test:
        # synthetic returns: normal with mean 0.001 std 0.02
        rng = np.random.default_rng(123)
        rets = rng.normal(0.001, 0.02, size=100000)
        out = compute_var_es(rets, alpha=0.95)
        print("VaR, ES (95%)", out)
        lo, hi = bootstrap_ci(rets, alpha=0.95, stat="VaR", n_boot=500, seed=123)
        print("Bootstrap VaR CI:", (lo,hi))
