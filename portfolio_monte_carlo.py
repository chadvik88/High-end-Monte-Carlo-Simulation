# portfolio_monte_carlo.py
"""
Portfolio Monte Carlo module
- simulate_portfolio_gbm: correlated GBM for many assets
- supports Cholesky and PCA reduction for high-dim speed
- deterministic RNG via SeedSequence (seed input)
- memory-aware chunking for very large sims
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class PortfolioConfig:
    n_assets: int = 100
    n_paths: int = 200_000
    n_steps: int = 1            # terminal price simulation (1 step) by default; set >1 for paths
    S0: float = 100.0
    mu: float = 0.05            # expected return (drift)
    sigma: float = 0.2          # per-asset baseline vol (if scalar)
    corr_matrix: Optional[np.ndarray] = None
    vol_vector: Optional[np.ndarray] = None
    seed: Optional[int] = 2025
    chunk: int = 50_000         # number of paths per chunk to memory control
    use_pca: bool = False       # use PCA dimension-reduction of covariance (keeps top k factors)
    pca_k: int = 10             # number of principal factors to keep if use_pca True
    portfolio_weights: Optional[np.ndarray] = None  # shape (n_assets,)
    discount: bool = True       # discount payoffs by exp(-r*T) at portfolio level if needed
    r: float = 0.01
    T: float = 1.0

    def validate(self):
        assert self.n_assets >= 1
        assert self.n_paths >= 1
        assert self.chunk >= 1
        if self.vol_vector is not None:
            assert len(self.vol_vector) == self.n_assets
        if self.corr_matrix is not None:
            assert self.corr_matrix.shape == (self.n_assets, self.n_assets)
        if self.portfolio_weights is not None:
            assert len(self.portfolio_weights) == self.n_assets

def _make_cov_matrix(cfg: PortfolioConfig) -> np.ndarray:
    # Build covariance using corr_matrix and vol_vector or scalar sigma
    if cfg.corr_matrix is None:
        # simple identity correlation
        corr = np.eye(cfg.n_assets)
    else:
        corr = cfg.corr_matrix
    if cfg.vol_vector is None:
        vols = np.full(cfg.n_assets, cfg.sigma)
    else:
        vols = np.asarray(cfg.vol_vector)
    cov = (vols[:, None] * vol_vector_to_row(vols)) * corr  # vol_i * vol_j * corr_ij
    # More numerically stable: cov = np.outer(vols, vols) * corr
    cov = np.outer(vols, vols) * corr
    return cov

def vol_vector_to_row(vols: np.ndarray):
    # helper used above in older style - kept for readability safety
    return vols[None, :]

def _cholesky_safe(mat: np.ndarray) -> np.ndarray:
    # numerically stable cholesky: jitter if needed
    try:
        L = np.linalg.cholesky(mat)
    except np.linalg.LinAlgError:
        # add tiny jitter
        eps = 1e-10
        jitter = np.eye(mat.shape[0]) * eps
        attempt = 0
        while attempt < 10:
            try:
                L = np.linalg.cholesky(mat + jitter)
                break
            except np.linalg.LinAlgError:
                eps *= 10
                jitter = np.eye(mat.shape[0]) * eps
                attempt += 1
        else:
            # as final fallback, use eigen-decomp
            vals, vecs = np.linalg.eigh(mat)
            vals[vals < 0] = 0.0
            L = vecs @ np.diag(np.sqrt(vals))
    return L

def _pca_factors(cov: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    # returns factor_loadings (n_assets x k) and residual_cov_diag (n_assets)
    vals, vecs = np.linalg.eigh(cov)  # ascending
    # take top k
    idx = np.argsort(vals)[::-1]
    vals_sorted = vals[idx]
    vecs_sorted = vecs[:, idx]
    k = min(k, len(vals_sorted))
    top_vals = vals_sorted[:k]
    top_vecs = vecs_sorted[:, :k]
    loadings = top_vecs * np.sqrt(top_vals[None, :])  # n_assets x k
    # residual diag
    approx_cov = loadings @ loadings.T
    residual = np.diag(cov - approx_cov).copy()
    residual[residual < 0] = 0.0
    return loadings, residual

def simulate_portfolio_gbm(cfg: PortfolioConfig) -> Dict[str, Any]:
    """
    Simulate terminal portfolio value distribution.
    Returns dict with: portfolio_prices (np.array n_paths), optionally full asset prices (if small)
    Memory: streams in chunks.
    """
    cfg.validate()
    # build cov
    if cfg.corr_matrix is None and cfg.vol_vector is None:
        cov = np.outer(np.full(cfg.n_assets, cfg.sigma), np.full(cfg.n_assets, cfg.sigma)) * np.eye(cfg.n_assets)
    else:
        if cfg.corr_matrix is None:
            corr = np.eye(cfg.n_assets)
        else:
            corr = cfg.corr_matrix
        vols = cfg.vol_vector if cfg.vol_vector is not None else np.full(cfg.n_assets, cfg.sigma)
        cov = np.outer(vols, vols) * corr

    # If PCA reduction requested
    if cfg.use_pca and cfg.pca_k < cfg.n_assets:
        loadings, residual_diag = _pca_factors(cov, cfg.pca_k)
        # We'll simulate via low-dim factors + independent residual noises.
        use_pca = True
    else:
        use_pca = False
        L = _cholesky_safe(cov)  # n_assets x n_assets

    # portfolio weights
    if cfg.portfolio_weights is None:
        weights = np.ones(cfg.n_assets) / cfg.n_assets
    else:
        weights = np.asarray(cfg.portfolio_weights)

    # seed control
    ss = np.random.SeedSequence(cfg.seed)
    rng = np.random.default_rng(ss.generate_state(1)[0])

    # simulate in chunks
    total = cfg.n_paths
    chunk = min(cfg.chunk, total)
    results = []
    remaining = total
    while remaining > 0:
        this = min(chunk, remaining)
        if use_pca:
            # simulate k-factor normals
            k = loadings.shape[1]
            Zf = rng.standard_normal(size=(this, k))
            factor_contrib = Zf @ loadings.T  # this x n_assets
            # residual noises per asset
            Zr = rng.standard_normal(size=(this, cfg.n_assets)) * np.sqrt(residual_diag)[None, :]
            asset_vol_noise = factor_contrib + Zr
        else:
            Z = rng.standard_normal(size=(this, cfg.n_assets))
            asset_vol_noise = Z @ L.T  # this x n_assets

        # convert noise to terminal prices via GBM formula
        # If vol per asset not uniform, our cov already encoded it: asset_vol_noise ~ multivariate normal increments
        drift = (cfg.mu - 0.5 * (np.diag(cov))) * cfg.T  # using diagonal of cov as var per asset
        # however diag(cov) might be vol^2; more simply, compute per-asset variance:
        per_asset_var = np.diag(cov)
        # For convert, we need per-asset sigma_i: sqrt(var)
        sigma_vec = np.sqrt(per_asset_var)
        # normalize noise to standard normals for each asset by dividing by sigma_vec, then multiply by sigma*sqrt(T)
        # But asset_vol_noise currently has same scale as cov (i.e., variance = per_asset_var)
        # Terminal price ST = S0_i * exp(mu*T - 0.5 sigma_i^2*T + sigma_i*sqrt(T)*Z_standard)
        # We can compute Z_standard = asset_vol_noise / sigma_vec
        Z_std = asset_vol_noise / (sigma_vec[None, :] + 1e-16)
        ST = cfg.S0 * np.exp((cfg.mu - 0.5 * sigma_vec**2) * cfg.T + sigma_vec[None, :] * np.sqrt(cfg.T) * Z_std)
        # portfolio value
        port_vals = ST @ weights
        if cfg.discount:
            port_vals = port_vals * np.exp(-cfg.r * cfg.T)
        results.append(port_vals)
        remaining -= this

    portfolio_prices = np.concatenate(results)
    return {"portfolio_prices": portfolio_prices, "cfg": asdict(cfg)}

# quick demo/test (if run as script)
if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser()
    p.add_argument("--quick-test", action="store_true")
    p.add_argument("--n", type=int, default=100_000)
    p.add_argument("--assets", type=int, default=100)
    args = p.parse_args()
    if args.quick_test:
        cfg = PortfolioConfig(n_assets=args.assets, n_paths=args.n, seed=12345, chunk=20000)
        out = simulate_portfolio_gbm(cfg)
        arr = out["portfolio_prices"]
        print("Simulated portfolio:", arr.mean(), arr.std(), "n=",arr.size)
        # Save quick histogram
        import matplotlib.pyplot as plt
        plt.hist(arr, bins=80)
        plt.title("Quick portfolio histogram")
        plt.show()
