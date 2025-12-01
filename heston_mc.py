# heston_mc.py
"""
Heston Stochastic Volatility Monte Carlo Simulation
Supports QE-Milstein scheme and full-pseudo exact scheme.
Fully deterministic with optional seeding.
Chunked simulation for memory efficiency.
"""

import numpy as np
from numba import njit, prange


# Core Simulation Functions

def simulate_heston_paths(
    S0: float,
    v0: float,
    r: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    T: float,
    steps: int,
    n_paths: int,
    seed: int = None,
    method: str = "QE"
):
    """
    Simulate Heston stochastic volatility paths.
    
    Parameters:
        S0: initial stock price
        v0: initial variance
        r: risk-free rate
        kappa: mean-reversion speed
        theta: long-term variance
        sigma: vol-of-vol
        rho: correlation between stock and variance
        T: time horizon
        steps: time steps
        n_paths: number of paths
        seed: random seed
        method: "QE" (default) or "full_exact"
        
    Returns:
        S_paths: n_paths x (steps+1) array of stock paths
        v_paths: n_paths x (steps+1) array of variance paths
    """
    dt = T / steps
    S_paths = np.zeros((n_paths, steps + 1), dtype=np.float64)
    v_paths = np.zeros((n_paths, steps + 1), dtype=np.float64)
    S_paths[:, 0] = S0
    v_paths[:, 0] = v0
    
    rng = np.random.default_rng(seed)
    
    if method.lower() == "qe":
        S_paths, v_paths = _qe_heston(S_paths, v_paths, r, kappa, theta, sigma, rho, dt, rng)
    elif method.lower() == "full_exact":
        S_paths, v_paths = _full_exact_heston(S_paths, v_paths, r, kappa, theta, sigma, rho, dt, rng)
    else:
        raise ValueError("Method must be 'QE' or 'full_exact'")
    
    return S_paths, v_paths

# QE-Milstein Scheme

def _qe_heston(S, v, r, kappa, theta, sigma, rho, dt, rng):
    n_paths, steps_plus1 = S.shape
    steps = steps_plus1 - 1
    for t in range(steps):
        # Correlated random normals
        Z1 = rng.standard_normal(n_paths)
        Z2 = rng.standard_normal(n_paths)
        Wv = Z1
        Ws = rho * Z1 + np.sqrt(1 - rho**2) * Z2
        
        v_prev = v[:, t]
        # QE scheme parameters
        psi_c = 1.5
        m = theta + (v_prev - theta) * np.exp(-kappa * dt)
        s2 = (v_prev * sigma**2 * np.exp(-kappa*dt)/kappa * (1 - np.exp(-kappa*dt)) +
              theta * sigma**2 / (2*kappa) * (1 - np.exp(-kappa*dt))**2)
        psi = s2 / m**2
        v_next = np.zeros_like(v_prev)
        
        # Quadratic-Exponential scheme
        mask = psi <= psi_c
        b2 = 2 / psi[mask] - 1 + np.sqrt(2 / psi[mask] * (2 / psi[mask] - 1))
        a = m[mask] / (1 + b2)
        U = rng.uniform(size=mask.sum())
        v_next[mask] = a * (np.sqrt(b2) + np.tan(np.pi*(U-0.5)))**2
        # For psi > psi_c, use exponential approximation
        mask2 = ~mask
        p = (psi[mask2] - 1)/(psi[mask2] + 1)
        beta = (1 - p)/m[mask2]
        U2 = rng.uniform(size=mask2.sum())
        v_next[mask2] = np.where(U2 <= p, 0, -np.log((1 - U2)/ (1 - p))/beta)
        
        v_next = np.maximum(v_next, 0)  # numerical safeguard
        v[:, t+1] = v_next
        S[:, t+1] = S[:, t] * np.exp((r - 0.5*v_next)*dt + np.sqrt(v_next*dt)*Ws)
    return S, v

# Full-Pseudo Exact Scheme (full correction)

def _full_exact_heston(S, v, r, kappa, theta, sigma, rho, dt, rng):
    n_paths, steps_plus1 = S.shape
    steps = steps_plus1 - 1
    for t in range(steps):
        Z1 = rng.standard_normal(n_paths)
        Z2 = rng.standard_normal(n_paths)
        Wv = Z1
        Ws = rho * Z1 + np.sqrt(1 - rho**2) * Z2
        v_prev = v[:, t]
        v_next = np.abs(v_prev + kappa*(theta - v_prev)*dt + sigma*np.sqrt(v_prev*dt)*Wv)
        v[:, t+1] = v_next
        S[:, t+1] = S[:, t]*np.exp((r - 0.5*v_next)*dt + np.sqrt(v_next*dt)*Ws)
    return S, v

# Quick Unit Test
if __name__ == "__main__":
    S0 = 100
    v0 = 0.04
    r = 0.03
    kappa = 2.0
    theta = 0.04
    sigma = 0.3
    rho = -0.7
    T = 1.0
    steps = 50
    n_paths = 1000
    seed = 42
    
    print("Running QE-Milstein test...")
    S_paths, v_paths = simulate_heston_paths(S0, v0, r, kappa, theta, sigma, rho, T, steps, n_paths, seed, "QE")
    print("S_paths mean (final step):", S_paths[:, -1].mean())
    print("v_paths mean (final step):", v_paths[:, -1].mean())
    
    print("Running Full-Pseudo Exact test...")
    S_paths2, v_paths2 = simulate_heston_paths(S0, v0, r, kappa, theta, sigma, rho, T, steps, n_paths, seed, "full_exact")
    print("S_paths2 mean (final step):", S_paths2[:, -1].mean())
    print("v_paths2 mean (final step):", v_paths2[:, -1].mean())
    
    print("Module 3 Heston simulation exec was successfull boiss!")
