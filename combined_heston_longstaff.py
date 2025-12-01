# combined_heston_longstaff.py

import numpy as np
from tqdm import tqdm

# Module 3: Heston stochastic volatility


def simulate_heston(S0, r, kappa, theta, sigma, rho, v0, T, N, M):
    dt = T / N
    S = np.zeros((M, N + 1))
    v = np.zeros((M, N + 1))
    S[:, 0] = S0
    v[:, 0] = v0

    for t in range(1, N + 1):
        Z1 = np.random.normal(size=M)
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.normal(size=M)
        v[:, t] = np.abs(v[:, t-1] + kappa*(theta - v[:, t-1])*dt + sigma*np.sqrt(v[:, t-1]*dt)*Z2)
        S[:, t] = S[:, t-1]*np.exp((r - 0.5*v[:, t-1])*dt + np.sqrt(v[:, t-1]*dt)*Z1)
    return S, v


# Module 4: Longstaff–Schwartz

def american_option_longstaff_schwartz(S, K, r, dt, option_type='put', degree=2):
    M, N = S.shape
    N -= 1
    if option_type == 'put':
        payoff = np.maximum(K - S[:, -1], 0)
    else:
        payoff = np.maximum(S[:, -1] - K, 0)

    cashflow = payoff.copy()

    for t in range(N-1, 0, -1):
        X = S[:, t]
        Y = cashflow * np.exp(-r*dt)
        # regression basis
        coeffs = np.polyfit(X, Y, degree)
        continuation = np.polyval(coeffs, X)
        if option_type == 'put':
            exercise = np.maximum(K - X, 0)
        else:
            exercise = np.maximum(X - K, 0)
        cashflow = np.where(exercise > continuation, exercise, cashflow*np.exp(-r*dt))
    return np.mean(cashflow)*np.exp(-r*dt)


# Parameters

S0 = 100      # initial stock price
K = 100       # strike price
r = 0.05      # risk-free rate
T = 1         # maturity in years
N = 50        # time steps
M = 10000     # number of paths

# Heston parameters
kappa = 2.0
theta = 0.05
sigma = 0.1
rho = -0.7
v0 = 0.05

# Run simulation

print("Simulating Heston paths...")
S_paths, v_paths = simulate_heston(S0, r, kappa, theta, sigma, rho, v0, T, N, M)
dt = T / N

print("Pricing American put option using Longstaff-Schwartz...")
american_put_price = american_option_longstaff_schwartz(S_paths, K, r, dt, option_type='put')
print(f"American Put Price: {american_put_price:.4f}")
