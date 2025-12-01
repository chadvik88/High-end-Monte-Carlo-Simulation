# heston_longstaff_dashboard.py

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------------
# Reuse Heston + Longstaff-Schwartz from Module 3+4
# ----------------------------

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
        coeffs = np.polyfit(X, Y, degree)
        continuation = np.polyval(coeffs, X)
        if option_type == 'put':
            exercise = np.maximum(K - X, 0)
        else:
            exercise = np.maximum(X - K, 0)
        cashflow = np.where(exercise > continuation, exercise, cashflow*np.exp(-r*dt))
    return np.mean(cashflow)*np.exp(-r*dt)

# ----------------------------
# Parameters
# ----------------------------

S0 = 100
K = 100
r = 0.05
T = 1
N = 50
M = 5000  # smaller for faster plotting

kappa = 2.0
theta = 0.05
sigma = 0.1
rho = -0.7
v0 = 0.05

# ----------------------------
# Run simulation
# ----------------------------

S_paths, v_paths = simulate_heston(S0, r, kappa, theta, sigma, rho, v0, T, N, M)
dt = T / N
american_put_price = american_option_longstaff_schwartz(S_paths, K, r, dt, option_type='put')
payoff = np.maximum(K - S_paths[:, -1], 0)

print(f"American Put Price: {american_put_price:.4f}")

# ----------------------------
# Plotly Dashboard
# ----------------------------

fig = make_subplots(rows=2, cols=1, subplot_titles=("Sample Heston Stock Paths", "Payoff Distribution at Maturity"))

# Sample 50 paths
for i in range(50):
    fig.add_trace(go.Scatter(y=S_paths[i], mode='lines', line=dict(width=1), name=f'Path {i+1}'),
                  row=1, col=1)

# Histogram of payoffs
fig.add_trace(go.Histogram(x=payoff, nbinsx=50, name='Payoffs', marker_color='crimson'), row=2, col=1)

fig.update_layout(height=800, width=900, title_text=f"Heston Simulation & American Put Option (Price={american_put_price:.2f})",
                  showlegend=False)

fig.show()
