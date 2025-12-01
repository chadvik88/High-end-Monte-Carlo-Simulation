# longstaff_schwartz.py

import numpy as np
from numpy.polynomial.polynomial import Polynomial

def price_american_option(S_paths, K, r, option_type="put", degree=2):
    """
    Price an American option using Longstaff-Schwartz method.
    
    Parameters:
        S_paths : ndarray
            Simulated underlying asset paths (n_paths x n_steps+1)
        K : float
            Strike price
        r : float
            Risk-free rate
        option_type : str
            "put" or "call"
        degree : int
            Degree of polynomial regression for continuation value
        
    Returns:
        float: Estimated American option price
    """
    dt = 1 / (S_paths.shape[1] - 1)
    n_paths, n_steps = S_paths.shape
    n_steps -= 1  # because paths include time 0
    
    # Payoff at maturity
    if option_type == "put":
        payoff = np.maximum(K - S_paths[:, -1], 0)
    else:
        payoff = np.maximum(S_paths[:, -1] - K, 0)
    
    # Work backward in time
    for t in reversed(range(1, n_steps)):
        S_t = S_paths[:, t]
        if option_type == "put":
            immediate_exercise = np.maximum(K - S_t, 0)
        else:
            immediate_exercise = np.maximum(S_t - K, 0)
        
        # Only consider paths where immediate exercise > 0
        in_the_money = immediate_exercise > 0
        if np.any(in_the_money):
            X = S_t[in_the_money]
            Y = payoff[in_the_money] * np.exp(-r * dt)
            
            # Regression to estimate continuation value
            coefs = np.polyfit(X, Y, degree)
            continuation_value = np.polyval(coefs, X)
            
            # Decide whether to exercise
            exercise = immediate_exercise[in_the_money] > continuation_value
            payoff[in_the_money] = np.where(exercise, immediate_exercise[in_the_money], payoff[in_the_money] * np.exp(-r * dt))
        else:
            # Discount all payoffs if no path is ITM
            payoff *= np.exp(-r * dt)
    
    # Discount to time 0
    price = np.mean(payoff) * np.exp(-r * dt)
    return price


# Quick test harness
if __name__ == "__main__":
    # Example: 3 paths, 5 steps (tiny for test)
    S_paths = np.array([[100, 102, 101, 105, 103, 104],
                        [100, 98, 97, 96, 95, 94],
                        [100, 101, 102, 100, 99, 101]])
    
    K = 100
    r = 0.03
    
    put_price = price_american_option(S_paths, K, r, option_type="put")
    call_price = price_american_option(S_paths, K, r, option_type="call")
    
    print(f"American Put Price: {put_price}")
    print(f"American Call Price: {call_price}")
