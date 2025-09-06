#!/usr/bin/env python3
"""
Option Pricing Script
Computes option prices for the EWMA-rBergomi model, 95% CI, market prices from yfinance, and relative error.
Generates a table for SPY, META, and BTC-USD with specified strikes.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from scipy.special import gamma
from typing import Dict, Tuple
import pickle
from datetime import datetime, timedelta

class EWMArBergoliModel:
    """Implementation of EWMA-driven rough Bergomi model with time-dependent Hurst parameter"""

    def __init__(self, V0: float = 0.02, nu: float = 0.3, rho: float = -0.7, epsilon: float = 0.01,
                 H_max: float = 0.5, lambda_ewma: float = 0.1, alpha: float = 0.2, beta: float = 0.1,
                 gamma: float = 0.5, theta_ref: float = 0.02, r: float = 0.05):
        self.V0 = V0
        self.nu = nu
        self.rho = rho
        self.epsilon = epsilon
        self.H_max = H_max
        self.lambda_ewma = lambda_ewma
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.theta_ref = theta_ref
        self.r = r
        self.sqrt_1_minus_rho2 = np.sqrt(1 - rho**2)

    def volterra_kernel(self, t: float, u: float, H_u: float) -> float:
        if u >= t or H_u <= 0:
            return 0.0
        with np.errstate(divide='ignore', invalid='ignore'):
            time_diff = t - u
            hurst_exp = H_u - 0.5
            gamma_term = gamma(H_u + 0.5)
            result = np.power(time_diff, hurst_exp) / gamma_term
            return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    def update_hurst_parameter(self, theta_t: float) -> float:
        ratio = theta_t / self.theta_ref
        H_raw = self.alpha * np.power(ratio, self.gamma) + self.beta
        return np.clip(H_raw, self.epsilon, self.H_max)

    def simulate_paths(self, S0: float, T: float, N: int, M: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dt = T / N
        t_grid = np.linspace(0, T, N+1)
        S_paths = np.zeros((M, N+1))
        sigma_paths = np.zeros((M, N+1))
        H_paths = np.zeros((M, N+1))
        V_paths = np.zeros((M, N+1))
        theta_paths = np.zeros((M, N+1))
        S_paths[:, 0] = S0
        sigma_paths[:, 0] = np.sqrt(self.V0)
        H_paths[:, 0] = self.beta
        theta_paths[:, 0] = self.V0
        dW = np.random.normal(0, np.sqrt(dt), (M, N))
        dW_perp = np.random.normal(0, np.sqrt(dt), (M, N))
        dZ = self.rho * dW + self.sqrt_1_minus_rho2 * dW_perp
        kernel_matrix = np.zeros((N, N))
        for n in range(N):
            for k in range(n):
                kernel_matrix[n, k] = self.volterra_kernel(t_grid[n+1], t_grid[k], self.beta)
        for n in range(N):
            V_paths[:, n+1] = np.sum(kernel_matrix[n, :n] * dZ[:, :n], axis=1)
            weights = self.lambda_ewma * np.exp(-self.lambda_ewma * (t_grid[n+1] - t_grid[:n+1]))
            weights = weights / np.sum(weights)
            theta_paths[:, n+1] = np.sum(weights * (sigma_paths[:, :n+1]**2) * dt, axis=1)
            H_paths[:, n+1] = np.array([self.update_hurst_parameter(theta_paths[m, n+1]) for m in range(M)])
            A_t = np.zeros(M)
            for m in range(M):
                A_t[m] = np.sum((kernel_matrix[n, :n+1] * dt)**2)
            sigma_paths[:, n+1] = np.sqrt(self.V0) * np.exp(self.nu * V_paths[:, n+1] - 0.5 * self.nu**2 * A_t)
            drift = self.r - 0.5 * sigma_paths[:, n]**2
            S_paths[:, n+1] = S_paths[:, n] * np.exp(drift * dt + sigma_paths[:, n] * dW[:, n])
        return S_paths, sigma_paths, H_paths

def compute_option_price(S_paths: np.ndarray, K: float, r: float, T: float) -> Dict:
    """Compute call option price and 95% CI using Monte Carlo simulation"""
    S_T = S_paths[:, -1]  # Terminal prices
    payoffs = np.maximum(S_T - K, 0)  # Call option payoffs
    discount_factor = np.exp(-r * T)
    option_price = discount_factor * np.mean(payoffs)
    std_err = np.std(payoffs) / np.sqrt(len(payoffs))
    ci_lower = option_price - 1.96 * std_err * discount_factor
    ci_upper = option_price + 1.96 * std_err * discount_factor
    return {
        'price': option_price,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }

def fetch_market_option_price(ticker: str, strike: float, expiration_days: int = 90) -> float:
    """Fetch market call option price from yfinance for the closest expiration"""
    try:
        asset = yf.Ticker(ticker)
        expirations = asset.options
        target_date = (datetime.now() + timedelta(days=expiration_days)).strftime('%Y-%m-%d')
        closest_date = min(expirations, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.strptime(target_date, '%Y-%m-%d')).days))
        option_chain = asset.option_chain(closest_date)
        calls = option_chain.calls
        closest_strike = calls['strike'].iloc[(calls['strike'] - strike).abs().argmin()]
        market_price = calls[calls['strike'] == closest_strike]['lastPrice'].iloc[0]
        if np.isnan(market_price) or market_price <= 0:
            return np.nan
        return market_price
    except Exception as e:
        print(f"Error fetching market price for {ticker} at strike {strike}: {e}")
        return np.nan

def create_option_pricing_table(processed_data: Dict, model: EWMArBergoliModel) -> pd.DataFrame:
    """Create table with option prices, 95% CI, market prices, and relative error"""
    print("Computing option pricing table...")
    options_specs = [
        {'asset': 'SPY', 'strikes': [500, 505, 510, 515]},
        {'asset': 'META', 'strikes': [500, 505, 510, 515]},
        {'asset': 'BTC-USD', 'strikes': [45000, 46000, 47000, 48000]}
    ]
    results = []
    T = 0.25  # 3 months
    r = 0.05
    N = int(T * 252)
    M = 1000  # Increased paths for better CI accuracy
    for spec in options_specs:
        asset = spec['asset']
        if asset not in processed_data:
            print(f"Warning: No data for {asset}")
            continue
        S0 = processed_data[asset]['spot_price']
        S_paths, _, _ = model.simulate_paths(S0, T, N, M)
        for K in spec['strikes']:
            pricing = compute_option_price(S_paths, K, r, T)
            model_price = pricing['price']
            ci_lower = pricing['ci_lower']
            ci_upper = pricing['ci_upper']
            market_price = fetch_market_option_price(asset, K)
            if np.isnan(market_price) or market_price <= 0:
                relative_error = np.nan
            else:
                relative_error = abs(model_price - market_price) / market_price * 100
            results.append({
                'Asset': asset.replace('-USD', ''),
                'Strike': K,
                'EWMA-rBergomi Price': f"{model_price:.2f}",
                '95% CI': f"({ci_lower:.2f}, {ci_upper:.2f})",
                'Market Price': f"{market_price:.2f}" if not np.isnan(market_price) else "N/A",
                'Relative Error': f"{relative_error:.2f}%" if not np.isnan(relative_error) else "N/A"
            })
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Load processed data
    try:
        with open('processed_data.pkl', 'rb') as f:
            processed_data = pickle.load(f)
        print(f"Loaded processed data for {len(processed_data)} assets")
    except FileNotFoundError:
        print("Error: processed_data.pkl not found. Run data_collection.py first.")
        processed_data = {}

    # Initialize model
    model = EWMArBergoliModel(
        V0=0.02, nu=0.3, alpha=0.2, beta=0.1, rho=-0.7,
        epsilon=0.01, H_max=0.5, lambda_ewma=0.1, gamma=0.5, theta_ref=0.02, r=0.05
    )

    # Compute and save option pricing table
    pricing_table = create_option_pricing_table(processed_data, model)
    if not pricing_table.empty:
        print("\nTable: Option Pricing Results")
        print(pricing_table.to_string(index=False))
        pricing_table.to_csv('option_pricing.csv', index=False)
        print("→ Saved to option_pricing.csv")
    else:
        print("Error: Could not generate option pricing table")
