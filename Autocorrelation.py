#!/usr/bin/env python3
"""
Autocorrelation Analysis Script
Computes rolling correlation of volatility and compares empirical vs. model results for SPY and BTC-USD.
Implements the compute_autocorrelation_analysis function and plotting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import pickle
from scipy.special import gamma

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

def compute_autocorrelation_analysis(processed_data: Dict, model: EWMArBergoliModel) -> Dict:
    """Compute rolling correlation of volatility following Section 5.1"""
    print("Computing autocorrelation analysis...")
    results = {}
    for asset, data in processed_data.items():
        if len(data['realized_var']) < 100:
            continue
        emp_vol = np.sqrt(data['realized_var'].dropna())
        if len(emp_vol) < 50:
            continue
        S0 = data['spot_price']
        T = len(emp_vol) / 252
        N = len(emp_vol)
        M = 100
        _, sigma_paths, _ = model.simulate_paths(S0, T, N, M)
        sim_vol = np.mean(sigma_paths, axis=0)[1:]
        lags = [1, 5, 10, 20, 40]
        T_w = 60
        autocorrs = {'empirical': {}, 'model': {}}
        for lag in lags:
            emp_corrs = []
            sim_corrs = []
            for t0 in range(0, len(emp_vol) - T_w - lag, 10):
                window_end = t0 + T_w
                if window_end + lag < len(emp_vol):
                    vol_t = emp_vol[t0:window_end]
                    vol_t_lag = emp_vol[t0+lag:window_end+lag]
                    if len(vol_t) > 10 and np.std(vol_t) > 0 and np.std(vol_t_lag) > 0:
                        emp_corr = np.corrcoef(vol_t, vol_t_lag)[0, 1]
                        if np.isfinite(emp_corr):
                            emp_corrs.append(emp_corr)
                if window_end + lag < len(sim_vol):
                    sim_vol_t = sim_vol[t0:window_end]
                    sim_vol_t_lag = sim_vol[t0+lag:window_end+lag]
                    if len(sim_vol_t) > 10 and np.std(sim_vol_t) > 0 and np.std(sim_vol_t_lag) > 0:
                        sim_corr = np.corrcoef(sim_vol_t, sim_vol_t_lag)[0, 1]
                        if np.isfinite(sim_corr):
                            sim_corrs.append(sim_corr)
            autocorrs['empirical'][lag] = np.mean(emp_corrs) if emp_corrs else np.nan
            autocorrs['model'][lag] = np.mean(sim_corrs) if sim_corrs else np.nan
        results[asset] = autocorrs
    return results

def plot_autocorrelation_analysis(autocorr_results: Dict):
    """Plot autocorrelation function comparisons for SPY and BTC-USD"""
    print("Creating autocorrelation plots...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes = axes.flatten()
    plot_assets = ['SPY', 'BTC-USD']  # Explicitly plot SPY and BTC-USD
    for i, asset in enumerate(plot_assets):
        if asset not in autocorr_results:
            print(f"Warning: No autocorrelation data for {asset}")
            continue
        data = autocorr_results[asset]
        lags = list(data['empirical'].keys())
        emp_corrs = [data['empirical'][lag] for lag in lags]
        model_corrs = [data['model'][lag] for lag in lags]
        valid_idx = [j for j in range(len(emp_corrs)) if np.isfinite(emp_corrs[j]) and np.isfinite(model_corrs[j])]
        if valid_idx:
            lags_plot = [lags[j] for j in valid_idx]
            emp_plot = [emp_corrs[j] for j in valid_idx]
            model_plot = [model_corrs[j] for j in valid_idx]
            ax = axes[i]
            ax.plot(lags_plot, emp_plot, 'o-', label='Empirical', linewidth=2, markersize=6)
            ax.plot(lags_plot, model_plot, 's-', label='EWMA-rBergomi', linewidth=2, markersize=6)
            ax.set_xlabel('Lag (days)')
            ax.set_ylabel('Autocorrelation')
            ax.set_title(f'{asset.replace("-USD", "")} Volatility Autocorrelation')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.1, 1.1)
            print(f"\n{asset} Autocorrelation Results:")
            print("Lag\tEmpirical\tModel")
            for lag, emp, mod in zip(lags_plot, emp_plot, model_plot):
                print(f"{lag}\t{emp:.4f}\t{mod:.4f}")
    plt.tight_layout()
    plt.savefig('autocorrelation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Load processed data
    try:
        with open('processed_data.pkl', 'rb') as f:
            processed_data = pickle.load(f)
        print(f"Loaded processed data for {len(processed_data)} assets")
    except FileNotFoundError:
        print("Error: processed_data.pkl not found. Run data_collection.py first.")
        processed_data = {}

    # Initialize model with default parameters
    model = EWMArBergoliModel(
        V0=0.02, nu=0.3, alpha=0.2, beta=0.1, rho=-0.7,
        epsilon=0.01, H_max=0.5, lambda_ewma=0.1, gamma=0.5, theta_ref=0.02, r=0.05
    )

    # Compute and plot autocorrelation analysis
    autocorr_results = compute_autocorrelation_analysis(processed_data, model)
    if autocorr_results:
        print("Rolling correlation analysis completed for assets:")
        for asset in autocorr_results.keys():
            print(f"  {asset}")
        plot_autocorrelation_analysis(autocorr_results)
        print("→ Autocorrelation plots saved to autocorrelation_analysis.png")
    else:
        print("Error: Could not complete autocorrelation analysis")
