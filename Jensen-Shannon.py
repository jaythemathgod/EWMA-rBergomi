#!/usr/bin/env python3
"""
Jensen-Shannon Table Script
Computes Jensen-Shannon distances for various models using fixed parameters for EWMA-rBergomi.
Prints the comparison table to the console.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gamma
from typing import Dict
import pickle

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

    def simulate_paths(self, S0: float, T: float, N: int, M: int) -> np.ndarray:
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
        return S_paths

class JSDistanceAnalyzer:
    """Jensen-Shannon distance computation and analysis"""

    @staticmethod
    def compute_js_distance(X: np.ndarray, Y: np.ndarray, n_bins: int = 100) -> float:
        X = X[np.isfinite(X)]
        Y = Y[np.isfinite(Y)]
        if len(X) == 0 or len(Y) == 0:
            return np.inf
        kde_X = stats.gaussian_kde(X)
        kde_Y = stats.gaussian_kde(Y)
        x_min = min(np.min(X), np.min(Y))
        x_max = max(np.max(X), np.max(Y))
        x_eval = np.linspace(x_min, x_max, n_bins)
        p = kde_X(x_eval)
        q = kde_Y(x_eval)
        p = p / np.trapz(p, x_eval)
        q = q / np.trapz(q, x_eval)
        p = np.maximum(p, 1e-10)
        q = np.maximum(q, 1e-10)
        m = 0.5 * (p + q)
        m = np.maximum(m, 1e-10)
        kl_pm = np.trapz(p * np.log(p / m), x_eval)
        kl_qm = np.trapz(q * np.log(q / m), x_eval)
        js_distance = np.sqrt(0.5 * (kl_pm + kl_qm))
        return js_distance if np.isfinite(js_distance) else np.inf

class BaselineModels:
    """Implementation of baseline models for comparison: rBergomi, Heston, GBM"""

    @staticmethod
    def simulate_rbergomi(S0: float, T: float, N: int, M: int,
                         V0: float = 0.02, nu: float = 0.3, H: float = 0.1,
                         rho: float = -0.7, r: float = 0.05) -> np.ndarray:
        dt = T / N
        t_grid = np.linspace(0, T, N+1)
        S_paths = np.zeros((M, N+1))
        sigma_paths = np.zeros((M, N+1))
        S_paths[:, 0] = S0
        sigma_paths[:, 0] = np.sqrt(V0)
        dW = np.random.normal(0, np.sqrt(dt), (M, N))
        dW_perp = np.random.normal(0, np.sqrt(dt), (M, N))
        dZ = rho * dW + np.sqrt(1 - rho**2) * dW_perp
        kernel_matrix = np.zeros((N, N))
        for n in range(N):
            for k in range(n):
                kernel_matrix[n, k] = ((t_grid[n+1] - t_grid[k])**(H - 0.5)) / gamma(H + 0.5)
        for n in range(N):
            V_t = np.sum(kernel_matrix[n, :n] * dZ[:, :n], axis=1)
            A_t = (t_grid[n+1]**(2*H)) / (2*H) if H > 0 else 0
            sigma_paths[:, n+1] = np.sqrt(V0) * np.exp(nu * V_t - 0.5 * nu**2 * A_t)
            drift = r - 0.5 * sigma_paths[:, n]**2
            S_paths[:, n+1] = S_paths[:, n] * np.exp(drift * dt + sigma_paths[:, n] * dW[:, n])
        return S_paths

    @staticmethod
    def simulate_heston(S0: float, T: float, N: int, M: int,
                        V0: float = 0.02, kappa: float = 2.0, theta: float = 0.02,
                        xi: float = 0.3, rho: float = -0.7, r: float = 0.05) -> np.ndarray:
        dt = T / N
        S_paths = np.zeros((M, N+1))
        V_paths = np.zeros((M, N+1))
        S_paths[:, 0] = S0
        V_paths[:, 0] = V0
        dW_S = np.random.normal(0, np.sqrt(dt), (M, N))
        dW_V = np.random.normal(0, np.sqrt(dt), (M, N))
        dW_V_corr = rho * dW_S + np.sqrt(1 - rho**2) * dW_V
        for n in range(N):
            dV = kappa * (theta - np.maximum(V_paths[:, n], 0)) * dt + xi * np.sqrt(np.maximum(V_paths[:, n], 0)) * dW_V_corr[:, n]
            V_paths[:, n+1] = np.maximum(V_paths[:, n] + dV, 1e-8)
            sqrt_V = np.sqrt(np.maximum(V_paths[:, n], 0))
            drift = r - 0.5 * V_paths[:, n]
            S_paths[:, n+1] = S_paths[:, n] * np.exp(drift * dt + sqrt_V * dW_S[:, n])
        return S_paths

    @staticmethod
    def simulate_gbm(S0: float, T: float, N: int, M: int,
                     sigma: float = 0.2, r: float = 0.05) -> np.ndarray:
        dt = T / N
        dW = np.random.normal(0, np.sqrt(dt), (M, N))
        S_paths = np.zeros((M, N+1))
        S_paths[:, 0] = S0
        for n in range(N):
            drift = (r - 0.5 * sigma**2) * dt
            S_paths[:, n+1] = S_paths[:, n] * np.exp(drift + sigma * dW[:, n])
        return S_paths

def create_js_distance_table(processed_data: Dict) -> pd.DataFrame:
    """Create Jensen-Shannon distance table using fixed parameters for EWMA-rBergomi"""
    print("Computing Jensen-Shannon distance table...")
    results = []
    js_analyzer = JSDistanceAnalyzer()
    ewma_params = {
        'V0': 0.02,
        'nu': 0.3,
        'alpha': 0.2,
        'beta': 0.1,
        'rho': -0.7,
        'epsilon': 0.01,
        'H_max': 0.5,
        'lambda_ewma': 0.1,
        'gamma': 0.5,
        'theta_ref': 0.02,
        'r': 0.05
    }
    for asset, data in processed_data.items():
        if len(data['train_returns']) < 100:
            print(f"Skipping {asset}: Insufficient training data")
            continue
        train_returns = data['train_returns'].values
        train_returns = train_returns[np.isfinite(train_returns)]
        if len(train_returns) == 0:
            print(f"Skipping {asset}: No valid training returns")
            continue
        S0 = 100
        T = 1
        N = 252
        M = 100
        model = EWMArBergoliModel(**ewma_params)
        S_paths = model.simulate_paths(S0, T, N, M)
        ewma_returns = np.log(S_paths[:, 1:] / S_paths[:, :-1]).flatten()
        ewma_returns = ewma_returns[np.isfinite(ewma_returns)]
        js_ewma = js_analyzer.compute_js_distance(train_returns, ewma_returns)
        rberg_S_paths = BaselineModels.simulate_rbergomi(S0, T, N, M, H=0.1)
        rberg_returns = np.log(rberg_S_paths[:, 1:] / rberg_S_paths[:, :-1]).flatten()
        rberg_returns = rberg_returns[np.isfinite(rberg_returns)]
        js_rberg = js_analyzer.compute_js_distance(train_returns, rberg_returns)
        heston_S_paths = BaselineModels.simulate_heston(S0, T, N, M)
        heston_returns = np.log(heston_S_paths[:, 1:] / heston_S_paths[:, :-1]).flatten()
        heston_returns = heston_returns[np.isfinite(heston_returns)]
        js_heston = js_analyzer.compute_js_distance(train_returns, heston_returns)
        gbm_S_paths = BaselineModels.simulate_gbm(S0, T, N, M, sigma=np.std(train_returns)*np.sqrt(252))
        gbm_returns = np.log(gbm_S_paths[:, 1:] / gbm_S_paths[:, :-1]).flatten()
        gbm_returns = gbm_returns[np.isfinite(gbm_returns)]
        js_gbm = js_analyzer.compute_js_distance(train_returns, gbm_returns)
        results.append({
            'Asset': asset.replace('-USD', ''),
            'EWMA-rBergomi': f"{js_ewma:.4f}",
            'rBergomi': f"{js_rberg:.4f}",
            'Heston': f"{js_heston:.4f}",
            'GBM': f"{js_gbm:.4f}"
        })
    return pd.DataFrame(results) if results else pd.DataFrame()

if __name__ == "__main__":
    # Load processed data
    try:
        with open('processed_data.pkl', 'rb') as f:
            processed_data = pickle.load(f)
        print(f"Loaded processed data for {len(processed_data)} assets")
    except FileNotFoundError:
        print("Error: processed_data.pkl not found. Run data_collection.py first.")
        processed_data = {}

    # Compute and print JS distance table
    js_table = create_js_distance_table(processed_data)
    if not js_table.empty:
        print("\nTable: Jensen-Shannon Distances Across Asset Classes and Models")
        print(js_table.to_string(index=False))
    else:
        print("Error: Could not generate JS distance table")
