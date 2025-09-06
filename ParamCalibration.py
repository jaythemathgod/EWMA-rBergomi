#!/usr/bin/env python3
"""
Calibrate EWMA-rBergomi Parameters Script
Optimizes parameters for the EWMA-rBergomi model based on training data by minimizing Jensen-Shannon distance.
Prints optimized parameters to the console.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gamma
from scipy.optimize import minimize
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

class ModelCalibrator:
    """Calibrate model parameters by minimizing JS distance on training data"""

    def __init__(self, empirical_returns: np.ndarray):
        self.empirical_returns = empirical_returns[np.isfinite(empirical_returns)]
        self.js_analyzer = JSDistanceAnalyzer()

    def objective_function(self, params: np.ndarray) -> float:
        V0, nu, alpha, beta = params
        if V0 <= 0 or V0 > 1 or nu <= 0 or nu > 2 or alpha <= 0 or alpha > 1 or beta <= 0.01 or beta > 0.5:
            return 1e6
        model = EWMArBergoliModel(V0=V0, nu=nu, alpha=alpha, beta=beta)
        S_paths = model.simulate_paths(S0=100, T=1, N=252, M=5000)
        sim_returns = np.log(S_paths[:, 1:] / S_paths[:, :-1]).flatten()
        sim_returns = sim_returns[np.isfinite(sim_returns)]
        js_dist = self.js_analyzer.compute_js_distance(self.empirical_returns, sim_returns)
        penalty = 0.01 * np.sum((params - np.array([0.02, 0.3, 0.2, 0.1])**2))
        return js_dist + penalty

    def calibrate_ewma_rbergomi(self) -> Dict:
        x0 = np.array([0.02, 0.3, 0.2, 0.1])
        bounds = [(0.001, 0.1), (0.1, 1.0), (0.05, 0.5), (0.01, 0.49)]
        result = minimize(
            self.objective_function,
            x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100}
        )
        return {
            'V0': result.x[0],
            'nu': result.x[1],
            'alpha': result.x[2],
            'beta': result.x[3],
            'js_distance': result.fun,
            'success': result.success
        }

def calibrate_parameters(processed_data: Dict) -> None:
    """Calibrate EWMA-rBergomi parameters for each asset and print results"""
    print("Calibrating EWMA-rBergomi parameters...")
    for asset, data in processed_data.items():
        if len(data['train_returns']) < 100:
            print(f"Skipping {asset}: Insufficient training data")
            continue
        train_returns = data['train_returns'].values
        train_returns = train_returns[np.isfinite(train_returns)]
        if len(train_returns) == 0:
            print(f"Skipping {asset}: No valid training returns")
            continue
        calibrator = ModelCalibrator(train_returns)
        params = calibrator.calibrate_ewma_rbergomi()
        print(f"\nOptimized parameters for {asset.replace('-USD', '')}:")
        print(f"  V0: {params['V0']:.4f}")
        print(f"  nu: {params['nu']:.4f}")
        print(f"  alpha: {params['alpha']:.4f}")
        print(f"  beta: {params['beta']:.4f}")
        print(f"  JS Distance: {params['js_distance']:.4f}")
        print(f"  Success: {params['success']}")

if __name__ == "__main__":
    # Load processed data
    try:
        with open('processed_data.pkl', 'rb') as f:
            processed_data = pickle.load(f)
        print(f"Loaded processed data for {len(processed_data)} assets")
    except FileNotFoundError:
        print("Error: processed_data.pkl not found. Run data_collection.py first.")
        processed_data = {}

    # Calibrate and print parameters
    calibrate_parameters(processed_data)
