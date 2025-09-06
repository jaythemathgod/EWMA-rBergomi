#!/usr/bin/env python3
"""
Data Collection Script
Handles downloading and processing historical market data for multiple assets.
Implements the DataProcessor class from the original code.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
from typing import Dict

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class DataProcessor:
    """Handle data downloading, processing, and train/validation splitting"""

    def __init__(self):
        self.assets = ['SPY', 'VOO', 'GS', 'META', 'BTC-USD', 'ETH-USD', 'GLD', 'USO']
        self.start_date = '2022-01-01'
        self.end_date = '2025-08-31'
        self.train_split = '2024-12-31'

    def download_data(self) -> Dict[str, pd.DataFrame]:
        """Download historical data for all assets"""
        print("Downloading market data...")
        data = {}
        for asset in self.assets:
            try:
                ticker = yf.Ticker(asset)
                df = ticker.history(start=self.start_date, end=self.end_date)
                if not df.empty and 'Close' in df.columns:
                    data[asset] = df
                    print(f"✓ Downloaded {asset}: {len(df)} observations")
                else:
                    print(f"✗ No data for {asset}")
            except Exception as e:
                print(f"✗ Error downloading {asset}: {e}")
                continue
        return data

    def process_returns(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Compute log returns and realized variance"""
        processed = {}
        for asset, df in data.items():
            log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
            realized_var = log_returns.rolling(window=20).var()
            train_mask = log_returns.index <= self.train_split
            processed[asset] = {
                'returns': log_returns,
                'realized_var': realized_var,
                'train_returns': log_returns[train_mask],
                'val_returns': log_returns[~train_mask],
                'train_var': realized_var[train_mask],
                'val_var': realized_var[~train_mask],
                'spot_price': df['Close'].iloc[-1],
                'price_data': df['Close']
            }
        return processed

if __name__ == "__main__":
    processor = DataProcessor()
    raw_data = processor.download_data()
    processed_data = processor.process_returns(raw_data)
    print(f"Processed data for {len(processed_data)} assets")
    for asset, data in processed_data.items():
        train_size = len(data['train_returns'])
        val_size = len(data['val_returns'])
        print(f"  {asset}: {train_size} training, {val_size} validation observations")
    # Save processed data for use in other scripts
    with open('processed_data.pkl', 'wb') as f:
        import pickle
        pickle.dump(processed_data, f)
    print("✓ Processed data saved to processed_data.pkl")
