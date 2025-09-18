# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 09:37:46 2025

@author: gilbe

Calculate Hurst exponents
"""

import yfinance as yf
import pandas as pd
import numpy as np

ticker = "^GSPC"
df = yf.download(ticker, start="2000-01-01", end="2025-09-10", group_by='ticker')
df = df[ticker]['Close']
df.plot(title="S&P 500")

def get_hurst_exponent(ts, max_lag=20):
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    
    return np.polyfit(np.log(lags), np.log(tau), 1)[0]

for lag in [20, 100, 250, 500, 1000]:
    hurst_exp = get_hurst_exponent(df.values, lag)
    print(f"{lag} lags: {hurst_exp:.4f}")
    
shorter_series = df.loc["2009":"2015"].values
for lag in [20, 100, 250, 500]:
    hurst_exp = get_hurst_exponent(shorter_series, lag)
    print(f"{lag} lags: {hurst_exp:.4f}")
    
shorter_series = df.loc["2024":"2025"].values
for lag in [20, 100, 250]:
    hurst_exp = get_hurst_exponent(shorter_series, lag)
    print(f"{lag} lags: {hurst_exp:.4f}")

