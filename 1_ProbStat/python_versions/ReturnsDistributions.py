#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ======================================================================
# RETURNS DISTRIBUTIONS ANALYSIS
# ======================================================================

# Import required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import skew, kurtosis, t, jarque_bera, norm
import yfinance as yf
import os
import datetime
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Set plots to be displayed inline (for Jupyter notebooks)
# %matplotlib inline

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# --------------------- SETUP AND HELPER FUNCTIONS ----------------------

def fetch_stock_data(ticker, start_date='2000-01-03', end_date=None):
    """
    Fetch stock data using yfinance
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format (defaults to today)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing stock data
    """
    if end_date is None:
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def calculate_returns(prices_df):
    """
    Calculate different types of returns from price data
    
    Parameters:
    -----------
    prices_df : pd.DataFrame
        DataFrame with price data (OHLC format)
        
    Returns:
    --------
    dict
        Dictionary containing different return series
    """
    # Calculate simple returns
    daily_simple_returns = 100 * prices_df['Adj Close'].pct_change().dropna()
    
    # Calculate log returns
    daily_log_returns = 100 * np.log(prices_df['Adj Close'] / prices_df['Adj Close'].shift(1)).dropna()
    
    # Calculate monthly returns
    # Resample to monthly frequency and calculate returns
    monthly_prices = prices_df['Adj Close'].resample('M').last()
    monthly_simple_returns = 100 * monthly_prices.pct_change().dropna()
    monthly_log_returns = 100 * np.log(monthly_prices / monthly_prices.shift(1)).dropna()
    
    return {
        'daily_simple': daily_simple_returns,
        'daily_log': daily_log_returns,
        'monthly_simple': monthly_simple_returns,
        'monthly_log': monthly_log_returns
    }

def calculate_basic_stats(returns):
    """
    Calculate basic summary statistics for a returns series
    
    Parameters:
    -----------
    returns : pd.Series or np.array
        Returns series
        
    Returns:
    --------
    pd.Series
        Series containing summary statistics
    """
    stats_dict = {
        'nobs': len(returns),
        'min': returns.min(),
        'max': returns.max(),
        'mean': returns.mean(),
        'median': returns.median(),
        'std_dev': returns.std(),
        'variance': returns.var(),
        'skewness': skew(returns),
        'kurtosis': kurtosis(returns, fisher=True),  # Excess kurtosis (normal = 0)
        '5%': np.percentile(returns, 5),
        '95%': np.percentile(returns, 95)
    }
    
    return pd.Series(stats_dict)

def test_mean_return(returns, mu=0, alternative='two-sided'):
    """
    Perform t-test for mean returns
    
    Parameters:
    -----------
    returns : pd.Series or np.array
        Returns series
    mu : float
        Hypothesized mean value
    alternative : str
        Alternative hypothesis: 'two-sided', 'greater', or 'less'
        
    Returns:
    --------
    dict
        Dictionary containing test results
    """
    if alternative == 'two-sided':
        alternative_scipy = 'two-sided'
    elif alternative == 'greater':
        alternative_scipy = 'greater'
    elif alternative == 'less':
        alternative_scipy = 'less'
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    
    t_stat, p_value = stats.ttest_1samp(returns, mu)
    
    # If alternative is not two-sided, adjust p-value
    if alternative == 'greater' and t_stat < 0:
        p_value = 1 - p_value/2
    elif alternative == 'greater' and t_stat >= 0:
        p_value = p_value/2
    elif alternative == 'less' and t_stat > 0:
        p_value = 1 - p_value/2
    elif alternative == 'less' and t_stat <= 0:
        p_value = p_value/2
    
    print(f"T-test of mean returns (H0: μ = {mu})")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value ({alternative}): {p_value:.4f}")
    print(f"Mean: {returns.mean():.4f}")
    print(f"Std Error: {returns.std()/np.sqrt(len(returns)):.4f}")
    
    return {'t_stat': t_stat, 'p_value': p_value}

def plot_return_distribution(returns, title="Returns Distribution"):
    """
    Plot histogram and density of returns vs. normal distribution
    
    Parameters:
    -----------
    returns : pd.Series or np.array
        Returns series
    title : str
        Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot histogram
    axes[0].hist(returns, bins=40, density=True, alpha=0.6, color='steelblue')
    axes[0].set_title(f"{title} - Histogram")
    axes[0].set_xlabel("Returns (%)")
    axes[0].set_ylabel("Frequency")
    
    # Add normal curve to histogram
    x = np.linspace(min(returns), max(returns), 100)
    mean, std = returns.mean(), returns.std()
    normal_curve = stats.norm.pdf(x, mean, std)
    axes[0].plot(x, normal_curve, 'r-', linewidth=2, label='Normal Distribution')
    axes[0].legend()
    
    # Plot empirical density
    returns_series = pd.Series(returns)
    returns_series.plot.kde(ax=axes[1], linewidth=2, label='Empirical')
    
    # Add normal curve to density plot
    x = np.linspace(min(returns), max(returns), 100)
    mean, std = returns.mean(), returns.std()
    normal_curve = stats.norm.pdf(x, mean, std)
    axes[1].plot(x, normal_curve, 'r--', linewidth=2, label='Normal')
    
    axes[1].set_title(f"{title} - Density")
    axes[1].set_xlabel("Returns (%)")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

def test_normality(returns, name="Returns"):
    """
    Test normality of returns distribution
    
    Parameters:
    -----------
    returns : pd.Series or np.array
        Returns series
    name : str
        Name of returns series for display
        
    Returns:
    --------
    dict
        Dictionary containing test results
    """
    print(f"Normality Tests for {name}")
    print("-" * 50)
    
    # Jarque-Bera test
    jb_stat, jb_pvalue = jarque_bera(returns)
    print(f"Jarque-Bera Test:")
    print(f"Statistic: {jb_stat:.4f}")
    print(f"p-value: {jb_pvalue:.6f}")
    print(f"Conclusion: {'Reject' if jb_pvalue < 0.05 else 'Fail to reject'} normality at 5% significance level")
    
    # Skewness test
    s = skew(returns)
    n = len(returns)
    skew_stat = s / np.sqrt(6/n)
    skew_pvalue = 2 * (1 - norm.cdf(abs(skew_stat)))
    
    print(f"\nSkewness Test:")
    print(f"Skewness: {s:.4f}")
    print(f"Test statistic: {skew_stat:.4f}")
    print(f"p-value: {skew_pvalue:.6f}")
    
    # Kurtosis test
    k = kurtosis(returns, fisher=True)  # Excess kurtosis (normal = 0)
    kurt_stat = k / np.sqrt(24/n)
    
    print(f"\nExcess Kurtosis Test:")
    print(f"Excess Kurtosis: {k:.4f}")
    print(f"Test statistic: {kurt_stat:.4f}")
    print("-" * 50)
    
    return {
        'jarque_bera': {'stat': jb_stat, 'pvalue': jb_pvalue},
        'skewness': {'value': s, 'stat': skew_stat, 'pvalue': skew_pvalue},
        'kurtosis': {'value': k, 'stat': kurt_stat}
    }

# -------------------- MAIN ANALYSIS FUNCTIONS -------------------------

def analyze_ibm_returns(data_path=None):
    """
    Analyze IBM returns data
    """
    print("=" * 70)
    print("IBM RETURNS ANALYSIS")
    print("=" * 70)
    
    if data_path:
        # Load data from file
        try:
            data_ibm = pd.read_csv(data_path, header=0)
            ibm_returns = data_ibm.iloc[:, 1]  # Assuming returns are in column 2
        except:
            print(f"Error reading file: {data_path}")
            print("Using Yahoo Finance data instead")
            ibm_data = fetch_stock_data('IBM', start_date='2010-01-01')
            ibm_returns = calculate_returns(ibm_data)['daily_simple']
    else:
        # Fetch IBM data from Yahoo Finance
        ibm_data = fetch_stock_data('IBM', start_date='2010-01-01')
        ibm_returns = calculate_returns(ibm_data)['daily_simple']
    
    # Calculate simple returns stats
    print("\nIBM Simple Returns Analysis:")
    simple_ibm_returns = ibm_returns
    stats_simple = calculate_basic_stats(simple_ibm_returns)
    print(stats_simple)
    
    # Test mean returns
    print("\nTest of Mean Returns:")
    test_mean_return(simple_ibm_returns)
    test_mean_return(simple_ibm_returns, alternative='greater')
    
    # Plot return distribution
    plot_return_distribution(simple_ibm_returns, "IBM Simple Returns")
    
    # Test normality
    test_normality(simple_ibm_returns, "IBM Simple Returns")
    
    # Calculate log returns
    log_ibm_returns = 100 * np.log(1 + simple_ibm_returns/100)
    
    # Repeat analysis for log returns
    print("\n" + "=" * 50)
    print("IBM Log Returns Analysis:")
    stats_log = calculate_basic_stats(log_ibm_returns)
    print(stats_log)
    
    # Test mean returns (log)
    print("\nTest of Mean Returns (Log):")
    test_mean_return(log_ibm_returns)
    test_mean_return(log_ibm_returns, alternative='greater')
    
    # Plot return distribution (log)
    plot_return_distribution(log_ibm_returns, "IBM Log Returns")
    
    # Test normality (log)
    test_normality(log_ibm_returns, "IBM Log Returns")

def analyze_energy_stocks():
    """
    Analyze energy sector stocks and indices
    """
    print("=" * 70)
    print("ENERGY SECTOR ANALYSIS")
    print("=" * 70)
    
    # Fetch data for market indices
    print("\nFetching market indices data...")
    xle_data = fetch_stock_data('XLE')  # SPDR Energy ETF
    spy_data = fetch_stock_data('SPY')  # S&P 500 Index
    
    # Calculate returns
    xle_returns = calculate_returns(xle_data)
    spy_returns = calculate_returns(spy_data)
    
    # Plot XLE price and returns
    plt.figure(figsize=(12, 6))
    xle_data['Adj Close'].plot()
    plt.title("SPDR Energy ETF (XLE) Prices")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.show()
    
    plt.figure(figsize=(12, 6))
    xle_returns['daily_log'].plot()
    plt.title("XLE Daily Log Returns")
    plt.xlabel("Date")
    plt.ylabel("Returns (%)")
    plt.show()
    
    # Fetch individual stock data
    print("\nFetching individual stock data...")
    tickers = ['SRE', 'BRK-B', 'DUK']  # Sempra, Berkshire, Duke
    
    stock_data = {}
    stock_returns = {}
    
    for ticker in tickers:
        stock_data[ticker] = fetch_stock_data(ticker)
        stock_returns[ticker] = calculate_returns(stock_data[ticker])
    
    # Create combined returns DataFrames
    daily_simple_returns = pd.DataFrame({
        'SPY': spy_returns['daily_simple'],
        'XLE': xle_returns['daily_simple'],
        'SRE': stock_returns['SRE']['daily_simple'],
        'BRK': stock_returns['BRK-B']['daily_simple'],
        'DUK': stock_returns['DUK']['daily_simple']
    })
    
    daily_log_returns = pd.DataFrame({
        'SPY': spy_returns['daily_log'],
        'XLE': xle_returns['daily_log'],
        'SRE': stock_returns['SRE']['daily_log'],
        'BRK': stock_returns['BRK-B']['daily_log'],
        'DUK': stock_returns['DUK']['daily_log']
    })
    
    monthly_simple_returns = pd.DataFrame({
        'SPY': spy_returns['monthly_simple'],
        'XLE': xle_returns['monthly_simple'],
        'SRE': stock_returns['SRE']['monthly_simple'],
        'BRK': stock_returns['BRK-B']['monthly_simple'],
        'DUK': stock_returns['DUK']['monthly_simple']
    })
    
    monthly_log_returns = pd.DataFrame({
        'SPY': spy_returns['monthly_log'],
        'XLE': xle_returns['monthly_log'],
        'SRE': stock_returns['SRE']['monthly_log'],
        'BRK': stock_returns['BRK-B']['monthly_log'],
        'DUK': stock_returns['DUK']['monthly_log']
    })
    
    # Calculate summary statistics
    print("\nSummary Statistics for Daily Simple Returns:")
    daily_simple_stats = daily_simple_returns.apply(calculate_basic_stats)
    print(daily_simple_stats.T)
    
    print("\nSummary Statistics for Daily Log Returns:")
    daily_log_stats = daily_log_returns.apply(calculate_basic_stats)
    print(daily_log_stats.T)
    
    print("\nSummary Statistics for Monthly Simple Returns:")
    monthly_simple_stats = monthly_simple_returns.apply(calculate_basic_stats)
    print(monthly_simple_stats.T)
    
    print("\nSummary Statistics for Monthly Log Returns:")
    monthly_log_stats = monthly_log_returns.apply(calculate_basic_stats)
    print(monthly_log_stats.T)
    
    # Analyze Berkshire Hathaway in detail
    analyze_brk_returns(stock_returns['BRK-B']['daily_log'])

def analyze_brk_returns(brk_returns):
    """
    Detailed analysis of Berkshire Hathaway returns
    
    Parameters:
    -----------
    brk_returns : pd.Series
        Berkshire Hathaway returns series
    """
    print("=" * 70)
    print("BERKSHIRE HATHAWAY DETAILED ANALYSIS")
    print("=" * 70)
    
    # Calculate statistics
    stats = calculate_basic_stats(brk_returns)
    print("\nBerkshire Hathaway Daily Log Returns Statistics:")
    print(stats)
    
    # Test mean returns
    print("\nTest of Mean Returns:")
    test_mean_return(brk_returns)
    test_mean_return(brk_returns, alternative='greater')
    
    # Plot return distribution
    plot_return_distribution(brk_returns, "Berkshire Hathaway Daily Log Returns")
    
    # Test normality
    test_normality(brk_returns, "Berkshire Hathaway Daily Log Returns")

# -------------------- EXAMPLE USAGE -------------------------

if __name__ == "__main__":
    print("\nPython version of ReturnsDistributions.R")
    print("=" * 50)
    
    # Set random seed for reproducibility
    np.random.seed(123)
    
    # Example 1: Analyze IBM returns
    # Specify the path to your IBM returns data if available
    # analyze_ibm_returns(data_path='path/to/ibm/data.csv')
    
    # Or use Yahoo Finance data
    analyze_ibm_returns()
    
    # Example 2: Analyze energy sector stocks
    analyze_energy_stocks()
    
    print("\nNote: The output of this script may differ from the original R script")
    print("due to differences in data sources and package implementations.")
    print("The Python version uses Yahoo Finance data by default, while the R")
    print("version used data from text files.")