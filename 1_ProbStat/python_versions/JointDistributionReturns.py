#!/usr/bin/env python
# -*- coding: utf-8 -*-

# =============================================================================
# JOINT DISTRIBUTIONS OF FINANCIAL RETURNS - HEATMAPS AND HISTOGRAMS
# =============================================================================

# Import required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import seaborn as sns
import yfinance as yf
from scipy.stats import gaussian_kde
from matplotlib.ticker import MaxNLocator
from datetime import datetime

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# =============================================================================
# DATA ACQUISITION AND PREPARATION
# =============================================================================

def fetch_stock_data(tickers, start_date='2000-01-03', end_date=None):
    """
    Fetch stock data for specified tickers
    
    Parameters:
    -----------
    tickers : list
        List of ticker symbols
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format (defaults to today)
        
    Returns:
    --------
    dict
        Dictionary containing DataFrames for each ticker
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    data = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date, group_by='ticker')
        df = df[ticker]
        data[ticker] = df
    
    return data

def calculate_returns(price_data):
    """
    Calculate returns from price data
    
    Parameters:
    -----------
    price_data : dict
        Dictionary with price DataFrames for each ticker
        
    Returns:
    --------
    dict
        Dictionary with returns DataFrames for each ticker
    """
    returns = {}
    for ticker, df in price_data.items():
        # Daily log returns
        daily_log_returns = 100 * np.log(df['Close'] / df['Close'].shift(1)).dropna()
        
        # Monthly log returns
        monthly_prices = df['Close'].resample('M').last()
        monthly_log_returns = 100 * np.log(monthly_prices / monthly_prices.shift(1)).dropna()
        
        returns[ticker] = {
            'daily_log': daily_log_returns,
            'monthly_log': monthly_log_returns
        }
    
    return returns

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_price_series(price_data):
    """
    Plot price series for all tickers
    
    Parameters:
    -----------
    price_data : dict
        Dictionary with price DataFrames for each ticker
    """
    plt.figure(figsize=(14, 7))
    
    for ticker, df in price_data.items():
        plt.plot(df.index, df['Close'], label=ticker)
    
    plt.title('Price Series', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_returns_scatter(returns, ticker1, ticker2, freq='daily_log'):
    """
    Plot scatter plot of returns for two tickers
    
    Parameters:
    -----------
    returns : dict
        Dictionary with returns DataFrames for each ticker
    ticker1 : str
        First ticker symbol
    ticker2 : str
        Second ticker symbol
    freq : str
        Frequency of returns ('daily_log' or 'monthly_log')
    """
    plt.figure(figsize=(10, 8))
    
    x = returns[ticker1][freq]
    y = returns[ticker2][freq]
    
    # Create a common index
    common_index = x.index.intersection(y.index)
    x = x.loc[common_index]
    y = y.loc[common_index]
    
    plt.scatter(x, y, alpha=0.5)
    plt.title(f'{freq.capitalize()} Returns Scatter Plot: {ticker1} vs {ticker2}', fontsize=16)
    plt.xlabel(f'{ticker1} Returns (%)', fontsize=12)
    plt.ylabel(f'{ticker2} Returns (%)', fontsize=12)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_heatmap_with_histograms(returns, ticker1, ticker2, freq='daily_log', bins=50, cmap='viridis'):
    """
    Plot heatmap with histograms for joint returns distribution
    
    Parameters:
    -----------
    returns : dict
        Dictionary with returns DataFrames for each ticker
    ticker1 : str
        First ticker symbol
    ticker2 : str
        Second ticker symbol
    freq : str
        Frequency of returns ('daily_log' or 'monthly_log')
    bins : int
        Number of bins for histograms and heatmap
    cmap : str
        Colormap for heatmap
    """
    x = returns[ticker1][freq]
    y = returns[ticker2][freq]
    
    # Create a common index
    common_index = x.index.intersection(y.index)
    x = x.loc[common_index]
    y = y.loc[common_index]
    
    # Set up figure and layout
    fig = plt.figure(figsize=(10, 8))
    
    # Create a gridspec for the figures
    gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3],
                         wspace=0.05, hspace=0.05)
    
    # Create the main heatmap
    ax_heatmap = plt.subplot(gs[1, 0])
    
    # Create the top histogram
    ax_histx = plt.subplot(gs[0, 0], sharex=ax_heatmap)
    plt.setp(ax_histx.get_xticklabels(), visible=False)
    
    # Create the right histogram
    ax_histy = plt.subplot(gs[1, 1], sharey=ax_heatmap)
    plt.setp(ax_histy.get_yticklabels(), visible=False)
    
    # Create area for colorbar (in the top right corner)
    ax_colorbar = plt.subplot(gs[0, 1])
    ax_colorbar.axis('off')  # Hide the axis
    
    # Create 2D histogram / heatmap
    h, xedges, yedges, im = ax_heatmap.hist2d(x, y, bins=bins, cmap=cmap)
    
    # Add a colorbar in the top right position
    cbar = plt.colorbar(im, cax=ax_colorbar, orientation='vertical')
    cbar.set_label('Frequency')
    
    # Create top histogram
    ax_histx.hist(x, bins=bins, alpha=0.7, color='red')
    ax_histx.set_title(f'Joint Distribution of {ticker1} and {ticker2} {freq.capitalize()} Returns', fontsize=15)
    
    # Create right histogram
    ax_histy.hist(y, bins=bins, orientation='horizontal', alpha=0.7, color='red')
    
    # Set labels
    ax_heatmap.set_xlabel(f'{ticker1} Returns (%)', fontsize=12)
    ax_heatmap.set_ylabel(f'{ticker2} Returns (%)', fontsize=12)
    
    # Remove y-axis ticks from top histogram
    ax_histx.tick_params(axis='y', labelsize=8)
    ax_histx.yaxis.set_major_locator(MaxNLocator(nbins=4))
    
    # Remove x-axis ticks from right histogram
    ax_histy.tick_params(axis='x', labelsize=8)
    ax_histy.xaxis.set_major_locator(MaxNLocator(nbins=4))
    
    plt.tight_layout()
    plt.show()

def plot_kde_with_histograms(returns, ticker1, ticker2, freq='daily_log', bins=50, cmap='Spectral_r', n=200):
    """
    Plot 2D KDE with histograms for joint returns distribution
    
    Parameters:
    -----------
    returns : dict
        Dictionary with returns DataFrames for each ticker
    ticker1 : str
        First ticker symbol
    ticker2 : str
        Second ticker symbol
    freq : str
        Frequency of returns ('daily_log' or 'monthly_log')
    bins : int
        Number of bins for histograms
    cmap : str
        Colormap for KDE plot
    n : int
        Grid size for KDE
    """
    x = returns[ticker1][freq]
    y = returns[ticker2][freq]
    
    # Create a common index
    common_index = x.index.intersection(y.index)
    x = x.loc[common_index]
    y = y.loc[common_index]
    
    # Convert to numpy arrays
    x_np = x.values
    y_np = y.values
    
    # Set up figure and layout
    fig = plt.figure(figsize=(10, 8))
    
    # Create a gridspec for the figures
    gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3],
                         wspace=0.05, hspace=0.05)
    
    # Create the main KDE plot
    ax_kde = plt.subplot(gs[1, 0])
    
    # Create the top histogram
    ax_histx = plt.subplot(gs[0, 0], sharex=ax_kde)
    plt.setp(ax_histx.get_xticklabels(), visible=False)
    
    # Create the right histogram
    ax_histy = plt.subplot(gs[1, 1], sharey=ax_kde)
    plt.setp(ax_histy.get_yticklabels(), visible=False)
    
    # Create area for colorbar (in the top right corner)
    ax_colorbar = plt.subplot(gs[0, 1])
    ax_colorbar.axis('off')  # Hide the axis
    
    # Calculate histogram counts for proportional plotting
    hist_x, bins_x = np.histogram(x_np, bins=bins)
    hist_y, bins_y = np.histogram(y_np, bins=bins)
    top = max(np.max(hist_x), np.max(hist_y))
    
    # Calculate 2D KDE
    x_min, x_max = np.min(x_np), np.max(x_np)
    y_min, y_max = np.min(y_np), np.max(y_np)
    
    # Add margins to min/max
    x_margin = 0.1 * (x_max - x_min)
    y_margin = 0.1 * (y_max - y_min)
    x_min -= x_margin
    x_max += x_margin
    y_min -= y_margin
    y_max += y_margin
    
    # Create grid for KDE
    xx, yy = np.mgrid[x_min:x_max:n*1j, y_min:y_max:n*1j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    
    # Calculate KDE
    values = np.vstack([x_np, y_np])
    kernel = gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    
    # Plot KDE
    cmap = plt.get_cmap(cmap)
    im = ax_kde.imshow(np.rot90(f), cmap=cmap, extent=[x_min, x_max, y_min, y_max], aspect='auto')
    
    # Add a colorbar in the top right position
    cbar = plt.colorbar(im, cax=ax_colorbar, orientation='vertical')
    cbar.set_label('Density')
    
    # Plot histograms
    ax_histx.hist(x_np, bins=bins, alpha=0.7, color='red', edgecolor='black')
    ax_histy.hist(y_np, bins=bins, orientation='horizontal', alpha=0.7, color='red', edgecolor='black')
    
    # Set histogram scales
    ax_histx.set_ylim(0, top * 1.1)
    ax_histy.set_xlim(0, top * 1.1)
    
    # Set title and labels
    ax_histx.set_title(f'KDE of {ticker1} and {ticker2} {freq.capitalize()} Returns', fontsize=15)
    ax_kde.set_xlabel(f'{ticker1} Returns (%)', fontsize=12)
    ax_kde.set_ylabel(f'{ticker2} Returns (%)', fontsize=12)
    
    # Remove y-axis ticks from top histogram
    ax_histx.tick_params(axis='y', labelsize=8)
    ax_histx.yaxis.set_major_locator(MaxNLocator(nbins=4))
    
    # Remove x-axis ticks from right histogram
    ax_histy.tick_params(axis='x', labelsize=8)
    ax_histy.xaxis.set_major_locator(MaxNLocator(nbins=4))
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# MAIN FUNCTION
# =============================================================================

if __name__ == "__main__":
    print("\nPython version of JointDistributionReturns.R")
    print("=" * 70)
    
    # Set random seed for reproducibility
    np.random.seed(123)
    
    # Fetch stock data
    tickers = ['SPY', 'XLE']  # S&P 500 index and Energy sector ETF
    print(f"\nFetching data for {', '.join(tickers)} from Yahoo Finance...")
    price_data = fetch_stock_data(tickers)
    
    # Plot price series
    print("\nPlotting price series...")
    plot_price_series(price_data)
    
    # Calculate returns
    print("\nCalculating returns...")
    returns_data = calculate_returns(price_data)
    
    # Plot simple scatter plot
    print("\nPlotting scatter plot of returns...")
    plot_returns_scatter(returns_data, 'SPY', 'XLE', freq='daily_log')
    
    # Plot heatmap with histograms
    print("\nPlotting heatmap with histograms...")
    plot_heatmap_with_histograms(returns_data, 'SPY', 'XLE', freq='daily_log', bins=50)
    
    # Plot KDE with histograms
    print("\nPlotting KDE with histograms...")
    plot_kde_with_histograms(returns_data, 'SPY', 'XLE', freq='daily_log', bins=50, cmap='Spectral_r')
    
    print("\nNote: The Python version may show slight differences from the R version")
    print("due to differences in implementation details of the visualization libraries.")
    print("\nAnalysis complete!")