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
import io
from io import BytesIO
import sys
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from IPython.display import HTML, display
import base64

# Set plots to be displayed inline (for Jupyter notebooks)
# %matplotlib inline

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Create a class to capture console output
class OutputCapture:
    def __init__(self, filename=None):
        self.terminal = sys.stdout
        self.output = io.StringIO()
        self.filename = filename
        
    def start(self):
        sys.stdout = self
        
    def stop(self):
        sys.stdout = self.terminal
        
    def write(self, message):
        # Handle Unicode characters safely for Windows
        try:
            self.terminal.write(message)
        except UnicodeEncodeError:
            # Replace problematic characters with ASCII equivalents
            safe_message = message.replace('μ', 'mu').replace('σ', 'sigma')
            self.terminal.write(safe_message)
        
        # Always store the original message in the buffer
        self.output.write(message)
        
    def flush(self):
        self.terminal.flush()
        
    def get_output(self):
        # Replace any Unicode characters that might cause issues when writing to HTML
        output = self.output.getvalue()
        # Replace common Greek letters used in statistics
        output = output.replace('μ', 'mu').replace('σ', 'sigma')
        output = output.replace('α', 'alpha').replace('β', 'beta')
        output = output.replace('ρ', 'rho').replace('τ', 'tau')
        return output
    
    def save_to_file(self, filename=None):
        if filename is None:
            filename = self.filename
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.get_output())
            print(f"Output saved to {filename}")

# Dictionary to store all figures for the report
all_figures = {}

"""    
def get_output(self):
    return self.output.getvalue()

def save_to_file(self, filename=None):
    if filename is None:
        filename = self.filename
    if filename:
        with open(filename, 'w') as f:
            f.write(self.output.getvalue())
        print(f"Output saved to {filename}")
        
# Dictionary to store all figures
all_figures = {}
"""

# --------------------- SETUP AND HELPER FUNCTIONS ----------------------

def fetch_stock_data(ticker, start_date='2000-01-03', end_date=None):
    """
    Fetch stock data using yfinance
    Works for one ticker at a time. 
    Needs modifying for multiple tickers at once.
    
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
    
    data = yf.download(ticker, start=start_date, end=end_date, group_by='ticker')
    
    # Handle MultiIndex if present
    data = data[ticker]
    
    print(f"Final DataFrame columns: {data.columns}")
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
    # Print available columns for debugging
    print(f"Available columns in prices_df: {prices_df.columns}")
    
    # Use 'Close' column since that's what yfinance provides
    if 'Close' in prices_df.columns:
        price_col = 'Close'
    elif 'Adj Close' in prices_df.columns:
        price_col = 'Adj Close'
    else:
        raise ValueError("Neither 'Close' nor 'Adj Close' found in DataFrame columns")
    
    print(f"Using column: {price_col}")
    
    # Ensure the price data is numeric
    prices = pd.to_numeric(prices_df[price_col], errors='coerce')
    prices = prices.dropna()
    print(f"First few prices: {prices.head()}")
    
    # Calculate simple returns
    daily_simple_returns = 100 * prices.pct_change().dropna()
    
    # Calculate log returns
    daily_log_returns = 100 * np.log(prices / prices.shift(1)).dropna()
    
    # Calculate monthly returns
    # Resample to monthly frequency and calculate returns
    monthly_prices = prices.resample('M').last()
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
    # Ensure returns is numeric
    if isinstance(returns, pd.Series):
        # Make a copy to avoid modifying the original
        returns_numeric = returns.copy()
        # Convert to numeric if needed
        returns_numeric = pd.to_numeric(returns_numeric, errors='coerce')
        # Drop any NaN values that resulted from conversion
        returns_numeric = returns_numeric.dropna()
    else:
        # Convert to numpy array and ensure it's numeric
        returns_numeric = np.array(returns, dtype=float)
        # Remove NaNs
        returns_numeric = returns_numeric[~np.isnan(returns_numeric)]
    
    # Print diagnostic info
    print(f"After ensuring numeric: Type={type(returns_numeric)}, Length={len(returns_numeric)}")
    
    stats_dict = {
        'nobs': len(returns_numeric),
        'min': returns_numeric.min(),
        'max': returns_numeric.max(),
        'mean': returns_numeric.mean(),
        'median': np.median(returns_numeric),
        'std_dev': returns_numeric.std(),
        'variance': returns_numeric.var(),
        'skewness': float(skew(returns_numeric)),
        'kurtosis': float(kurtosis(returns_numeric, fisher=True)),  # Excess kurtosis (normal = 0)
        '5%': float(np.percentile(returns_numeric, 5)),
        '95%': float(np.percentile(returns_numeric, 95))
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
    
    print(f"T-test of mean returns (H0: mu = {mu})")
    print(f"t-statistic: {float(t_stat):.4f}")
    print(f"p-value ({alternative}): {float(p_value):.4f}")
    print(f"Mean: {float(returns.mean()):.4f}")
    print(f"Std Error: {float(returns.std()/np.sqrt(len(returns))):.4f}")
    
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
    # Ensure returns is numeric
    if isinstance(returns, pd.Series):
        # Make a copy to avoid modifying the original
        returns = returns.copy()
        # Convert to numeric if needed
        returns = pd.to_numeric(returns, errors='coerce')
        # Drop any NaN values that resulted from conversion
        returns = returns.dropna()
    else:
        # Convert to numpy array and ensure it's numeric
        returns = np.array(returns, dtype=float)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot histogram
    axes[0].hist(returns, bins=40, density=True, alpha=0.6, color='steelblue')
    axes[0].set_title(f"{title} - Histogram")
    axes[0].set_xlabel("Returns (%)")
    axes[0].set_ylabel("Frequency")
    
    # Add normal curve to histogram - using numpy min/max for safety
    x = np.linspace(np.nanmin(returns), np.nanmax(returns), 100)
    mean, std = np.nanmean(returns), np.nanstd(returns)
    normal_curve = stats.norm.pdf(x, mean, std)
    axes[0].plot(x, normal_curve, 'r-', linewidth=2, label='Normal Distribution')
    axes[0].legend()
    
    # Plot empirical density
    # If it's already a Series, skip conversion to avoid issues
    if not isinstance(returns, pd.Series):
        returns_series = pd.Series(returns)
    else:
        returns_series = returns
    
    returns_series.plot.kde(ax=axes[1], linewidth=2, label='Empirical')
    
    # Add normal curve to density plot - using same x range as above
    axes[1].plot(x, normal_curve, 'r--', linewidth=2, label='Normal')
    
    axes[1].set_title(f"{title} - Density")
    axes[1].set_xlabel("Returns (%)")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    
    plt.tight_layout()
    
    # Store the figure in the global dictionary
    global all_figures
    fig_id = f"dist_{title.replace(' ', '_').lower()}"
    all_figures[fig_id] = fig
    
    # Show the plot in Spyder
    plt.show()
    
    return fig

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
    print(f"Statistic: {float(jb_stat):.4f}")
    print(f"p-value: {float(jb_pvalue):.6f}")
    print(f"Conclusion: {'Reject' if jb_pvalue < 0.05 else 'Fail to reject'} normality at 5% significance level")
    
    # Skewness test
    s = skew(returns)
    n = len(returns)
    skew_stat = s / np.sqrt(6/n)
    skew_pvalue = 2 * (1 - norm.cdf(abs(skew_stat)))
    
    print(f"\nSkewness Test:")
    print(f"Skewness: {float(s):.4f}")
    print(f"Test statistic: {float(skew_stat):.4f}")
    print(f"p-value: {float(skew_pvalue):.6f}")
    
    # Kurtosis test
    k = kurtosis(returns, fisher=True)  # Excess kurtosis (normal = 0)
    kurt_stat = k / np.sqrt(24/n)
    
    print(f"\nExcess Kurtosis Test:")
    print(f"Excess Kurtosis: {float(k):.4f}")
    print(f"Test statistic: {float(kurt_stat):.4f}")
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
    
    # Debug information
    print(f"Type of simple_ibm_returns: {type(simple_ibm_returns)}")
    print(f"First few values: {simple_ibm_returns.head() if hasattr(simple_ibm_returns, 'head') else simple_ibm_returns[:5]}")
    
    # Convert to numeric if needed
    if isinstance(simple_ibm_returns, pd.Series):
        simple_ibm_returns = pd.to_numeric(simple_ibm_returns, errors='coerce')
        print(f"After conversion - First few values: {simple_ibm_returns.head()}")
    
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
    xle_data['Close'].plot()
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

# -------------------- REPORT GENERATION -------------------------

def figure_to_base64(fig):
    """Convert a matplotlib figure to base64 encoded string"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

def generate_html_report(output_text, figures_dict, filename='returns_analysis_report.html'):
    """Generate an HTML report with all output and figures"""
    # Process the output text to handle any Unicode characters (already done in OutputCapture)
    # HTML escape the output text
    from html import escape
    escaped_output = escape(output_text)
    
    html_content = f"""<!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Returns Analysis Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 40px;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            pre {{
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
                white-space: pre-wrap;
                font-family: monospace;
            }}
            .figure {{
                margin: 20px 0;
                text-align: center;
            }}
            .figure img {{
                max-width: 100%;
                height: auto;
            }}
            .figure-caption {{
                margin-top: 10px;
                font-style: italic;
                color: #666;
            }}
            .section {{
                margin-bottom: 40px;
                border-bottom: 1px solid #eee;
                padding-bottom: 20px;
            }}
        </style>
    </head>
    <body>
        <h1>Returns Distribution Analysis Report</h1>
        <div class="section">
            <h2>Console Output</h2>
            <pre>{escaped_output}</pre>
        </div>
        <div class="section">
            <h2>Figures</h2>
    """
    
    # Add all figures to the HTML
    for fig_id, fig in figures_dict.items():
        img_str = figure_to_base64(fig)
        fig_title = fig_id.replace('_', ' ').title()
        html_content += f"""
            <div class="figure">
                <h3>{fig_title}</h3>
                <img src="data:image/png;base64,{img_str}" alt="{fig_title}">
            </div>
        """
    
    # Add the current timestamp for the footer
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html_content += f"""
        </div>
        <footer>
            <p>Generated on {current_time}</p>
        </footer>
    </body>
    </html>
    """
    
    # Save the HTML file with UTF-8 encoding to handle Unicode characters
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\nHTML report saved to: {os.path.abspath(filename)}")

# -------------------- EXAMPLE USAGE -------------------------

if __name__ == "__main__":
    # Initialize output capture
    output_capture = OutputCapture()
    output_capture.start()
    
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
    
    # Stop capturing output
    output_capture.stop()
    
    # Generate HTML report with all output and figures
    generate_html_report(output_capture.get_output(), all_figures)
    
    print("\nAnalysis complete! The HTML report contains all results and figures.")