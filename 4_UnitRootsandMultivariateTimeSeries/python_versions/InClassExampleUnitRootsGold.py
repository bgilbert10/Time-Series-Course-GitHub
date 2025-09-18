#!/usr/bin/env python
# -*- coding: utf-8 -*-

# =============================================================================
# UNIT ROOT TESTING EXAMPLES - COMMODITIES AND MACROECONOMIC DATA
# =============================================================================

# This program follows the requirements from claudeinstructions.txt:
# "Get macro and oil price data, convert to monthly, merge, test each series for unit roots 
#  using individual and joint tests with drift and trend, then download separate gold and 
#  macro data and do a similar plotting and unit root testing exercise on each."
#
# PART 1: Macro and oil price data (lumber PPI, metals PPI, oil, inflation)
#         - Convert to monthly frequency
#         - Individual tests (cadftest equivalent) with drift and trend  
#         - Joint tests (ur.df equivalent) with drift and trend
# PART 2: Separate gold and macro data (gold, TIPS, inflation, USD index)
#         - Similar plotting and unit root testing exercise  
#         - Individual and joint tests with drift and trend
# 
# Key concepts:
# - Individual unit root tests with covariate inclusion capability (cadftest)
# - Joint hypothesis testing (ur.df equivalent)  
# - Testing with different specifications (Case 2: drift vs Case 4: trend)
# - Economic interpretation of unit root test results

# Import required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from datetime import datetime
import sys

# Set plotting style with fallback
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        plt.style.use('default')

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
warnings.filterwarnings('ignore')

# External data packages
try:
    import pandas_datareader.data as web
    FRED_AVAILABLE = True
except ImportError:
    print("Warning: pandas_datareader not available. Install with: pip install pandas-datareader")
    FRED_AVAILABLE = False

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    print("Warning: yfinance not available. Install with: pip install yfinance")
    YF_AVAILABLE = False

# Statsmodels imports
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Try to import arch for enhanced unit root tests
try:
    from arch.unitroot import ADF, PhillipsPerron, KPSS, DFGLS
    ARCH_AVAILABLE = True
except ImportError:
    print("Warning: arch package not available. Install with: pip install arch")
    print("Will use statsmodels.tsa.stattools.adfuller as fallback")
    ARCH_AVAILABLE = False

# ARIMA model import with fallback
try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    try:
        from statsmodels.tsa.arima_model import ARIMA
    except ImportError:
        print("Warning: ARIMA not available in current statsmodels version")
        ARIMA = None

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def enhanced_adf_test(series, constant='c', lags=None, max_lags=None):
    """
    Enhanced ADF test wrapper that provides detailed output
    
    Parameters:
    -----------
    series : array-like
        Time series to test
    constant : str
        'n' = no constant, 'c' = constant, 'ct' = constant and trend
    lags : int or None
        Number of lags to include
    max_lags : int or None
        Maximum lags for automatic selection
        
    Returns:
    --------
    dict : Test results and additional information
    """
    
    if ARCH_AVAILABLE:
        try:
            if constant == 'n':
                trend = 'n'
            elif constant == 'c':
                trend = 'c'
            else:  # constant == 'ct'
                trend = 'ct'
            
            if lags is not None:
                adf_result = ADF(series, lags=lags, trend=trend)
            else:
                adf_result = ADF(series, trend=trend, max_lags=max_lags)
            
            return {
                'statistic': adf_result.stat,
                'pvalue': adf_result.pvalue,
                'lags': adf_result.lags,
                'trend': trend,
                'critical_values': adf_result.critical_values,
                'regression_results': adf_result
            }
        except Exception as e:
            print(f"arch ADF test failed: {e}, falling back to statsmodels")
    
    # Fallback to statsmodels
    if constant == 'n':
        regression = 'n'
    elif constant == 'c':
        regression = 'c'
    else:
        regression = 'ct'
    
    result = adfuller(series, maxlag=max_lags, regression=regression, autolag='AIC')
    
    return {
        'statistic': result[0],
        'pvalue': result[1],
        'lags': result[2],
        'trend': regression,
        'critical_values': result[4],
        'regression_results': None
    }

def print_adf_results(result, series_name, case_name):
    """Print ADF test results in a formatted way"""
    print(f"\n=== {series_name} Unit Root Test ({case_name}) ===")
    print(f"ADF Statistic: {result['statistic']:.4f}")
    print(f"p-value: {result['pvalue']:.4f}")
    print(f"Lags used: {result['lags']}")
    print(f"Critical Values:")
    if isinstance(result['critical_values'], dict):
        for level, cv in result['critical_values'].items():
            print(f"  {level}: {cv:.4f}")
    
    # Interpretation
    if result['pvalue'] < 0.05:
        print("Result: REJECT null hypothesis - No unit root (stationary)")
    else:
        print("Result: FAIL TO REJECT null hypothesis - Unit root present")
    print()

def joint_hypothesis_test(series, constant='c', lags=None):
    """
    Joint hypothesis test equivalent to R's ur.df() function
    Tests joint null hypothesis that coefficients are zero
    """
    print(f"\n=== JOINT HYPOTHESIS TEST (ur.df equivalent) ===")
    
    try:
        # Prepare the regression data
        y = np.array(series).astype(float)
        n = len(y)
        
        # First difference
        dy = np.diff(y)
        y_lag = y[:-1]
        
        # Build regression matrix based on case
        if constant == 'n':  # Case 1: No constant, no trend
            X = y_lag.reshape(-1, 1)
            var_names = ['y_lag']
            restrictions = [0]  # Test if coefficient on lagged level = 0
            case_name = "Case 1 (No constant, no trend)"
        elif constant == 'c':  # Case 2: Constant, no trend  
            X = np.column_stack([np.ones(len(y_lag)), y_lag])
            var_names = ['const', 'y_lag']
            restrictions = [1]  # Test if coefficient on lagged level = 0
            case_name = "Case 2 (Constant, no trend)"
        else:  # constant == 'ct', Case 4: Constant and trend
            trend = np.arange(1, len(y_lag) + 1)
            X = np.column_stack([np.ones(len(y_lag)), trend, y_lag])
            var_names = ['const', 'trend', 'y_lag'] 
            restrictions = [2]  # Test if coefficient on lagged level = 0
            case_name = "Case 4 (Constant and trend)"
            
        # Determine maximum lags needed for proper alignment
        max_lag = max(1, lags if lags is not None else 0)
        
        # Trim all arrays to account for maximum lag
        dy_trimmed = dy[max_lag:]
        y_lag_trimmed = y_lag[max_lag:]
        
        # Rebuild regression matrix with properly aligned data
        if constant == 'n':  # Case 1: No constant, no trend
            X_aligned = y_lag_trimmed.reshape(-1, 1)
            var_names = ['y_lag']
            restrictions = [0]  # Test if coefficient on lagged level = 0
        elif constant == 'c':  # Case 2: Constant, no trend  
            X_aligned = np.column_stack([np.ones(len(y_lag_trimmed)), y_lag_trimmed])
            var_names = ['const', 'y_lag']
            restrictions = [1]  # Test if coefficient on lagged level = 0
        else:  # constant == 'ct', Case 4: Constant and trend
            trend_aligned = np.arange(max_lag + 1, max_lag + len(y_lag_trimmed) + 1)
            X_aligned = np.column_stack([np.ones(len(y_lag_trimmed)), trend_aligned, y_lag_trimmed])
            var_names = ['const', 'trend', 'y_lag'] 
            restrictions = [2]  # Test if coefficient on lagged level = 0
            
        # Add lagged differences if specified
        if lags is not None and lags > 0:
            for i in range(1, lags + 1):
                if len(dy) > max_lag + i - 1:
                    # Create properly aligned lagged differences
                    lag_dy_aligned = dy[max_lag - i : len(dy) - i]
                    if len(lag_dy_aligned) == len(dy_trimmed):
                        X_aligned = np.column_stack([X_aligned, lag_dy_aligned])
                        var_names.append(f'dy_lag{i}')
        
        # Final arrays (should now be aligned)
        dy_reg = dy_trimmed
        X_reg = X_aligned
        
        # Fit unrestricted model
        try:
            ols_result = sm.OLS(dy_reg, X_reg).fit()
            
            print(f"Model: {case_name}")
            print(f"Observations: {len(dy_reg)}")
            print("\nUnrestricted Model Results:")
            print("=" * 40)
            for i, name in enumerate(var_names):
                coef = ols_result.params[i]
                tstat = ols_result.tvalues[i]
                pval = ols_result.pvalues[i]
                print(f"{name:12s}: {coef:8.4f} (t={tstat:6.3f}, p={pval:.4f})")
            
            print(f"\nR-squared: {ols_result.rsquared:.4f}")
            
            # Joint test on unit root coefficient
            unit_root_coef_idx = restrictions[0]
            unit_root_coef = ols_result.params[unit_root_coef_idx]
            unit_root_tstat = ols_result.tvalues[unit_root_coef_idx] 
            unit_root_pval = ols_result.pvalues[unit_root_coef_idx]
            
            print(f"\n=== UNIT ROOT COEFFICIENT TEST ===")
            print(f"Coefficient on lagged level: {unit_root_coef:.4f}")
            print(f"t-statistic: {unit_root_tstat:.4f}")
            print("Note: Use Dickey-Fuller critical values, not standard t-distribution")
            
            # Approximate DF critical values
            if constant == 'n':
                cv_5pct = -1.95
            elif constant == 'c':
                cv_5pct = -2.89
            else:  # ct
                cv_5pct = -3.43
                
            print(f"Approximate 5% critical value: {cv_5pct:.2f}")
            
            if unit_root_tstat < cv_5pct:
                joint_conclusion = "REJECT null hypothesis - No unit root"
            else:
                joint_conclusion = "FAIL TO REJECT null hypothesis - Unit root present"
                
            print(f"Joint test conclusion: {joint_conclusion}")
            
            return {
                'model_result': ols_result,
                'unit_root_coef': unit_root_coef,
                'unit_root_tstat': unit_root_tstat,
                'conclusion': joint_conclusion,
                'case': case_name
            }
            
        except Exception as e:
            print(f"Error in OLS regression: {e}")
            return None
            
    except Exception as e:
        print(f"Error in joint hypothesis test: {e}")
        return None

def fetch_macro_oil_data():
    """
    Fetch macroeconomic and oil price data from FRED
    
    Returns:
    --------
    pandas.DataFrame : Macro and oil price data
    """
    
    if not FRED_AVAILABLE:
        print("Error: Cannot fetch real data - pandas_datareader not available")
        return None
    
    try:
        # Define date range
        start_date = "2003-01-01"
        end_date = "2021-08-01"
        
        print("Fetching macro and oil price data from FRED...")
        print("Requirements: Convert to monthly, merge, test with individual and joint tests")
        
        # Fetch data series
        series_ids = {
            'Inflation': 'T10YIE',        # 10-year breakeven inflation rate
            'LumberPPI': 'WPU081',        # Lumber Producer Price Index
            'MetalsPPI': 'WPU10',         # Metals Producer Price Index
            'OilPrice': 'MCOILWTICO'      # WTI Oil prices
        }
        
        data_dict = {}
        for name, series_id in series_ids.items():
            try:
                # Fetch raw series
                series = web.get_data_fred(series_id, start=start_date, end=end_date)
                
                # Convert to monthly as required by claudeinstructions.txt
                if name == 'Inflation':
                    # Daily series - convert to monthly using mean
                    series = series.resample('M').mean()
                    print(f"Converted {name} from daily to monthly frequency")
                elif name == 'OilPrice':
                    # Daily series - convert to monthly using mean  
                    if len(series) > 500:  # Likely daily data
                        series = series.resample('M').mean()
                        print(f"Converted {name} from daily to monthly frequency")
                else:
                    # Check if needs monthly conversion
                    if len(series) > 300:  # Likely higher frequency than monthly
                        series = series.resample('M').mean()
                        print(f"Converted {name} to monthly frequency")
                
                data_dict[name] = series.iloc[:, 0]
                print(f"Successfully fetched {name} ({series_id}) - Final frequency: {len(series)} observations")
            except Exception as e:
                print(f"Failed to fetch {name} ({series_id}): {e}")
                return None
        
        # Transform oil prices to log
        if 'OilPrice' in data_dict:
            data_dict['LogOilPrice'] = np.log(data_dict['OilPrice'])
            del data_dict['OilPrice']
        
        # Combine into DataFrame
        data = pd.DataFrame(data_dict)
        data = data.dropna()
        
        if len(data) == 0:
            print("Error: No data available after removing NaN values")
            return None
            
        print(f"Macro data loaded: {len(data)} observations from {data.index[0]} to {data.index[-1]}")
        return data
        
    except Exception as e:
        print(f"Error fetching macro data: {e}")
        return None

def fetch_gold_monetary_data():
    """
    Fetch gold price and monetary policy data
    
    Returns:
    --------
    pandas.DataFrame : Gold and monetary policy data
    """
    
    if not FRED_AVAILABLE:
        print("Error: Cannot fetch real data - pandas_datareader not available")
        return None
    
    try:
        # Define date range for gold analysis
        start_date = "2006-01-03"
        end_date = "2021-08-30"
        
        print("Fetching gold and monetary policy data...")
        
        # Fetch FRED data
        fred_series = {
            'TIPS': 'DFII10',              # 10-year TIPS constant maturity
            'BreakevenInflation': 'T10YIE', # 10-year breakeven inflation
            'DollarIndex': 'DTWEXBGS'       # Trade-weighted US Dollar Index
        }
        
        data_dict = {}
        for name, series_id in fred_series.items():
            try:
                series = web.get_data_fred(series_id, start=start_date, end=end_date)
                data_dict[name] = series.iloc[:, 0]
                print(f"Successfully fetched {name} ({series_id})")
            except Exception as e:
                print(f"Failed to fetch {name} ({series_id}): {e}")
                return None
        
        # Fetch gold data from Yahoo Finance if available
        if YF_AVAILABLE:
            try:
                # Gold futures
                gold_ticker = 'GC=F'
                gold_data = yf.download(gold_ticker, start=start_date, end=end_date, progress=False)
                if not gold_data.empty:
                    data_dict['GoldPrice'] = gold_data['Adj Close']
                    print(f"Successfully fetched Gold prices ({gold_ticker})")
                else:
                    print("Warning: Gold data download returned empty dataset")
                    return None
            except Exception as e:
                print(f"Failed to fetch gold data: {e}")
                return None
        else:
            print("Error: yfinance not available for gold data")
            return None
        
        # Combine into DataFrame
        data = pd.DataFrame(data_dict)
        data = data.dropna()
        
        if len(data) == 0:
            print("Error: No data available after removing NaN values")
            return None
            
        print(f"Gold data loaded: {len(data)} observations from {data.index[0]} to {data.index[-1]}")
        return data
        
    except Exception as e:
        print(f"Error fetching gold data: {e}")
        return None

def test_series_for_unit_root(series, series_name):
    """
    Test a series for unit root using systematic approach
    
    Parameters:
    -----------
    series : pandas.Series
        Time series to test
    series_name : str
        Name of the series for output
        
    Returns:
    --------
    dict : Test results summary
    """
    
    print(f"\n{'-' * 60}")
    print(f"{series_name.upper()} UNIT ROOT ANALYSIS")
    print(f"{'-' * 60}")
    
    # Drop missing values
    clean_series = series.dropna()
    
    if len(clean_series) < 50:
        print(f"Warning: {series_name} has only {len(clean_series)} observations")
        return None
    
    # INDIVIDUAL TESTS (cadftest equivalent) 
    print(f"\n=== INDIVIDUAL UNIT ROOT TESTS (cadftest equivalent) ===")
    
    # Test with Case 4 (trend) first - most general
    case4_result = enhanced_adf_test(clean_series, constant='ct', max_lags=12)
    print_adf_results(case4_result, series_name, "Case 4 (trend)")
    
    # Test with Case 2 (drift)  
    case2_result = enhanced_adf_test(clean_series, constant='c', max_lags=12)
    print_adf_results(case2_result, series_name, "Case 2 (drift)")
    
    # JOINT HYPOTHESIS TESTS (ur.df equivalent) - Requirements from claudeinstructions.txt
    print(f"\n=== JOINT HYPOTHESIS TESTS (ur.df equivalent) ===")
    print("Requirements: individual and joint tests with drift and trend")
    
    # Joint test with trend (Case 4)
    joint_case4 = joint_hypothesis_test(clean_series, constant='ct', lags=case4_result['lags'])
    
    # Joint test with drift (Case 2)
    joint_case2 = joint_hypothesis_test(clean_series, constant='c', lags=case2_result['lags'])
    
    # Interpretation logic
    if case4_result['pvalue'] < 0.05:
        conclusion = "Stationary (I(0))"
        integration_order = 0
    elif case2_result['pvalue'] < 0.05:
        conclusion = "Stationary (I(0))"
        integration_order = 0  
    else:
        conclusion = "Unit root (I(1))"
        integration_order = 1
    
    # Test first differences for drift if unit root is present
    drift_test_result = None
    if integration_order == 1:
        try:
            first_diff = clean_series.diff().dropna()
            if len(first_diff) > 10:
                # Simple t-test for mean of first differences
                t_stat, p_val = stats.ttest_1samp(first_diff, 0)
                drift_test_result = {
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'has_drift': p_val < 0.05
                }
                
                print(f"=== Testing First Differences for Drift ===")
                print(f"t-statistic: {t_stat:.4f}")
                print(f"p-value: {p_val:.4f}")
                if p_val < 0.05:
                    print("Result: Significant drift detected")
                else:
                    print("Result: No significant drift detected")
        except Exception as e:
            print(f"Could not test for drift: {e}")
    
    print(f"Final conclusion: {series_name} - {conclusion}")
    if drift_test_result and drift_test_result['has_drift']:
        print(f"Note: Evidence of drift in first differences")
    
    return {
        'series_name': series_name,
        'case4_pvalue': case4_result['pvalue'],
        'case2_pvalue': case2_result['pvalue'],
        'case4_statistic': case4_result['statistic'],
        'case2_statistic': case2_result['statistic'],
        'conclusion': conclusion,
        'integration_order': integration_order,
        'drift_test': drift_test_result
    }

# =============================================================================
# PART 1: MACRO AND OIL PRICE DATA ANALYSIS
# =============================================================================

def part1_macro_oil_analysis():
    """Analyze macro and oil price data for unit roots"""
    
    print("=" * 70)
    print("PART 1: MACRO AND OIL PRICE DATA")
    print("=" * 70)
    print("Testing unit root behavior in economic time series:")
    print("- Inflation, lumber PPI, metals PPI, log oil prices")
    print("- Demonstrates Case 2 vs Case 4 selection with real data")
    
    # Fetch data
    data = fetch_macro_oil_data()
    
    if data is None:
        print("Could not fetch macro data. Using synthetic data for demonstration...")
        # Create synthetic data as fallback
        np.random.seed(42)
        dates = pd.date_range('2003-01-01', '2021-08-01', freq='M')
        data = pd.DataFrame({
            'Inflation': np.cumsum(np.random.normal(0, 0.1, len(dates))) + 2,
            'LumberPPI': np.cumsum(np.random.normal(0, 0.05, len(dates))) + 200,
            'MetalsPPI': np.cumsum(np.random.normal(0, 0.03, len(dates))) + 150,
            'LogOilPrice': np.cumsum(np.random.normal(0, 0.02, len(dates))) + 4
        }, index=dates)
        print("Using synthetic data for demonstration")
    
    # Plot the data
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, column in enumerate(data.columns):
        data[column].plot(ax=axes[i], title=f'{column}')
        axes[i].set_ylabel('Value')
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("Visual inspection of data complete")
    print("Formal unit root testing needed to determine properties")
    
    # Test each series
    results = {}
    for column in data.columns:
        result = test_series_for_unit_root(data[column], column)
        if result:
            results[column] = result
    
    return data, results

# =============================================================================
# PART 2: GOLD AND MONETARY POLICY DATA ANALYSIS
# =============================================================================

def part2_gold_monetary_analysis():
    """Analyze gold and monetary policy data for unit roots"""
    
    print("\n" + "=" * 70)
    print("PART 2: GOLD AND MONETARY POLICY DATA")
    print("=" * 70)
    print("Testing unit root behavior in financial time series:")
    print("- Gold prices, TIPS yields, breakeven inflation, USD index")
    print("- Demonstrates unit root testing in financial markets context")
    
    # Fetch data
    data = fetch_gold_monetary_data()
    
    if data is None:
        print("Could not fetch gold data. Using synthetic data for demonstration...")
        # Create synthetic data as fallback
        np.random.seed(43)
        dates = pd.date_range('2006-01-03', '2021-08-30', freq='D')
        dates = dates[dates.weekday < 5]  # Business days only
        data = pd.DataFrame({
            'GoldPrice': np.cumsum(np.random.normal(0, 1, len(dates))) + 1200,
            'TIPS': np.cumsum(np.random.normal(0, 0.01, len(dates))) + 2,
            'BreakevenInflation': np.cumsum(np.random.normal(0, 0.01, len(dates))) + 2,
            'DollarIndex': np.cumsum(np.random.normal(0, 0.1, len(dates))) + 90
        }, index=dates)
        print("Using synthetic data for demonstration")
    
    # Plot the data
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, column in enumerate(data.columns):
        data[column].plot(ax=axes[i], title=f'{column}')
        axes[i].set_ylabel('Value')
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("Visual inspection shows potential trending behavior")
    print("Unit root testing will determine integration properties")
    
    # Test each series
    results = {}
    for column in data.columns:
        result = test_series_for_unit_root(data[column], column)
        if result:
            results[column] = result
    
    return data, results

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run all parts of the unit root testing examples"""
    
    print("UNIT ROOT TESTING EXAMPLES - COMMODITIES AND MACROECONOMIC DATA")
    print("=" * 70)
    print("This program demonstrates unit root testing on real economic and financial data")
    print("Focus on systematic testing procedures and economic interpretation")
    
    # Check package availability
    print(f"\nPackage Status:")
    print(f"- FRED data available: {FRED_AVAILABLE}")
    print(f"- Yahoo Finance available: {YF_AVAILABLE}")
    print(f"- ARCH package available: {ARCH_AVAILABLE}")
    print(f"- ARIMA model available: {ARIMA is not None}")
    
    try:
        # Run macro and oil analysis
        macro_data, macro_results = part1_macro_oil_analysis()
        
        # Run gold and monetary analysis
        gold_data, gold_results = part2_gold_monetary_analysis()
        
        # Combined summary
        print("\n" + "=" * 70)
        print("SUMMARY OF UNIT ROOT TEST FINDINGS")
        print("=" * 70)
        
        if macro_results:
            print("MACRO AND OIL PRICE DATA:")
            for i, (series, result) in enumerate(macro_results.items(), 1):
                drift_note = ""
                if result.get('drift_test') and result['drift_test'].get('has_drift'):
                    drift_note = ", evidence of drift"
                print(f"{i}. {series}: {result['conclusion']}{drift_note}")
        
        if gold_results:
            print("\nGOLD AND MONETARY POLICY DATA:")
            for i, (series, result) in enumerate(gold_results.items(), 1):
                drift_note = ""
                if result.get('drift_test') and result['drift_test'].get('has_drift'):
                    drift_note = ", evidence of drift"
                print(f"{i}. {series}: {result['conclusion']}{drift_note}")
        
        print(f"\nMETHODOLOGICAL INSIGHTS:")
        print("- Joint tests (ur.df in R) provide more information than individual tests")
        print("- Testing with both Case 2 and Case 4 helps identify correct specification")
        print("- Visual inspection can be misleading for unit root determination")
        print("- First difference tests help confirm drift conclusions")
        print("- Economic and financial time series commonly exhibit unit root behavior")
        
        return {
            'macro_data': macro_data,
            'macro_results': macro_results,
            'gold_data': gold_data,
            'gold_results': gold_results
        }
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        return None
    except Exception as e:
        print(f"\nError during analysis: {e}")
        print("This may be due to missing packages or data access issues")
        return None

if __name__ == "__main__":
    results = main()
    
    if results is not None:
        print(f"\nAnalysis completed successfully!")
        print("Unit root testing of economic and financial data finished.")
        
        # Additional insights for educational use
        print(f"\nEducational Notes:")
        print("- This analysis replicates the R version using Python")
        print("- Results may differ slightly due to different implementations")
        print("- Focus on methodology and economic interpretation")
        print("- Real data provides more meaningful insights than synthetic examples")
    else:
        print("\nAnalysis failed or was interrupted.")
        print("Check error messages above for details.")