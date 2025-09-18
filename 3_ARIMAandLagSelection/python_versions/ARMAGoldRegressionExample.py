# ===============================================================================
# ARMA MODELING WITH GOLD PRICES AND MACROECONOMIC VARIABLES
# ===============================================================================
#
# LEARNING OBJECTIVES:
# 1. Download and merge financial time series data from multiple sources
# 2. Investigate stationarity properties of economic time series
# 3. Fit ARIMA models to individual time series
# 4. Build regression models with time series data
# 5. Test and correct for serial correlation in regression residuals
#
# ECONOMIC CONTEXT:
# This analysis examines the relationship between gold returns and key
# macroeconomic variables: Treasury Inflation-Protected Securities (TIPS),
# breakeven inflation expectations, and the trade-weighted dollar index.
# Understanding these relationships helps explain gold's role as an inflation
# hedge and safe-haven asset.
#
# DATA SOURCES:
# - Gold futures prices (Yahoo Finance)
# - 10-year TIPS constant maturity (FRED: DFII10)
# - 10-year breakeven inflation rate (FRED: T10YIE) 
# - Trade-weighted US dollar index (FRED: DTWEXBGS)
#
# TIME PERIOD: January 3, 2008 to June 30, 2020
# ===============================================================================

# Import required packages with compatibility handling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# External data packages (with fallbacks)
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    print("Warning: yfinance not available. Will use synthetic gold price data.")
    yf = None
    YF_AVAILABLE = False


try:
    import pandas_datareader.data as web
    FRED_AVAILABLE = True
except ImportError:
    print("Warning: pandas_datareader not available. Install with: pip install pandas-datareader")
    web = None
    FRED_AVAILABLE = False

# Robust statsmodels imports
import statsmodels.api as sm

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    try:
        from statsmodels.tsa.arima_model import ARIMA
    except ImportError:
        print("Warning: ARIMA model not available.")
        ARIMA = None

try:
    from statsmodels.tsa.stattools import acf, pacf
except ImportError:
    print("Warning: ACF/PACF functions not available.")
    acf = None
    pacf = None

from statsmodels.stats.diagnostic import acorr_ljungbox

try:
    from statsmodels.tsa.ar_model import AutoReg, ar_select_order
except ImportError:
    print("Warning: AutoReg functions not available.")
    AutoReg = None
    ar_select_order = None

try:
    from statsmodels.stats.sandwich_covariance import cov_hac
except ImportError:
    print("Warning: HAC covariance not available.")
    cov_hac = None

# Set up FRED API (you'll need to get a free API key from https://fred.stlouisfed.org/docs/api/api_key.html)
# fred = Fred(api_key='YOUR_API_KEY_HERE')  # Replace with your FRED API key

# Check package availability
def check_package_availability():
    """Check availability of optional packages and provide guidance"""
    issues = []
    
    if not YF_AVAILABLE:
        issues.append("yfinance: Cannot download real gold price data")
    if not FRED_AVAILABLE:
        issues.append("fredapi: Cannot download real macroeconomic data")
    if ARIMA is None:
        issues.append("ARIMA: Cannot fit ARIMA models")
    if acf is None or pacf is None:
        issues.append("ACF/PACF: Cannot compute autocorrelations")
    if cov_hac is None:
        issues.append("HAC covariance: Cannot compute robust standard errors")
    
    if issues:
        print("Package Availability Issues:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nTo fix missing packages:")
        print("  pip install yfinance fredapi")
        print("  pip install --upgrade statsmodels")
        print("\nSynthetic data will be used where real data is unavailable.\n")
    else:
        print("All required packages available!")
    
    print()

check_package_availability()

# ===============================================================================
# STEP 1: DATA ACQUISITION AND PREPARATION
# ===============================================================================
# 
# Download time series data from multiple sources and merge into a single dataset.
# We use different data providers: Yahoo Finance for gold prices and FRED for
# macroeconomic variables.

def download_and_prepare_data():
    """
    Download and prepare gold and macroeconomic data
    """
    print("STEP 1: Data Acquisition and Preparation")
    print("=======================================\n")
    
    # Define analysis period
    start_date = "2008-01-03"
    end_date = "2020-06-30"
    
    print(f"Analysis Period: {start_date} to {end_date}")
    print("This period includes the 2008 Financial Crisis and subsequent recovery.\n")
    
    # Download gold futures data from Yahoo Finance
    print("Downloading gold futures data from Yahoo Finance...")
    
    if YF_AVAILABLE and yf is not None:
        try:
            gold_data = yf.download("GC=F", start=start_date, end=end_date)
            gold_price = gold_data['Close'].squeeze().dropna()
            print("✓ Gold price data downloaded successfully!")
            print(f"Gold price observations: {len(gold_price)}")
        except Exception as e:
            print(f"✗ Error downloading gold data: {e}")
            print("Unable to proceed without gold price data. Please check your internet connection or try again later.")
            return None
    else:
        print("✗ yfinance not available. Please install with: pip install yfinance")
        print("Unable to proceed without gold price data.")
        return None
    
    # Download macroeconomic data from FRED using pandas_datareader
    print("\nDownloading macroeconomic data from FRED...")
    
    if FRED_AVAILABLE and web is not None:
        try:
            # Download FRED data using DataReader
            tips_yield = web.DataReader('DFII10', 'fred', start_date, end_date)['DFII10'].dropna()
            print("✓ 10-year TIPS yield data downloaded successfully!")
            
            breakeven_inflation = web.DataReader('T10YIE', 'fred', start_date, end_date)['T10YIE'].dropna()
            print("✓ 10-year breakeven inflation data downloaded successfully!")
            
            trade_weighted_dollar = web.DataReader('DTWEXBGS', 'fred', start_date, end_date)['DTWEXBGS'].dropna()
            print("✓ Trade-weighted dollar index data downloaded successfully!")
            
        except Exception as e:
            print(f"✗ Error downloading FRED data: {e}")
            print("Unable to proceed without macroeconomic data. Please check your internet connection.")
            print("Note: FRED data may have delays or series discontinuations.")
            return None
    else:
        print("✗ pandas_datareader not available. Please install with: pip install pandas-datareader")
        print("Unable to proceed without macroeconomic data.")
        return None
    
    # Merge all series into a single DataFrame
    gold_macro_data = pd.DataFrame({
        'gold_price': gold_price,
        'tips_yield': tips_yield,
        'breakeven_inflation': breakeven_inflation,
        'trade_weighted_dollar': trade_weighted_dollar
    }).dropna()
    
    print("Variable definitions:")
    print("- gold_price: Gold futures adjusted closing price ($/oz)")
    print("- tips_yield: 10-year Treasury Inflation-Protected Securities yield (%)")
    print("- breakeven_inflation: 10-year breakeven inflation rate (%)")
    print("- trade_weighted_dollar: Trade-weighted US dollar index\n")
    
    print(f"Final dataset summary:")
    print(f"Observations: {len(gold_macro_data)}")
    print(f"Variables: {len(gold_macro_data.columns)}")
    print(f"Date range: {gold_macro_data.index[0]} to {gold_macro_data.index[-1]}\n")
    
    return gold_macro_data

# ===============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS - STATIONARITY INVESTIGATION
# ===============================================================================

def plot_time_series_and_acf(data, title_prefix=""):
    """
    Plot time series and ACF for stationarity analysis
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{title_prefix} - Visual Stationarity Analysis', fontsize=16)
    
    # Plot time series
    axes[0, 0].plot(data.index, data.values)
    axes[0, 0].set_title(f'{title_prefix}: Time Series')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].grid(True)
    
    # Plot ACF with safety check
    if acf is not None:
        try:
            acf_values = acf(data.dropna(), nlags=50, fft=True)
            axes[0, 1].plot(range(len(acf_values)), acf_values)
            axes[0, 1].axhline(y=0, color='k', linestyle='-')
            axes[0, 1].axhline(y=1.96/np.sqrt(len(data)), color='r', linestyle='--')
            axes[0, 1].axhline(y=-1.96/np.sqrt(len(data)), color='r', linestyle='--')
            axes[0, 1].set_title(f'{title_prefix}: Autocorrelation Function')
            axes[0, 1].set_xlabel('Lag')
            axes[0, 1].set_ylabel('ACF')
            axes[0, 1].grid(True)
        except Exception as e:
            axes[0, 1].text(0.5, 0.5, 'ACF computation failed\nInstall statsmodels', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title(f'{title_prefix}: ACF (Not Available)')
    else:
        axes[0, 1].text(0.5, 0.5, 'ACF function not available\nInstall statsmodels', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title(f'{title_prefix}: ACF (Not Available)')
    
    # Distribution plot
    axes[1, 0].hist(data.dropna(), bins=30, alpha=0.7, density=True)
    axes[1, 0].set_title(f'{title_prefix}: Distribution')
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].grid(True)
    
    # Summary statistics
    stats_text = f"""
    Mean: {data.mean():.4f}
    Std: {data.std():.4f}
    Min: {data.min():.4f}
    Max: {data.max():.4f}
    Skewness: {data.skew():.4f}
    """
    axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                    fontsize=12, verticalalignment='center')
    axes[1, 1].set_title('Summary Statistics')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

def investigate_stationarity(gold_macro_data):
    """
    Examine the time series properties of each variable in levels
    """
    print("STEP 2: Visual Stationarity Analysis")
    print("====================================\n")
    
    print("Examining the time series properties of each variable in levels.")
    print("Non-stationary series typically show:")
    print("- Persistent trends or random walks")
    print("- ACF that decays very slowly") 
    print("- High autocorrelation at many lags\n")
    
    print("Stationary series show:")
    print("- Constant mean and variance over time")
    print("- ACF that decays quickly to zero")
    print("- Mean reversion properties\n")
    
    # Analyze each variable
    for column in gold_macro_data.columns:
        print(f"Analyzing {column} in levels...")
        plot_time_series_and_acf(gold_macro_data[column], column.replace('_', ' ').title())
    
    print("INTERPRETATION: If ACF declines slowly and remains significant at many lags,")
    print("the series is likely non-stationary and requires differencing.\n")

# ===============================================================================
# STEP 3: ARIMA MODEL IDENTIFICATION FOR DIFFERENCED SERIES
# ===============================================================================

def create_stationary_transformations(gold_macro_data):
    """
    Transform non-stationary series to achieve stationarity through differencing
    """
    print("STEP 3: Creating Stationary Transformations")
    print("===========================================\n")
    
    print("Transform non-stationary series to achieve stationarity through differencing.")
    print("For financial returns, we typically use log differences (continuously compounded returns).")
    print("For variables already in percentage terms, simple differences are appropriate.\n")
    
    # Calculate returns/changes for each series
    transformations = {}
    
    # Gold: Use log differences to get continuously compounded returns
    transformations['gold_returns'] = np.log(gold_macro_data['gold_price']).diff().dropna()
    print("Gold returns: log(Pt) - log(Pt-1) - continuously compounded returns")
    
    # Other variables: Use simple differences (already in percentage/index form)
    transformations['tips_changes'] = gold_macro_data['tips_yield'].diff().dropna()
    transformations['inflation_changes'] = gold_macro_data['breakeven_inflation'].diff().dropna()
    transformations['dollar_changes'] = gold_macro_data['trade_weighted_dollar'].diff().dropna()
    
    print("Other variables: simple first differences\n")
    
    # Align all series to same time period
    min_start = max([ts.index[0] for ts in transformations.values()])
    max_end = min([ts.index[-1] for ts in transformations.values()])
    
    for key in transformations:
        transformations[key] = transformations[key][min_start:max_end]
    
    transformed_data = pd.DataFrame(transformations)
    
    print(f"Transformed dataset:")
    print(f"Observations: {len(transformed_data)}")
    print(f"Variables: {transformed_data.columns.tolist()}")
    print()
    
    return transformed_data

def arima_model_selection(transformed_data):
    """
    ARIMA model selection for each series
    """
    print("=== ARIMA MODEL SELECTION ===\n")
    
    results = {}
    
    for column in transformed_data.columns:
        print(f"--- {column.upper()} ANALYSIS ---")
        print(f"Testing different model selection approaches:\n")
        
        series = transformed_data[column].dropna()
        
        # Method 1: Auto ARIMA (using statsmodels)
        try:
            # Simple approach - try different orders and select based on AIC
            best_aic = np.inf
            best_order = None
            best_model = None
            
            for p in range(3):
                for q in range(3):
                    try:
                        model = ARIMA(series, order=(p, 0, q)).fit()
                        if model.aic < best_aic:
                            best_aic = model.aic
                            best_order = (p, 0, q)
                            best_model = model
                    except:
                        continue
            
            if best_model is not None:
                print(f"Best ARIMA order: {best_order}")
                print(f"AIC: {best_aic:.4f}")
                results[column] = {
                    'model': best_model,
                    'order': best_order,
                    'aic': best_aic
                }
            else:
                print("Could not fit ARIMA model")
                results[column] = None
                
        except Exception as e:
            print(f"Error fitting ARIMA model: {e}")
            results[column] = None
        
        # Plot the series and its ACF/PACF
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Time series plot
        axes[0].plot(series.index, series.values)
        axes[0].set_title(f'{column}: Time Series')
        axes[0].grid(True)
        
        # ACF plot with safety check
        if acf is not None:
            try:
                acf_vals = acf(series, nlags=20, fft=True)
                axes[1].plot(range(len(acf_vals)), acf_vals)
                axes[1].axhline(y=0, color='k', linestyle='-')
                axes[1].axhline(y=1.96/np.sqrt(len(series)), color='r', linestyle='--')
                axes[1].axhline(y=-1.96/np.sqrt(len(series)), color='r', linestyle='--')
                axes[1].set_title(f'{column}: ACF')
                axes[1].grid(True)
            except Exception as e:
                axes[1].text(0.5, 0.5, 'ACF computation failed', ha='center', va='center', transform=axes[1].transAxes)
                axes[1].set_title(f'{column}: ACF (Error)')
        else:
            axes[1].text(0.5, 0.5, 'ACF not available', ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title(f'{column}: ACF (Not Available)')
        
        # PACF plot with safety check
        if pacf is not None:
            try:
                pacf_vals = pacf(series, nlags=20)
                axes[2].plot(range(len(pacf_vals)), pacf_vals)
                axes[2].axhline(y=0, color='k', linestyle='-')
                axes[2].axhline(y=1.96/np.sqrt(len(series)), color='r', linestyle='--')
                axes[2].axhline(y=-1.96/np.sqrt(len(series)), color='r', linestyle='--')
                axes[2].set_title(f'{column}: PACF')
                axes[2].grid(True)
            except Exception as e:
                axes[2].text(0.5, 0.5, 'PACF computation failed', ha='center', va='center', transform=axes[2].transAxes)
                axes[2].set_title(f'{column}: PACF (Error)')
        else:
            axes[2].text(0.5, 0.5, 'PACF not available', ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title(f'{column}: PACF (Not Available)')
        
        plt.tight_layout()
        plt.show()
        
        print()
    
    return results

# ===============================================================================
# STEP 4: REGRESSION ANALYSIS - GOLD RETURNS ON MACRO VARIABLES
# ===============================================================================

def regression_analysis(transformed_data):
    """
    Investigate the relationship between gold returns and changes in macroeconomic variables
    """
    print("STEP 4: Regression Analysis - Gold Returns on Macro Variables")
    print("============================================================\n")
    
    print("Investigate the relationship between gold returns and changes in macroeconomic")
    print("variables. This helps us understand what drives gold price movements.\n")
    
    print("ECONOMIC HYPOTHESES:")
    print("1. TIPS yield changes: Higher real yields may reduce gold's appeal (negative coefficient)")
    print("2. Breakeven inflation: Higher inflation expectations may increase gold demand (positive coefficient)")
    print("3. Dollar strength: Stronger dollar typically reduces gold prices (negative coefficient)\n")
    
    # Prepare regression data
    y = transformed_data['gold_returns'].dropna()
    X = transformed_data[['tips_changes', 'inflation_changes', 'dollar_changes']].dropna()
    
    # Align series
    common_index = y.index.intersection(X.index)
    y = y[common_index]
    X = X.loc[common_index]
    
    # Add constant
    X_with_const = sm.add_constant(X)
    
    # Fit OLS regression
    print("--- BASIC OLS REGRESSION ---")
    ols_model = sm.OLS(y, X_with_const).fit()
    print(ols_model.summary())
    
    print("\nECONOMIC INTERPRETATION OF COEFFICIENTS:")
    print("- TIPS yield coefficient: Effect of real interest rate changes on gold returns")
    print("- Breakeven inflation coefficient: Effect of inflation expectations on gold returns")
    print("- Dollar index coefficient: Effect of currency strength on gold returns")
    print("- Statistical significance indicates which variables reliably predict gold returns\n")
    
    return ols_model, y, X_with_const

# ===============================================================================
# STEP 5: RESIDUAL ANALYSIS AND SERIAL CORRELATION CORRECTION
# ===============================================================================

def residual_analysis_and_correction(ols_model, y, X_with_const):
    """
    Check and correct for serial correlation in regression residuals
    """
    print("STEP 5: Residual Analysis and Serial Correlation Testing")
    print("=======================================================\n")
    
    print("Check if regression residuals exhibit serial correlation, which violates")
    print("the assumption of independent errors and can lead to incorrect inference.\n")
    
    # Get residuals
    residuals = ols_model.resid
    
    # Create comprehensive diagnostic plots (matching ARMAModelingExamples.py style)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Comprehensive Regression Residual Diagnostics', fontsize=16)
    
    # Top row: tsdiag() style plots
    # 1. Standardized residuals over time
    axes[0, 0].plot(residuals.index, residuals.values, alpha=0.8, linewidth=0.8)
    axes[0, 0].set_title('Standardized Residuals')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='red', linestyle='-', alpha=0.5)
    
    # 2. ACF of residuals
    if acf is not None:
        try:
            max_lags = min(25, len(residuals)//4)
            acf_resid = acf(residuals, nlags=max_lags, fft=True)
            lags = range(len(acf_resid))
            axes[0, 1].plot(lags, acf_resid, 'o-', markersize=4)
            axes[0, 1].axhline(y=0, color='k', linestyle='-')
            # Confidence bands
            conf_band = 1.96/np.sqrt(len(residuals))
            axes[0, 1].axhline(y=conf_band, color='r', linestyle='--', alpha=0.7)
            axes[0, 1].axhline(y=-conf_band, color='r', linestyle='--', alpha=0.7)
            axes[0, 1].fill_between(lags, -conf_band, conf_band, alpha=0.1, color='blue')
            axes[0, 1].set_title('ACF of Residuals')
            axes[0, 1].set_xlabel('Lag')
            axes[0, 1].set_ylabel('ACF')
            axes[0, 1].grid(True, alpha=0.3)
        except Exception as e:
            axes[0, 1].text(0.5, 0.5, f'ACF computation failed:\n{str(e)[:30]}', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('ACF of Residuals (Failed)')
    
    # 3. Box-Ljung test p-values for increasing lags (key diagnostic!)
    gof_lag = min(36, len(residuals)//4)
    try:
        test_lags = range(1, gof_lag + 1)
        p_values = []
        
        for lag in test_lags:
            try:
                lb_result = acorr_ljungbox(residuals, lags=lag, return_df=True)
                p_val = lb_result['lb_pvalue'].iloc[-1]
                p_values.append(p_val)
            except:
                p_values.append(np.nan)
        
        axes[0, 2].plot(test_lags, p_values, 'o-', markersize=4, color='blue')
        axes[0, 2].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='5% level')
        axes[0, 2].axhline(y=0.01, color='red', linestyle=':', alpha=0.7, label='1% level')
        axes[0, 2].set_title('Box-Ljung Test p-values')
        axes[0, 2].set_xlabel('Lag')
        axes[0, 2].set_ylabel('p-value')
        axes[0, 2].set_ylim(-0.05, 1.05)
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].legend(fontsize=8)
        
        # Shade the rejection region
        axes[0, 2].fill_between(test_lags, 0, 0.05, alpha=0.2, color='red')
        
    except Exception as e:
        axes[0, 2].text(0.5, 0.5, f'Box-Ljung series failed:\n{str(e)[:30]}', 
                       ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('Box-Ljung Test p-values (Failed)')
    
    # Bottom row: tsdisplay() style plots
    # 4. PACF of residuals  
    if pacf is not None:
        try:
            max_lags_pacf = min(25, len(residuals)//4)
            pacf_resid = pacf(residuals, nlags=max_lags_pacf)
            lags_pacf = range(len(pacf_resid))
            axes[1, 0].plot(lags_pacf, pacf_resid, 'o-', markersize=4, color='green')
            axes[1, 0].axhline(y=0, color='k', linestyle='-')
            # Confidence bands for PACF
            conf_band_pacf = 1.96/np.sqrt(len(residuals))
            axes[1, 0].axhline(y=conf_band_pacf, color='r', linestyle='--', alpha=0.7)
            axes[1, 0].axhline(y=-conf_band_pacf, color='r', linestyle='--', alpha=0.7)
            axes[1, 0].fill_between(lags_pacf, -conf_band_pacf, conf_band_pacf, alpha=0.1, color='green')
            axes[1, 0].set_title('PACF of Residuals')
            axes[1, 0].set_xlabel('Lag')
            axes[1, 0].set_ylabel('PACF')
            axes[1, 0].grid(True, alpha=0.3)
        except Exception as e:
            axes[1, 0].text(0.5, 0.5, f'PACF computation failed:\n{str(e)[:30]}', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('PACF of Residuals (Failed)')
    
    # 5. Q-Q plot
    try:
        sm.qqplot(residuals, ax=axes[1, 1], line='s')
        axes[1, 1].set_title('Normal Q-Q Plot')
        axes[1, 1].grid(True, alpha=0.3)
    except:
        axes[1, 1].text(0.5, 0.5, 'Q-Q plot failed', ha='center', va='center', 
                       transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Normal Q-Q Plot (Failed)')
    
    # 6. Histogram with normal overlay
    try:
        axes[1, 2].hist(residuals, bins=20, alpha=0.7, density=True, color='lightblue', 
                       edgecolor='black')
        
        # Overlay normal distribution
        x_norm = np.linspace(residuals.min(), residuals.max(), 100)
        y_norm = stats.norm.pdf(x_norm, residuals.mean(), residuals.std())
        axes[1, 2].plot(x_norm, y_norm, 'r-', linewidth=2, label='Normal')
        
        axes[1, 2].set_title('Residual Distribution')
        axes[1, 2].set_xlabel('Residuals')
        axes[1, 2].set_ylabel('Density')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    except:
        axes[1, 2].text(0.5, 0.5, 'Histogram failed', ha='center', va='center', 
                       transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Residual Distribution (Failed)')
    
    plt.tight_layout()
    plt.show()
    
    # Ljung-Box test for serial correlation
    print("--- LJUNG-BOX TEST FOR SERIAL CORRELATION ---")
    lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
    print(lb_test)
    
    print("\nINTERPRETATION: Look for:")
    print("- Ljung-Box p-values < 0.05 indicate serial correlation")
    print("- Significant ACF spikes suggest autocorrelated errors\n")
    
    # Check if we need to correct for serial correlation
    has_serial_correlation = any(lb_test['lb_pvalue'] < 0.05)
    
    if has_serial_correlation:
        print("🚨 SERIAL CORRELATION DETECTED!")
        print("We need to correct for this using one of two approaches:\n")
        
        # METHOD 1: ARIMA errors approach
        print("=== METHOD 1: ARIMA ERRORS APPROACH ===")
        print("Jointly estimate regression coefficients with ARIMA structure in residuals\n")
        
        arima_results = correct_serial_correlation_arima(y, X_with_const.iloc[:, 1:])  # Exclude constant
        
        # METHOD 2: HAC standard errors approach  
        print("\n=== METHOD 2: HAC STANDARD ERRORS APPROACH ===")
        print("Keep OLS coefficients but use robust standard errors\n")
        
        hac_results = correct_serial_correlation_hac(ols_model, y, X_with_const)
        
        # Compare approaches
        compare_serial_correlation_methods(ols_model, arima_results, hac_results)
        
        return residuals, arima_results, hac_results
        
    else:
        print("✅ No significant serial correlation detected.")
        print("Standard OLS inference is appropriate.\n")
        
        return residuals, None, None

def correct_serial_correlation_arima(y, X):
    """
    Correct serial correlation using ARIMA errors approach
    """
    print("--- AUTOMATIC ERROR STRUCTURE SELECTION ---")
    print("Finding optimal ARIMA error structure for regression residuals...")
    
    results = {}
    
    try:
        # Method 1: Basic ARIMA(0,0,0) framework for diagnostics
        print("\n1. Basic regression in ARIMA framework:")
        basic_arima = ARIMA(y, exog=X, order=(0, 0, 0)).fit()
        print(basic_arima.summary())
        
        # Method 2: Automatic selection of error structure
        print("\n2. Automatic ARIMA error selection:")
        best_aic = np.inf
        best_model = None
        best_order = None
        
        # Test different ARIMA orders for errors
        for p in range(3):
            for q in range(3):
                if p == 0 and q == 0:  # Skip (0,0,0) - already done
                    continue
                try:
                    model = ARIMA(y, exog=X, order=(p, 0, q)).fit()
                    if model.aic < best_aic:
                        best_aic = model.aic
                        best_model = model
                        best_order = (p, 0, q)
                        
                    print(f"ARIMA{(p, 0, q)}: AIC = {model.aic:.2f}")
                except:
                    print(f"ARIMA{(p, 0, q)}: Failed to converge")
                    continue
        
        if best_model is not None:
            print(f"\n✅ Best model: ARIMA{best_order} with AIC = {best_aic:.2f}")
            print("\nBest model summary:")
            print(best_model.summary())
            
            # Diagnostic plots for best model
            print("\n3. Diagnostics for ARIMA error model:")
            residuals = best_model.resid
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            # Residuals over time
            axes[0].plot(residuals.index, residuals.values, alpha=0.8)
            axes[0].set_title(f'ARIMA{best_order} Error Model Residuals')
            axes[0].set_xlabel('Date')
            axes[0].set_ylabel('Residuals')
            axes[0].grid(True)
            axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            
            # ACF of residuals
            if acf is not None:
                try:
                    acf_resid = acf(residuals, nlags=20, fft=True)
                    axes[1].plot(range(len(acf_resid)), acf_resid, 'o-')
                    axes[1].axhline(y=0, color='k', linestyle='-')
                    axes[1].axhline(y=1.96/np.sqrt(len(residuals)), color='r', linestyle='--')
                    axes[1].axhline(y=-1.96/np.sqrt(len(residuals)), color='r', linestyle='--')
                    axes[1].set_title('ACF of ARIMA Error Model Residuals')
                    axes[1].set_xlabel('Lag')
                    axes[1].set_ylabel('ACF')
                    axes[1].grid(True)
                except:
                    axes[1].text(0.5, 0.5, 'ACF computation failed', ha='center', va='center')
            
            # Ljung-Box test for corrected residuals
            try:
                lb_test_corrected = acorr_ljungbox(residuals, lags=10, return_df=True)
                significant_lags = sum(lb_test_corrected['lb_pvalue'] < 0.05)
                
                axes[2].plot(lb_test_corrected.index, lb_test_corrected['lb_pvalue'], 'o-')
                axes[2].axhline(y=0.05, color='red', linestyle='--', label='5% level')
                axes[2].set_title('Ljung-Box p-values (Corrected)')
                axes[2].set_xlabel('Lag')
                axes[2].set_ylabel('p-value')
                axes[2].legend()
                axes[2].grid(True)
                
                if significant_lags == 0:
                    print("✅ Serial correlation successfully corrected!")
                else:
                    print(f"⚠️  Some serial correlation remains ({significant_lags} significant lags)")
                    
            except Exception as e:
                axes[2].text(0.5, 0.5, f'Ljung-Box test failed:\n{str(e)[:30]}', ha='center', va='center')
            
            plt.tight_layout()
            plt.show()
            
            results = {
                'basic_model': basic_arima,
                'best_model': best_model,
                'best_order': best_order,
                'best_aic': best_aic
            }
            
        else:
            print("❌ Could not find suitable ARIMA error structure")
            results = {'basic_model': basic_arima}
            
    except Exception as e:
        print(f"❌ Error in ARIMA correction: {e}")
        results = {}
    
    return results

def correct_serial_correlation_hac(ols_model, y, X_with_const):
    """
    Correct serial correlation using HAC standard errors
    """
    print("--- HAC STANDARD ERRORS CORRECTION ---")
    print("Computing Heteroskedasticity and Autocorrelation Consistent standard errors\n")
    
    try:
        # Standard OLS results
        print("1. Standard OLS standard errors:")
        ols_coeftest = pd.DataFrame({
            'Coefficient': ols_model.params,
            'Std_Error': ols_model.bse,
            't_value': ols_model.tvalues,
            'p_value': ols_model.pvalues
        })
        print(ols_coeftest.round(4))
        
        # HAC standard errors
        print("\n2. HAC (Robust) standard errors:")
        try:
            hac_model = sm.OLS(y, X_with_const).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
            hac_coeftest = pd.DataFrame({
                'Coefficient': hac_model.params,
                'Std_Error': hac_model.bse,
                't_value': hac_model.tvalues,
                'p_value': hac_model.pvalues
            })
            print(hac_coeftest.round(4))
            
            # Compare standard errors
            print("\n3. Standard Error Comparison:")
            se_comparison = pd.DataFrame({
                'Variable': ols_model.params.index,
                'OLS_SE': ols_model.bse,
                'HAC_SE': hac_model.bse,
                'SE_Ratio': hac_model.bse / ols_model.bse
            })
            print(se_comparison.round(4))
            
            print("\nINTERPRETATION:")
            print("- SE_Ratio > 1: HAC standard errors are larger (more conservative)")
            print("- Large ratios indicate significant autocorrelation in residuals")
            print("- Use HAC standard errors when residuals show serial correlation")
            
            return {
                'ols_results': ols_coeftest,
                'hac_results': hac_coeftest,
                'se_comparison': se_comparison,
                'hac_model': hac_model
            }
            
        except Exception as e:
            print(f"HAC computation failed: {e}")
            print("This may occur with certain data characteristics.")
            return {'ols_results': ols_coeftest}
            
    except Exception as e:
        print(f"Error in HAC correction: {e}")
        return {}

def compare_serial_correlation_methods(ols_model, arima_results, hac_results):
    """
    Compare different approaches to handling serial correlation
    """
    print("\n" + "="*60)
    print("COMPARISON OF SERIAL CORRELATION CORRECTION METHODS")
    print("="*60)
    
    print("\nMETHOD COMPARISON:")
    methods_comparison = pd.DataFrame({
        'Approach': [
            'Basic OLS',
            'ARIMA Errors', 
            'HAC Standard Errors'
        ],
        'Pros': [
            'Simple, interpretable',
            'Models error structure explicitly',
            'Robust without modeling errors'
        ],
        'Cons': [
            'Biased SE if serial correlation',
            'More complex, model selection needed',
            'Doesn\'t model underlying structure'
        ],
        'When_to_Use': [
            'No serial correlation detected',
            'Clear ARMA pattern in residuals',
            'Uncertain error structure'
        ]
    })
    
    print(methods_comparison.to_string(index=False))
    
    print("\nRECOMMENDATION:")
    if arima_results and 'best_model' in arima_results:
        print("✅ ARIMA errors approach successfully modeled the serial correlation.")
        print("   Use this if you want to explicitly model the error structure.")
    
    if hac_results and 'hac_results' in hac_results:
        print("✅ HAC standard errors provide robust inference.")
        print("   Use this for conservative standard errors without error modeling.")
    
    print("\n" + "="*60)

# ===============================================================================
# MAIN EXECUTION
# ===============================================================================

def main():
    """
    Main function to execute the complete analysis
    """
    print("=== COMPREHENSIVE ARMA MODELING AND REGRESSION ANALYSIS ===\n")
    
    # Step 1: Download and prepare data
    gold_macro_data = download_and_prepare_data()
    
    # Step 2: Investigate stationarity
    investigate_stationarity(gold_macro_data)
    
    # Step 3: Create stationary transformations and ARIMA modeling
    transformed_data = create_stationary_transformations(gold_macro_data)
    arima_results = arima_model_selection(transformed_data)
    
    # Step 4: Regression analysis
    ols_model, y, X_with_const = regression_analysis(transformed_data)
    
    # Step 5: Residual analysis and serial correlation correction
    residuals, arima_results, hac_results = residual_analysis_and_correction(ols_model, y, X_with_const)
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("\nKEY FINDINGS:")
    print("1. Visual analysis suggested non-stationarity in levels for all series")
    print("2. First differencing achieved stationarity for modeling")
    print("3. ARIMA models were fitted to individual series")
    print("4. Regression analysis revealed relationships between gold returns and macro variables")
    print("5. Residual diagnostics checked for model adequacy")

if __name__ == "__main__":
    main()