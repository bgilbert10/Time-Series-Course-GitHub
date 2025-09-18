# ===============================================================================
# IN-CLASS ACTIVITY: GOLD PRICES AND MACROECONOMIC VARIABLES
# ===============================================================================
# Date: September 25 Class Session
#
# ACTIVITY OBJECTIVES:
# 1. Practice downloading and merging multiple financial time series
# 2. Conduct preliminary stationarity analysis using visual inspection
# 3. Apply ARIMA modeling to individual economic time series
# 4. Estimate regression models with time series data
# 5. Diagnose and correct serial correlation in regression residuals
#
# STUDENT LEARNING OUTCOMES:
# By completing this activity, students will be able to:
# - Retrieve data from multiple sources (Yahoo Finance, FRED)
# - Merge time series with different frequencies and missing observations
# - Distinguish between stationary and non-stationary series visually
# - Select appropriate transformations for economic variables
# - Interpret ARIMA model selection criteria
# - Test for and correct serial correlation in regression models
#
# ECONOMIC CONTEXT:
# Gold is traditionally viewed as a hedge against inflation and currency debasement.
# This activity explores the empirical relationships between gold returns and:
# - Real interest rates (TIPS yields)
# - Inflation expectations (breakeven inflation)
# - Currency strength (trade-weighted dollar index)
#
# TIME PERIOD UPDATED: January 3, 2008 to June 30, 2020
# (Note: Original activity used 2006-2020, updated for data availability)
# ===============================================================================

# Import required packages with compatibility handling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

# Set plotting style with fallback
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        try:
            sns.set_style("whitegrid")
        except:
            pass
np.random.seed(42)


print("=" * 70)
print("IN-CLASS ACTIVITY: Gold Prices and Macroeconomic Variables")
print("=" * 70)
print()

# ===============================================================================
# STEP 1: DATA COLLECTION AND PREPARATION
# ===============================================================================

def step1_data_collection():
    """
    INSTRUCTION FOR STUDENTS:
    Follow along as we download data from different sources. Note the different
    data providers and how we handle missing values and date alignment.
    """
    print("STEP 1: Data Collection and Preparation")
    print("=" * 50)
    print()
    
    print("INSTRUCTION FOR STUDENTS:")
    print("Follow along as we download data from different sources. Note the different")
    print("data providers and how we handle missing values and date alignment.")
    print()
    
    # Define study period
    study_start_date = "2008-01-03"
    study_end_date = "2020-06-30"
    
    print("STUDENT TASK: Download financial time series data")
    print(f"Analysis period: {study_start_date} to {study_end_date}")
    print("This period includes the 2008 Financial Crisis and subsequent recovery.")
    print()
    
    # Download gold futures data from Yahoo Finance
    print("Downloading data from Yahoo Finance and FRED...")
    print()
    
    # Yahoo Finance: Gold futures contract
    if YF_AVAILABLE and yf is not None:
        try:
            gold_data = yf.download("GC=F", start=study_start_date, end=study_end_date)
            gold_price = gold_data['Close'].squeeze().dropna()
            print("✓ Gold futures data downloaded successfully!")
        except Exception as e:
            print(f"✗ Error downloading gold data: {e}")
            print("Unable to proceed without gold price data. Please check your internet connection.")
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
            tips_yield = web.DataReader('DFII10', 'fred', study_start_date, study_end_date)['DFII10'].dropna()
            print("✓ 10-year TIPS yield data downloaded successfully!")
            
            breakeven_inflation = web.DataReader('T10YIE', 'fred', study_start_date, study_end_date)['T10YIE'].dropna()
            print("✓ 10-year breakeven inflation data downloaded successfully!")
            
            trade_weighted_dollar = web.DataReader('DTWEXBGS', 'fred', study_start_date, study_end_date)['DTWEXBGS'].dropna()
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
    
    # STUDENT EXERCISE: Examine each series individually before merging
    print("STUDENT EXERCISE: Examine each series individually")
    print("Uncomment the following lines in your code to explore:")
    print("# print(gold_price.head())")
    print("# print(tips_yield.head())")
    print("# print(breakeven_inflation.head())")
    print("# print(trade_weighted_dollar.head())")
    print()
    
    print("Merging datasets with different frequencies and observation dates...")
    
    # Merge all time series into a single DataFrame
    economics_data = pd.DataFrame({
        'gold_price': gold_price,
        'tips_yield': tips_yield,
        'breakeven_inflation': breakeven_inflation,
        'trade_weighted_dollar': trade_weighted_dollar
    }).dropna()
    
    print("Variable definitions:")
    print("- gold_price: Gold futures adjusted closing price ($/oz)")
    print("- tips_yield: 10-year Treasury Inflation-Protected Securities yield (%)")
    print("- breakeven_inflation: 10-year breakeven inflation rate (%)")
    print("- trade_weighted_dollar: Trade-weighted US dollar index")
    print()
    
    print(f"Final dataset summary:")
    print(f"Observations: {len(economics_data)}")
    print(f"Variables: {len(economics_data.columns)}")
    print(f"Date range: {economics_data.index[0].strftime('%Y-%m-%d')} to {economics_data.index[-1].strftime('%Y-%m-%d')}")
    print()
    
    return economics_data

# ===============================================================================
# STEP 2: STATIONARITY INVESTIGATION - VISUAL ANALYSIS
# ===============================================================================

def step2_visual_stationarity_analysis(economics_data):
    """
    INSTRUCTION FOR STUDENTS:
    Before formal statistical tests, we examine each series visually.
    Look for trends, structural breaks, and persistent patterns that indicate
    non-stationarity.
    """
    print("STEP 2: Visual Stationarity Analysis")
    print("=" * 40)
    print()
    
    print("INSTRUCTION FOR STUDENTS:")
    print("Before formal statistical tests, we examine each series visually.")
    print("Look for trends, structural breaks, and persistent patterns that indicate")
    print("non-stationarity.")
    print()
    
    print("QUESTION FOR STUDENTS: What visual patterns suggest non-stationarity?")
    print("ANSWER: Look for trends, permanent shifts, slow-decaying ACF")
    print()
    
    # Create subplots for each variable
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Visual Stationarity Analysis of Economic Variables', fontsize=16)
    
    variables = list(economics_data.columns)
    
    for i, var in enumerate(variables):
        # Time series plot
        axes[0, i].plot(economics_data.index, economics_data[var], alpha=0.8)
        axes[0, i].set_title(f'{var.replace("_", " ").title()}: Level Series')
        axes[0, i].set_ylabel('Value')
        axes[0, i].grid(True)
        axes[0, i].tick_params(axis='x', rotation=45)
        
        # ACF plot with safety check
        if acf is not None:
            try:
                acf_values = acf(economics_data[var].dropna(), nlags=50, fft=True)
                axes[1, i].plot(range(len(acf_values)), acf_values, 'o-', markersize=3)
                axes[1, i].axhline(y=0, color='k', linestyle='-')
                axes[1, i].axhline(y=1.96/np.sqrt(len(economics_data[var])), color='r', linestyle='--')
                axes[1, i].axhline(y=-1.96/np.sqrt(len(economics_data[var])), color='r', linestyle='--')
                axes[1, i].set_title(f'ACF: {var.replace("_", " ").title()}')
                axes[1, i].set_xlabel('Lag')
                axes[1, i].set_ylabel('ACF')
                axes[1, i].grid(True)
            except Exception as e:
                axes[1, i].text(0.5, 0.5, 'ACF computation failed', ha='center', va='center', transform=axes[1, i].transAxes)
                axes[1, i].set_title(f'ACF: {var.replace("_", " ").title()} (Error)')
        else:
            axes[1, i].text(0.5, 0.5, 'ACF not available', ha='center', va='center', transform=axes[1, i].transAxes)
            axes[1, i].set_title(f'ACF: {var.replace("_", " ").title()} (Not Available)')
    
    plt.tight_layout()
    plt.show()
    
    print("STUDENT TASKS FOR EACH PLOT:")
    print("1. Gold Prices: Describe the trend and volatility patterns you observe")
    print("2. TIPS Yields: Do interest rates appear mean-reverting or trending?") 
    print("3. Breakeven Inflation: How do inflation expectations behave over time?")
    print("4. Trade-Weighted Dollar: Compare this to other exchange rate series")
    print()
    
    print("INSTRUCTOR NOTES:")
    print("- Slowly decaying ACF suggests unit root/non-stationarity")
    print("- Interest rates show some mean reversion but can be persistent")
    print("- Inflation expectations can have unit roots or be near-unit root")
    print("- Exchange rates often exhibit random walk behavior")
    print()
    
    print("PRELIMINARY CONCLUSION FROM VISUAL ANALYSIS:")
    print("- All series appear non-stationary in levels")
    print("- Slowly decaying ACF patterns support this conclusion") 
    print("- First differencing will likely be needed for modeling")
    print("- Next step: Transform to stationary series")
    print()

# ===============================================================================
# STEP 3: CREATING STATIONARY TRANSFORMATIONS  
# ===============================================================================

def step3_stationary_transformations(economics_data):
    """
    INSTRUCTION FOR STUDENTS:
    Now we create stationary versions of each series using appropriate transformations.
    The choice of transformation depends on the economic interpretation we want.
    """
    print("STEP 3: Creating Stationary Transformations")
    print("=" * 50)
    print()
    
    print("INSTRUCTION FOR STUDENTS:")
    print("Now we create stationary versions of each series using appropriate transformations.")
    print("The choice of transformation depends on the economic interpretation we want.")
    print()
    
    print("TRANSFORMATION STRATEGY:")
    print("- Gold prices → Log differences (continuously compounded returns)")
    print("- Interest rates → Simple differences (changes in percentage points)")
    print("- Exchange rates → Simple differences (changes in index points)")
    print()
    
    print("STUDENT QUESTION: Why log differences for gold but not for interest rates?")
    print("ANSWER: Gold prices are in levels (multiplicative), rates are already in percentages")
    print()
    
    # Transform each series appropriately
    transformations = {}
    
    # Gold: Log differences for returns
    transformations['gold_returns'] = np.log(economics_data['gold_price']).diff().dropna()
    print("✓ Gold returns: Continuously compounded gold returns")
    
    # Other variables: Simple differences  
    transformations['tips_changes'] = economics_data['tips_yield'].diff().dropna()
    print("✓ TIPS changes: Changes in TIPS yields (percentage points)")
    
    transformations['inflation_changes'] = economics_data['breakeven_inflation'].diff().dropna()
    print("✓ Inflation changes: Changes in breakeven inflation (percentage points)")
    
    transformations['dollar_changes'] = economics_data['trade_weighted_dollar'].diff().dropna()
    print("✓ Dollar changes: Changes in trade-weighted dollar index")
    print()
    
    # Combine into DataFrame and align dates
    min_start = max([ts.index[0] for ts in transformations.values()])    
    max_end = min([ts.index[-1] for ts in transformations.values()])
    
    for key in transformations:
        transformations[key] = transformations[key][min_start:max_end]
    
    transformed_data = pd.DataFrame(transformations)
    
    # STUDENT EXERCISE: Plot the transformed series
    print("STUDENT EXERCISE: Plot the transformed series and comment on stationarity")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Transformed Series - Should Appear Stationary', fontsize=16)
    
    axes[0, 0].plot(transformed_data.index, transformed_data['gold_returns'], alpha=0.8, color='gold')
    axes[0, 0].set_title('Gold Returns')
    axes[0, 0].set_ylabel('Log Difference')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(transformed_data.index, transformed_data['tips_changes'], alpha=0.8, color='blue')
    axes[0, 1].set_title('TIPS Yield Changes')  
    axes[0, 1].set_ylabel('Percentage Points')
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(transformed_data.index, transformed_data['inflation_changes'], alpha=0.8, color='red')
    axes[1, 0].set_title('Inflation Expectation Changes')
    axes[1, 0].set_ylabel('Percentage Points')
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(transformed_data.index, transformed_data['dollar_changes'], alpha=0.8, color='green') 
    axes[1, 1].set_title('Dollar Index Changes')
    axes[1, 1].set_ylabel('Index Points')
    axes[1, 1].grid(True)
    
    for ax in axes.flat:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Transformed dataset: {len(transformed_data)} observations")
    print()
    
    return transformed_data

# ===============================================================================
# STEP 4: ARIMA MODEL SELECTION (Simplified for Class)
# ===============================================================================

def step4_simplified_arima_modeling(transformed_data):
    """
    Simplified ARIMA modeling suitable for in-class demonstration
    """
    print("STEP 4: ARIMA Model Selection (Class Exercise)")
    print("=" * 55)
    print()
    
    print("CLASS EXERCISE: ARIMA Model Selection")
    print("For each transformed series, we'll:")
    print("1. Examine ACF and PACF patterns")
    print("2. Try automatic model selection")
    print("3. Interpret results")
    print()
    
    results = {}
    
    for column in transformed_data.columns:
        print(f"--- {column.upper().replace('_', ' ')} ANALYSIS ---")
        
        series = transformed_data[column].dropna()
        
        # Plot ACF and PACF for visual identification
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Time series
        axes[0].plot(series.index, series.values, alpha=0.8)
        axes[0].set_title(f'{column}: Time Series')
        axes[0].grid(True)
        axes[0].tick_params(axis='x', rotation=45)
        
        # ACF with safety check
        if acf is not None:
            try:
                acf_vals = acf(series, nlags=20, fft=True)
                axes[1].plot(range(len(acf_vals)), acf_vals, 'o-')
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
        
        # PACF with safety check
        if pacf is not None:
            try:
                pacf_vals = pacf(series, nlags=20)
                axes[2].plot(range(len(pacf_vals)), pacf_vals, 'o-')
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
        
        # Simple automatic model selection
        print("STUDENT TASK: What do the ACF and PACF patterns suggest?")
        print("- AR(p): ACF decays exponentially, PACF cuts off after lag p")  
        print("- MA(q): ACF cuts off after lag q, PACF decays exponentially")
        print("- ARMA(p,q): Both ACF and PACF decay exponentially")
        print()
        
        # Try to fit a simple ARIMA model
        try:
            best_aic = np.inf
            best_order = None
            
            for p in range(3):
                for q in range(3):
                    try:
                        model = ARIMA(series, order=(p, 0, q)).fit()
                        if model.aic < best_aic:
                            best_aic = model.aic
                            best_order = (p, 0, q)
                    except:
                        continue
            
            if best_order is not None:
                print(f"Suggested ARIMA order: {best_order}")
                print(f"AIC: {best_aic:.4f}")
                results[column] = best_order
            else:
                print("White noise model suggested")
                results[column] = (0, 0, 0)
                
        except Exception as e:
            print(f"Model fitting issue: {e}")
            results[column] = None
        
        print()
    
    print("CLASS DISCUSSION:")
    print("- Gold returns often appear as white noise (efficient markets)")
    print("- Interest rate changes may show some autocorrelation") 
    print("- Exchange rate changes typically show minimal autocorrelation")
    print()
    
    return results

# ===============================================================================
# STEP 5: REGRESSION AND RESIDUAL ANALYSIS
# ===============================================================================

def step5_regression_analysis(transformed_data):
    """
    Regression analysis and residual diagnostics for class
    """
    print("STEP 5: Regression Analysis - Gold Returns and Macro Variables")
    print("=" * 70)
    print()
    
    print("CLASS QUESTION: What economic relationships do we expect?")
    print("1. TIPS yields ↑ → Gold returns ↓ (higher real rates reduce gold appeal)")
    print("2. Inflation expectations ↑ → Gold returns ↑ (inflation hedge)")
    print("3. Dollar strength ↑ → Gold returns ↓ (currency effects)")
    print()
    
    # Prepare regression data
    y = transformed_data['gold_returns'].dropna()
    X = transformed_data[['tips_changes', 'inflation_changes', 'dollar_changes']].dropna()
    
    # Align data
    common_index = y.index.intersection(X.index)
    y = y[common_index]
    X = X.loc[common_index]
    
    # Add constant term
    X_with_const = sm.add_constant(X)
    
    # Fit OLS regression
    print("--- BASIC OLS REGRESSION RESULTS ---")
    ols_model = sm.OLS(y, X_with_const).fit()
    print(ols_model.summary())
    print()
    
    print("STUDENT EXERCISE: Interpret the coefficients")
    print("- Are the signs consistent with economic theory?")
    print("- Which variables are statistically significant?")
    print("- What is the economic significance vs statistical significance?")
    print()
    
    # Comprehensive residual diagnostics
    print("--- COMPREHENSIVE RESIDUAL DIAGNOSTICS ---")
    residuals = ols_model.resid
    
    # Create comprehensive diagnostic plots (matching other Python programs)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Comprehensive Regression Residual Diagnostics', fontsize=16)
    
    # Top row: tsdiag() style plots
    # 1. Standardized residuals over time
    axes[0, 0].plot(residuals.index, residuals.values, alpha=0.8, linewidth=0.8)
    axes[0, 0].set_title('Standardized Residuals')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='red', linestyle='-', alpha=0.5)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
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
    
    # Statistical tests for serial correlation
    print("STUDENT QUESTION: Do the residuals show serial correlation?")
    lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
    print("Ljung-Box Test Results:")
    print(lb_test[['lb_stat', 'lb_pvalue']].round(4))
    
    # Check if we need to correct for serial correlation
    has_serial_correlation = any(lb_test['lb_pvalue'] < 0.05)
    
    if has_serial_correlation:
        print(f"\n🚨 WARNING: Serial correlation detected!")
        print("This violates OLS assumptions and may lead to incorrect inference")
        print("\nCLASS DISCUSSION: What should we do?")
        print("Two main approaches to correct this problem:")
        print("1. ARIMA errors: Model the error structure explicitly")
        print("2. HAC standard errors: Use robust standard errors")
        print()
        
        # Show both correction methods
        arima_results, hac_results = demonstrate_serial_correlation_corrections(
            y, X_with_const, ols_model
        )
        
        return ols_model, residuals, arima_results, hac_results
    else:
        print("\n✅ GOOD: No significant serial correlation detected")
        print("Standard OLS inference is appropriate")
        print()
        
        return ols_model, residuals, None, None

def demonstrate_serial_correlation_corrections(y, X_with_const, ols_model):
    """
    Demonstrate both ARIMA errors and HAC standard errors approaches
    (Educational version for in-class use)
    """
    print("=" * 70)
    print("IN-CLASS DEMONSTRATION: SERIAL CORRELATION CORRECTION METHODS")
    print("=" * 70)
    
    # Method 1: ARIMA Errors Approach
    print("\n🔧 METHOD 1: ARIMA ERRORS APPROACH")
    print("Idea: Model the error structure explicitly using ARIMA")
    print()
    
    arima_results = None
    try:
        print("Step 1: Try basic ARIMA framework")
        basic_arima = ARIMA(y, exog=X_with_const.iloc[:, 1:], order=(0, 0, 0)).fit()
        print(f"Basic model AIC: {basic_arima.aic:.2f}")
        
        print("\nStep 2: Test different error structures...")
        best_aic = basic_arima.aic
        best_model = basic_arima
        best_order = (0, 0, 0)
        
        # Test a few ARIMA orders
        test_orders = [(1, 0, 0), (0, 0, 1), (1, 0, 1), (0, 0, 2), (2, 0, 0)]
        
        for order in test_orders:
            try:
                model = ARIMA(y, exog=X_with_const.iloc[:, 1:], order=order).fit()
                print(f"ARIMA{order}: AIC = {model.aic:.2f}")
                
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_model = model
                    best_order = order
            except:
                print(f"ARIMA{order}: Failed to converge")
        
        print(f"\n✅ Best ARIMA error model: ARIMA{best_order} (AIC = {best_aic:.2f})")
        
        # Quick diagnostic
        corrected_residuals = best_model.resid
        lb_test_corrected = acorr_ljungbox(corrected_residuals, lags=5, return_df=True)
        remaining_autocorr = sum(lb_test_corrected['lb_pvalue'] < 0.05)
        
        if remaining_autocorr == 0:
            print("✅ Serial correlation successfully corrected!")
        else:
            print(f"⚠️  Some serial correlation remains ({remaining_autocorr} significant lags)")
        
        arima_results = {
            'best_model': best_model,
            'best_order': best_order,
            'best_aic': best_aic
        }
        
    except Exception as e:
        print(f"❌ ARIMA approach failed: {e}")
    
    # Method 2: HAC Standard Errors
    print(f"\n🔧 METHOD 2: HAC STANDARD ERRORS APPROACH")
    print("Idea: Keep OLS coefficients but use robust standard errors")
    print()
    
    hac_results = None
    try:
        # Standard OLS
        print("Standard OLS results:")
        ols_summary = pd.DataFrame({
            'Coefficient': ols_model.params.round(4),
            'Std_Error': ols_model.bse.round(4),
            't_value': ols_model.tvalues.round(3),
            'p_value': ols_model.pvalues.round(4)
        })
        print(ols_summary)
        
        # HAC standard errors
        print("\nHAC robust standard errors:")
        hac_model = sm.OLS(y, X_with_const).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
        hac_summary = pd.DataFrame({
            'Coefficient': hac_model.params.round(4),
            'Std_Error': hac_model.bse.round(4),
            't_value': hac_model.tvalues.round(3),
            'p_value': hac_model.pvalues.round(4)
        })
        print(hac_summary)
        
        # Compare standard errors
        print("\nStandard Error Comparison:")
        se_comparison = pd.DataFrame({
            'Variable': ols_model.params.index,
            'OLS_SE': ols_model.bse.round(4),
            'HAC_SE': hac_model.bse.round(4),
            'SE_Ratio': (hac_model.bse / ols_model.bse).round(3)
        })
        print(se_comparison)
        
        print("\nCLASS INTERPRETATION:")
        print("- SE_Ratio > 1: HAC standard errors are larger (more conservative)")
        print("- Large ratios indicate significant autocorrelation effects")
        
        hac_results = {
            'hac_model': hac_model,
            'se_comparison': se_comparison
        }
        
    except Exception as e:
        print(f"❌ HAC approach failed: {e}")
    
    # Final comparison
    print(f"\n📊 WHICH METHOD TO CHOOSE?")
    print("-" * 40)
    print("ARIMA Errors:")
    print("  ✅ Models error structure explicitly")
    print("  ❌ More complex, requires model selection")
    print("  🎯 Use when: Clear autocorrelation pattern")
    
    print("\nHAC Standard Errors:")
    print("  ✅ Simple, doesn't require error modeling")
    print("  ✅ Robust to various types of serial correlation")
    print("  🎯 Use when: Uncertain about error structure")
    
    print(f"\n{'=' * 70}")
    
    return arima_results, hac_results

# ===============================================================================
# MAIN CLASS EXECUTION
# ===============================================================================

def main_class_activity():
    """
    Execute the complete in-class activity
    """
    print("Welcome to the In-Class Gold and Macroeconomic Variables Activity!")
    print("Follow along as we work through each step together.")
    print()
    
    # Step 1: Data collection
    economics_data = step1_data_collection()
    
    # Step 2: Visual stationarity analysis
    step2_visual_stationarity_analysis(economics_data)
    
    # Step 3: Create stationary transformations  
    transformed_data = step3_stationary_transformations(economics_data)
    
    # Step 4: ARIMA model selection (simplified)
    arima_results = step4_simplified_arima_modeling(transformed_data)
    
    # Step 5: Regression analysis
    ols_model, residuals, arima_results, hac_results = step5_regression_analysis(transformed_data)
    
    print("=" * 70)
    print("CLASS ACTIVITY COMPLETE!")
    print("=" * 70)
    print()
    print("KEY LEARNING POINTS:")
    print("1. Data preparation is crucial for time series analysis")
    print("2. Visual inspection helps identify stationarity issues") 
    print("3. Appropriate transformations depend on economic interpretation")
    print("4. ARIMA modeling requires careful specification")
    print("5. Regression residuals must be checked for serial correlation")
    print()
    print("HOMEWORK ASSIGNMENT:")
    print("1. Download real FRED data using your API key")
    print("2. Repeat this analysis with different time periods")
    print("3. Try different model specifications")
    print("4. Interpret results in light of economic theory")

if __name__ == "__main__":
    main_class_activity()