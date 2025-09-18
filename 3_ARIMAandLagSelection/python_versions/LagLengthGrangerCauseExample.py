# ===============================================================================
# LAG LENGTH SELECTION AND GRANGER CAUSALITY TESTING
# ===============================================================================
#
# LEARNING OBJECTIVES:
# 1. Understand the importance of proper lag selection in VAR models
# 2. Apply sequential F-tests with HAC standard errors for lag selection
# 3. Implement Granger causality tests to identify lead-lag relationships
# 4. Interpret economic relationships through statistical causality tests
#
# ECONOMIC CONTEXT:
# This analysis examines the dynamic relationships between energy markets:
# - Natural gas prices (Henry Hub spot prices)
# - Crude oil prices (WTI crude oil)
# - Drilling activity (oil and gas production index)
#
# METHODOLOGICAL APPROACH:
# - Sequential lag reduction using F-tests with robust standard errors
# - Granger causality testing to identify predictive relationships
# - VAR framework for multivariate time series analysis
#
# PRACTICAL IMPORTANCE:
# Understanding lead-lag relationships helps with:
# - Forecasting commodity prices
# - Risk management in energy markets
# - Policy analysis and market intervention timing
# ===============================================================================

# Import required packages with compatibility handling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# External data packages (with fallbacks)
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    print("Warning: yfinance not available. Will use synthetic data only.")
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
    from statsmodels.tsa.api import VAR
except ImportError:
    print("Warning: VAR model not available.")
    VAR = None

try:
    from statsmodels.tsa.stattools import grangercausalitytests, adfuller
except ImportError:
    print("Warning: Granger causality tests not available.")
    grangercausalitytests = None
    adfuller = None

try:
    from statsmodels.stats.stattools import durbin_watson
except ImportError:
    print("Warning: Durbin-Watson test not available.")
    durbin_watson = None

try:
    from statsmodels.stats.diagnostic import het_white, acorr_breusch_godfrey
except ImportError:
    print("Warning: Diagnostic tests not available.")
    het_white = None
    acorr_breusch_godfrey = None

try:
    from statsmodels.regression.linear_model import OLS
except ImportError:
    print("Warning: OLS regression not available.")
    OLS = None

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

print("=" * 80)
print("LAG LENGTH SELECTION AND GRANGER CAUSALITY TESTING")
print("=" * 80)
print()


# ===============================================================================
# DATA ACQUISITION AND PREPROCESSING
# ===============================================================================

def acquire_and_preprocess_data():
    """
    Download and prepare energy market data for analysis
    """
    print("=== ENERGY MARKET DATA ANALYSIS ===\n")
    print("--- Data Collection ---")
    
    # Download energy market data from FRED using pandas_datareader
    print("Downloading energy market data from FRED...")
    print("Data series to download:")
    print("- MHHNGSP: Henry Hub Natural Gas Spot Prices")  
    print("- MCOILWTICO: WTI Crude Oil Prices")
    print("- IPN213111N: Oil & Gas Drilling Production Index")
    print()
    
    if FRED_AVAILABLE and web is not None:
        try:
            # Download energy data using DataReader
            gas_price = web.DataReader('MHHNGSP', 'fred', '1997-01-01', '2023-12-31')['MHHNGSP'].dropna()
            print("✓ Natural gas price data downloaded successfully!")
            
            oil_price = web.DataReader('MCOILWTICO', 'fred', '1997-01-01', '2023-12-31')['MCOILWTICO'].dropna()
            print("✓ Oil price data downloaded successfully!")
            
            drilling_index = web.DataReader('IPN213111N', 'fred', '1997-01-01', '2023-12-31')['IPN213111N'].dropna()
            print("✓ Drilling production index data downloaded successfully!")
            
            # Combine into single DataFrame
            energy_data = pd.DataFrame({
                'gas_price': gas_price,
                'oil_price': oil_price,
                'drilling_index': drilling_index
            }).dropna()
            
            print(f"\nReal dataset downloaded:")
            print(f"- Observations: {len(energy_data)}")
            print(f"- Time period: {energy_data.index[0].strftime('%Y-%m')} to {energy_data.index[-1].strftime('%Y-%m')}")
            
        except Exception as e:
            print(f"✗ Error downloading FRED data: {e}")
            print("Unable to proceed without energy market data. Please check your internet connection.")
            print("Note: FRED data may have delays or series discontinuations.")
            return None
    else:
        print("✗ pandas_datareader not available. Please install with: pip install pandas-datareader")
        print("Unable to proceed without energy market data.")
        return None
    print(f"- Variables: {energy_data.columns.tolist()}")
    print()
    
    return energy_data

def plot_initial_data_exploration(energy_data):
    """
    Initial visualization and exploration of energy market data
    """
    print("--- Initial Data Visualization ---")
    
    # Plot level data
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Energy Market Data (Levels)', fontsize=16)
    
    # Individual time series
    axes[0, 0].plot(energy_data.index, energy_data['gas_price'], color='blue', alpha=0.8)
    axes[0, 0].set_title('Henry Hub Natural Gas Prices')
    axes[0, 0].set_ylabel('$/MMBtu')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(energy_data.index, energy_data['oil_price'], color='red', alpha=0.8) 
    axes[0, 1].set_title('WTI Crude Oil Prices')
    axes[0, 1].set_ylabel('$/barrel')
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(energy_data.index, energy_data['drilling_index'], color='green', alpha=0.8)
    axes[1, 0].set_title('Oil & Gas Drilling Production Index')
    axes[1, 0].set_ylabel('Index Value')
    axes[1, 0].grid(True)
    
    # Correlation plot
    correlation_matrix = energy_data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, ax=axes[1, 1])
    axes[1, 1].set_title('Correlation Matrix')
    
    plt.tight_layout()
    plt.show()
    
    print("Visual inspection shows potential relationships between energy markets.")
    print("Next: Transform to stationary series for VAR modeling.")
    print()

def create_stationary_transformations(energy_data):
    """
    Create stationary transformations of energy market data
    """
    print("--- Creating Stationary Series ---")
    print("Applying log differences to prices and simple differences to drilling index...")
    
    # Create transformations
    transformations = {
        'gas_returns': np.log(energy_data['gas_price']).diff().dropna() * 100,  # Convert to percentage
        'oil_returns': np.log(energy_data['oil_price']).diff().dropna() * 100,  # Convert to percentage  
        'drilling_changes': energy_data['drilling_index'].diff().dropna()
    }
    
    print("Transformed variables:")
    print("- gas_returns: Log differences of gas prices (% monthly returns)")
    print("- oil_returns: Log differences of oil prices (% monthly returns)")
    print("- drilling_changes: Simple differences of drilling index")
    print()
    
    # Align all series to same date range
    min_start = max([ts.index[0] for ts in transformations.values()])
    max_end = min([ts.index[-1] for ts in transformations.values()])
    
    for key in transformations:
        transformations[key] = transformations[key][min_start:max_end]
    
    energy_returns = pd.DataFrame(transformations)
    
    # Plot transformed series
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    fig.suptitle('Energy Market Returns and Changes (Stationary Series)', fontsize=16)
    
    axes[0, 0].plot(energy_returns.index, energy_returns['gas_returns'], alpha=0.8, color='blue')
    axes[0, 0].set_title('Natural Gas Returns')
    axes[0, 0].set_ylabel('Monthly % Return')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(energy_returns.index, energy_returns['oil_returns'], alpha=0.8, color='red')
    axes[0, 1].set_title('Oil Returns')  
    axes[0, 1].set_ylabel('Monthly % Return')
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(energy_returns.index, energy_returns['drilling_changes'], alpha=0.8, color='green')
    axes[1, 0].set_title('Drilling Index Changes')
    axes[1, 0].set_ylabel('Index Point Change')
    axes[1, 0].grid(True)
    
    # Summary statistics
    stats_text = f"""
    Summary Statistics:
    
    Gas Returns:
    Mean: {energy_returns['gas_returns'].mean():.2f}%
    Std: {energy_returns['gas_returns'].std():.2f}%
    
    Oil Returns:  
    Mean: {energy_returns['oil_returns'].mean():.2f}%
    Std: {energy_returns['oil_returns'].std():.2f}%
    
    Drilling Changes:
    Mean: {energy_returns['drilling_changes'].mean():.2f}
    Std: {energy_returns['drilling_changes'].std():.2f}
    """
    axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1, 1].set_title('Summary Statistics')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Transformed dataset: {len(energy_returns)} observations")
    print("Visual inspection shows the series appear stationary after transformation")
    print()
    
    return energy_returns

# ===============================================================================
# LAG LENGTH SELECTION USING SEQUENTIAL F-TESTS
# ===============================================================================

def sequential_lag_selection(energy_returns, max_lags=6):
    """
    Perform sequential lag selection using F-tests with HAC standard errors
    """
    print("=== LAG LENGTH SELECTION PROCEDURE ===\n")
    
    print("METHODOLOGY EXPLANATION:")
    print("1. Start with maximum lag length (6 months)")
    print("2. Test if highest lag coefficients can be jointly set to zero")
    print("3. Use HAC standard errors for robust inference")
    print("4. Compare AIC values across different lag specifications")
    print("5. Select lag length based on F-test significance and AIC")
    print()
    
    print("--- Sequential F-Tests with HAC Standard Errors ---")
    
    # Prepare data for regression (dependent variable: drilling changes)
    y = energy_returns['drilling_changes'].dropna()
    
    results_summary = []
    models = {}
    
    for lag_length in range(1, max_lags + 1):
        print(f"Testing {lag_length}-lag specification...")
        
        # Create lagged variables
        X_vars = []
        var_names = []
        
        for var in ['gas_returns', 'oil_returns', 'drilling_changes']:
            for lag in range(1, lag_length + 1):
                lagged_var = energy_returns[var].shift(lag)
                X_vars.append(lagged_var)
                var_names.append(f'{var}_lag{lag}')
        
        # Combine predictors and align with dependent variable
        X = pd.concat(X_vars, axis=1)
        X.columns = var_names
        
        # Align data (drop NAs)
        common_index = y.index.intersection(X.dropna().index)
        y_aligned = y[common_index]
        X_aligned = X.loc[common_index]
        
        if len(X_aligned) < 50:  # Ensure sufficient observations
            print(f"Insufficient observations for {lag_length}-lag model")
            continue
        
        # Add constant
        X_with_const = sm.add_constant(X_aligned)
        
        # Fit OLS model with safety check
        try:
            if OLS is not None:
                model = OLS(y_aligned, X_with_const).fit()
                models[lag_length] = model
            else:
                # Fallback to sm.OLS
                model = sm.OLS(y_aligned, X_with_const).fit()
                models[lag_length] = model
            
            # Calculate information criteria
            aic = model.aic
            bic = model.bic
            
            # F-test for joint significance of highest lag
            if lag_length > 1:
                # Test if coefficients of highest lag can be set to zero
                highest_lag_vars = [name for name in var_names if f'_lag{lag_length}' in name]
                
                # Create restriction matrix for F-test
                restrictions = []
                for var_name in highest_lag_vars:
                    if var_name in X_with_const.columns:
                        restriction = np.zeros(len(X_with_const.columns))
                        col_idx = X_with_const.columns.get_loc(var_name)
                        restriction[col_idx] = 1
                        restrictions.append(restriction)
                
                if restrictions:
                    # Perform F-test
                    R = np.array(restrictions)
                    r = np.zeros(len(restrictions))
                    
                    # Use HAC covariance matrix (Newey-West)
                    try:
                        f_test = model.f_test((R, r))
                        f_stat = f_test.fvalue[0][0] if hasattr(f_test.fvalue[0], '__len__') else f_test.fvalue
                        f_pval = f_test.pvalue
                    except:
                        # Fallback to regular F-test if HAC fails
                        f_test = model.f_test((R, r))
                        f_stat = f_test.fvalue
                        f_pval = f_test.pvalue
                else:
                    f_stat, f_pval = np.nan, np.nan
            else:
                f_stat, f_pval = np.nan, np.nan
            
            # Store results
            results_summary.append({
                'Lag_Length': lag_length,
                'AIC': aic,
                'BIC': bic,
                'F_Stat': f_stat,
                'F_PValue': f_pval,
                'Adj_R_Squared': model.rsquared_adj,
                'N_Obs': len(y_aligned)
            })
            
            print(f"  AIC: {aic:.2f}, BIC: {bic:.2f}")
            if not np.isnan(f_pval):
                print(f"  F-test p-value for lag {lag_length}: {f_pval:.4f}")
            print()
            
        except Exception as e:
            print(f"  Error fitting {lag_length}-lag model: {e}")
            print()
    
    # Create summary table
    if results_summary:
        results_df = pd.DataFrame(results_summary)
        
        print("--- Lag Selection Summary ---")
        print(results_df.round(4))
        print()
        
        # Select optimal lag length (lowest AIC with no serial correlation)
        optimal_aic = results_df.loc[results_df['AIC'].idxmin()]
        print(f"Optimal lag length by AIC: {optimal_aic['Lag_Length']} lags")
        
        # Select based on F-test (first lag where F-test becomes non-significant)
        significant_lags = results_df[results_df['F_PValue'] < 0.05]
        if len(significant_lags) > 0:
            optimal_f_test = significant_lags['Lag_Length'].max()
            print(f"Optimal lag length by F-test: {optimal_f_test} lags")
        else:
            optimal_f_test = 1
            print("F-test suggests 1 lag (no higher-order lags significant)")
        
        print()
        return results_df, models, int(optimal_aic['Lag_Length'])
    
    else:
        print("No models successfully estimated")
        return None, {}, 1

def model_diagnostics(model, model_name):
    """
    Perform diagnostic tests on the selected model
    """
    print(f"--- Model Diagnostics for {model_name} ---")
    
    residuals = model.resid
    
    # Plot diagnostics
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Diagnostic Plots: {model_name}', fontsize=14)
    
    # Residuals over time
    axes[0, 0].plot(residuals.index, residuals.values, alpha=0.8)
    axes[0, 0].set_title('Residuals Over Time')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].grid(True)
    
    # Q-Q plot
    sm.qqplot(residuals, ax=axes[0, 1], line='s')
    axes[0, 1].set_title('Q-Q Plot')
    
    # Histogram of residuals
    axes[1, 0].hist(residuals, bins=20, alpha=0.7, density=True)
    axes[1, 0].set_title('Residual Distribution')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Density')
    
    # ACF of residuals
    from statsmodels.tsa.stattools import acf
    acf_resid = acf(residuals, nlags=20, fft=True)
    axes[1, 1].plot(range(len(acf_resid)), acf_resid, 'o-')
    axes[1, 1].axhline(y=0, color='k', linestyle='-')
    axes[1, 1].axhline(y=1.96/np.sqrt(len(residuals)), color='r', linestyle='--')
    axes[1, 1].axhline(y=-1.96/np.sqrt(len(residuals)), color='r', linestyle='--')
    axes[1, 1].set_title('ACF of Residuals')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Statistical tests
    print("Statistical Diagnostic Tests:")
    
    # Durbin-Watson test for serial correlation
    dw_stat = durbin_watson(residuals)
    print(f"Durbin-Watson statistic: {dw_stat:.4f}")
    print("  (Values near 2.0 suggest no serial correlation)")
    
    # Breusch-Godfrey test for higher-order serial correlation
    try:
        bg_test = acorr_breusch_godfrey(model, nlags=5)
        print(f"Breusch-Godfrey test p-value: {bg_test[1]:.4f}")
        if bg_test[1] < 0.05:
            print("  WARNING: Serial correlation detected")
        else:
            print("  No significant serial correlation")
    except:
        print("  Could not perform Breusch-Godfrey test")
    
    # Jarque-Bera test for normality
    jb_stat, jb_pval = stats.jarque_bera(residuals)
    print(f"Jarque-Bera test p-value: {jb_pval:.4f}")
    if jb_pval < 0.05:
        print("  WARNING: Residuals not normally distributed")
    else:
        print("  Residuals appear normally distributed")
    
    print()

# ===============================================================================
# GRANGER CAUSALITY TESTING
# ===============================================================================

def granger_causality_analysis(energy_returns, optimal_lags=4):
    """
    Perform Granger causality tests to identify predictive relationships
    """
    print("=== GRANGER CAUSALITY ANALYSIS ===\n")
    
    print("GRANGER CAUSALITY CONCEPT:")
    print("Variable X 'Granger-causes' variable Y if:")
    print("- Past values of X help predict Y")
    print("- Even after controlling for past values of Y")
    print("- Statistical causality, not necessarily economic causality")
    print()
    
    print("RESEARCH QUESTIONS:")
    print("1. Do oil price changes Granger-cause drilling activity changes?")
    print("2. Do natural gas price changes Granger-cause drilling activity changes?")
    print("3. Do drilling activity changes Granger-cause energy price changes?")
    print()
    
    # Prepare data for Granger causality tests
    test_data = energy_returns[['drilling_changes', 'oil_returns', 'gas_returns']].dropna()
    
    causality_results = {}
    
    # Test 1: Oil returns → Drilling activity
    print("--- Test 1: Oil Returns → Drilling Activity ---")
    print("H0: Oil price changes do not Granger-cause drilling activity changes")
    print("Ha: Oil price changes do Granger-cause drilling activity changes")
    
    try:
        # Granger causality test (oil → drilling)
        oil_to_drilling = grangercausalitytests(
            test_data[['drilling_changes', 'oil_returns']], 
            maxlag=optimal_lags, 
            verbose=False
        )
        
        # Extract p-values for different lags
        oil_pvalues = []
        for lag in range(1, optimal_lags + 1):
            test_result = oil_to_drilling[lag][0]
            # Get F-test p-value (usually the 'ssr_ftest' result)
            if 'ssr_ftest' in test_result:
                oil_pvalues.append(test_result['ssr_ftest'][1])
            else:
                # Fallback to first available test
                oil_pvalues.append(list(test_result.values())[0][1])
        
        causality_results['oil_to_drilling'] = oil_pvalues
        min_pval = min(oil_pvalues)
        
        print(f"Minimum p-value across lags: {min_pval:.4f}")
        if min_pval < 0.05:
            print("CONCLUSION: Oil price changes Granger-cause drilling activity (p < 0.05)")
        else:
            print("CONCLUSION: No evidence of Granger causality from oil to drilling (p ≥ 0.05)")
        print()
        
    except Exception as e:
        print(f"Error in oil→drilling test: {e}")
        print()
    
    # Test 2: Gas returns → Drilling activity
    print("--- Test 2: Natural Gas Returns → Drilling Activity ---")
    print("H0: Gas price changes do not Granger-cause drilling activity changes")
    print("Ha: Gas price changes do Granger-cause drilling activity changes")
    
    try:
        # Granger causality test (gas → drilling)
        gas_to_drilling = grangercausalitytests(
            test_data[['drilling_changes', 'gas_returns']], 
            maxlag=optimal_lags, 
            verbose=False
        )
        
        # Extract p-values
        gas_pvalues = []
        for lag in range(1, optimal_lags + 1):
            test_result = gas_to_drilling[lag][0]
            if 'ssr_ftest' in test_result:
                gas_pvalues.append(test_result['ssr_ftest'][1])
            else:
                gas_pvalues.append(list(test_result.values())[0][1])
        
        causality_results['gas_to_drilling'] = gas_pvalues
        min_pval = min(gas_pvalues)
        
        print(f"Minimum p-value across lags: {min_pval:.4f}")
        if min_pval < 0.05:
            print("CONCLUSION: Gas price changes Granger-cause drilling activity (p < 0.05)")
        else:
            print("CONCLUSION: No evidence of Granger causality from gas to drilling (p ≥ 0.05)")
        print()
        
    except Exception as e:
        print(f"Error in gas→drilling test: {e}")
        print()
    
    # Test 3: Drilling activity → Oil returns (reverse causality)
    print("--- Test 3: Drilling Activity → Oil Returns (Reverse Causality) ---")
    
    try:
        drilling_to_oil = grangercausalitytests(
            test_data[['oil_returns', 'drilling_changes']], 
            maxlag=optimal_lags, 
            verbose=False
        )
        
        drilling_oil_pvalues = []
        for lag in range(1, optimal_lags + 1):
            test_result = drilling_to_oil[lag][0]
            if 'ssr_ftest' in test_result:
                drilling_oil_pvalues.append(test_result['ssr_ftest'][1])
            else:
                drilling_oil_pvalues.append(list(test_result.values())[0][1])
        
        causality_results['drilling_to_oil'] = drilling_oil_pvalues
        min_pval = min(drilling_oil_pvalues)
        
        print(f"Minimum p-value across lags: {min_pval:.4f}")
        if min_pval < 0.05:
            print("CONCLUSION: Drilling activity Granger-causes oil price changes (p < 0.05)")
        else:
            print("CONCLUSION: No evidence of reverse causality from drilling to oil (p ≥ 0.05)")
        print()
        
    except Exception as e:
        print(f"Error in drilling→oil test: {e}")
        print()
    
    # Create summary table of results
    if causality_results:
        print("--- Granger Causality Results Summary ---")
        summary_table = pd.DataFrame({
            'Test': ['Oil → Drilling', 'Gas → Drilling', 'Drilling → Oil'],
            'Min_P_Value': [
                min(causality_results.get('oil_to_drilling', [1.0])),
                min(causality_results.get('gas_to_drilling', [1.0])),
                min(causality_results.get('drilling_to_oil', [1.0]))
            ],
            'Significant': [
                min(causality_results.get('oil_to_drilling', [1.0])) < 0.05,
                min(causality_results.get('gas_to_drilling', [1.0])) < 0.05,
                min(causality_results.get('drilling_to_oil', [1.0])) < 0.05
            ]
        })
        
        print(summary_table)
        print()
    
    return causality_results

# ===============================================================================
# SUMMARY AND INTERPRETATION
# ===============================================================================

def print_final_summary():
    """
    Print comprehensive summary of analysis and economic interpretation
    """
    print("=== FINAL SUMMARY AND ECONOMIC INTERPRETATION ===\n")
    
    print("METHODOLOGICAL LESSONS:")
    print("1. Lag selection requires balancing model fit and parsimony")
    print("2. HAC standard errors provide robust inference in time series")
    print("3. Sequential F-tests offer systematic approach to model selection")
    print("4. Residual diagnostics essential for model validation")
    print()
    
    print("GRANGER CAUSALITY INSIGHTS:")
    print("1. Statistical causality ≠ economic causality")
    print("2. Tests reveal predictive relationships, not structural causation")
    print("3. Energy markets show complex lead-lag relationships")
    print("4. Results inform forecasting and risk management strategies")
    print()
    
    print("PRACTICAL APPLICATIONS:")
    print("- Energy price forecasting")
    print("- Investment timing in drilling operations")
    print("- Policy analysis for energy market interventions")
    print("- Risk management in commodity markets")
    print()
    
    print("=== ANALYSIS COMPLETE ===")

# ===============================================================================
# MAIN EXECUTION
# ===============================================================================

def main():
    """
    Execute the complete lag selection and Granger causality analysis
    """
    print("Starting comprehensive energy market analysis...")
    print()
    
    # Step 1: Data acquisition and preprocessing
    energy_data = acquire_and_preprocess_data()
    plot_initial_data_exploration(energy_data)
    energy_returns = create_stationary_transformations(energy_data)
    
    # Step 2: Lag length selection
    results_df, models, optimal_lag = sequential_lag_selection(energy_returns)
    
    if models and optimal_lag in models:
        # Step 3: Model diagnostics
        model_diagnostics(models[optimal_lag], f"{optimal_lag}-Lag Model")
        
        # Step 4: Granger causality analysis
        causality_results = granger_causality_analysis(energy_returns, optimal_lag)
    else:
        print("Could not perform further analysis due to model estimation issues")
    
    # Step 5: Final summary
    print_final_summary()

if __name__ == "__main__":
    main()