# ===============================================================================
# COMPREHENSIVE ARIMA MODELING AND LAG SELECTION TUTORIAL
# ===============================================================================
#
# LEARNING OBJECTIVES:
# 1. Understand the process of ARIMA model identification and estimation
# 2. Learn multiple approaches to lag selection in time series models
# 3. Master residual diagnostics for assessing model adequacy
# 4. Generate and evaluate forecasts using holdout samples
# 5. Interpret cyclical patterns through characteristic polynomial roots
# 6. Compare model performance using information criteria
#
# METHODOLOGICAL APPROACH:
# This tutorial follows the Box-Jenkins methodology:
# 1. Model Identification (ACF/PACF analysis, automatic selection)
# 2. Model Estimation (parameter fitting)
# 3. Model Diagnostic Checking (residual analysis)
# 4. Forecasting (prediction and evaluation)
#
# DATA: We analyze two contrasting time series:
# - Industrial drilling activity (highly cyclical economic series)
# - Unemployment rates (macroeconomic indicator with persistence)
# ===============================================================================

# Import required packages with compatibility handling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# External data packages
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
    from statsmodels.tsa.stattools import acf, pacf, adfuller
except ImportError:
    print("Warning: ACF/PACF functions not available.")
    acf = None
    pacf = None
    adfuller = None

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

# Set plotting style
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
print("COMPREHENSIVE ARIMA MODELING AND LAG SELECTION TUTORIAL")
print("=" * 80)
print()

# ===============================================================================
# CASE STUDY 1: DRILLING ACTIVITY
# ===============================================================================

def load_drilling_data():
    """
    Load and preprocess industrial drilling activity data from FRED
    """
    print("=== CASE STUDY 1: DRILLING ACTIVITY ===\n")
    print("Data Acquisition - Retrieve drilling activity data from FRED")
    
    if not FRED_AVAILABLE or web is None:
        print("✗ pandas_datareader not available. Please install with: pip install pandas-datareader")
        print("Unable to proceed without FRED data access.")
        return None, None
    
    try:
        # Get oil and gas drilling production index
        print("Downloading Industrial Production: Drilling Oil and Gas Wells (IPN213111S)")
        start_date = "1972-04-01"
        end_date = "2022-08-03"
        
        drilling_data = web.DataReader('IPN213111S', 'fred', start_date, end_date)['IPN213111S'].dropna()
        print(f"✓ Data downloaded successfully: {len(drilling_data)} observations")
        print(f"Date range: {drilling_data.index[0]} to {drilling_data.index[-1]}\n")
        
        # Calculate first differences to achieve stationarity
        drilling_changes = drilling_data.diff().dropna()
        
        return drilling_data, drilling_changes
        
    except Exception as e:
        print(f"✗ Error downloading drilling data: {e}")
        print("Unable to proceed without drilling activity data.")
        return None, None

def visualize_drilling_data(drilling_data, drilling_changes):
    """
    Visualize drilling data in levels and differences
    """
    print("--- Initial Data Visualization ---")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Oil and Gas Drilling Production Index', fontsize=16)
    
    # Level data
    axes[0, 0].plot(drilling_data.index, drilling_data.values, alpha=0.8, color='blue')
    axes[0, 0].set_title('Drilling Production Index (Levels)')
    axes[0, 0].set_ylabel('Index Value')
    axes[0, 0].grid(True)
    
    # ACF of levels
    if acf is not None:
        try:
            acf_levels = acf(drilling_data, nlags=50, fft=True)
            axes[0, 1].plot(range(len(acf_levels)), acf_levels, 'o-', markersize=3)
            axes[0, 1].axhline(y=0, color='k', linestyle='-')
            axes[0, 1].axhline(y=1.96/np.sqrt(len(drilling_data)), color='r', linestyle='--')
            axes[0, 1].axhline(y=-1.96/np.sqrt(len(drilling_data)), color='r', linestyle='--')
            axes[0, 1].set_title('ACF of Drilling Production Index')
            axes[0, 1].set_xlabel('Lag')
            axes[0, 1].set_ylabel('ACF')
            axes[0, 1].grid(True)
        except:
            axes[0, 1].text(0.5, 0.5, 'ACF computation failed', ha='center', va='center')
    
    # Differenced data
    axes[1, 0].plot(drilling_changes.index, drilling_changes.values, alpha=0.8, color='red')
    axes[1, 0].set_title('Changes in Drilling Production Index')
    axes[1, 0].set_ylabel('First Difference')
    axes[1, 0].grid(True)
    
    # ACF and PACF of changes
    if acf is not None and pacf is not None:
        try:
            acf_changes = acf(drilling_changes, nlags=30, fft=True)
            pacf_changes = pacf(drilling_changes, nlags=30)
            
            # Plot ACF
            x_vals = range(len(acf_changes))
            axes[1, 1].plot(x_vals, acf_changes, 'o-', markersize=3, label='ACF')
            axes[1, 1].axhline(y=0, color='k', linestyle='-')
            axes[1, 1].axhline(y=1.96/np.sqrt(len(drilling_changes)), color='r', linestyle='--')
            axes[1, 1].axhline(y=-1.96/np.sqrt(len(drilling_changes)), color='r', linestyle='--')
            axes[1, 1].set_title('ACF/PACF of Drilling Changes')
            axes[1, 1].set_xlabel('Lag')
            axes[1, 1].set_ylabel('Correlation')
            axes[1, 1].grid(True)
            axes[1, 1].legend()
        except:
            axes[1, 1].text(0.5, 0.5, 'ACF/PACF computation failed', ha='center', va='center')
    
    plt.tight_layout()
    plt.show()
    print()

def evaluate_arima_model(model, model_name, data, gof_lag=36):
    """
    Comprehensive evaluation of ARIMA model with diagnostics
    """
    print("=" * 60)
    print(f"Model Evaluation: {model_name}")
    print("=" * 60)
    
    # Print model summary
    print(model.summary())
    print()
    
    # Information criteria
    aic_value = model.aic
    bic_value = model.bic
    
    print("Information Criteria:")
    print(f"AIC: {aic_value:.2f}")
    print(f"BIC: {bic_value:.2f}\n")
    
    # Residual analysis
    residuals = model.resid
    
    # Create comprehensive diagnostic plots (tsdiag + tsdisplay style)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Comprehensive Model Diagnostics: {model_name}', fontsize=16)
    
    # Top row: tsdiag() style plots
    # 1. Standardized residuals over time
    axes[0, 0].plot(residuals.index, residuals.values, alpha=0.8, linewidth=0.8)
    axes[0, 0].set_title('Standardized Residuals')
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
    
    # 3. Box-Ljung test p-values for increasing lags (tsdiag style)
    try:
        test_lags = range(1, min(gof_lag + 1, len(residuals)//4))
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
        axes[0, 2].fill_between(test_lags, 0, 0.05, alpha=0.2, color='red', 
                               label='Rejection region (5%)')
        
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
    
    # Box-Ljung test for residual autocorrelation
    try:
        lb_test = acorr_ljungbox(residuals, lags=gof_lag, return_df=True)
        lb_stat = lb_test['lb_stat'].iloc[-1]
        lb_pval = lb_test['lb_pvalue'].iloc[-1]
        
        print(f"Box-Ljung Test (lag = {gof_lag}):")
        print(f"Q statistic: {lb_stat:.4f}")
        print(f"p-value: {lb_pval:.4f}")
        
        # Adjusted p-value with degrees of freedom correction
        n_params = len(model.params)
        p_df = gof_lag - n_params
        adj_p_value = 1 - stats.chi2.cdf(lb_stat, p_df)
        print(f"Adjusted p-value (df = {p_df}): {adj_p_value:.4f}")
        
    except Exception as e:
        print(f"Could not compute Box-Ljung test: {e}")
        adj_p_value = np.nan
    
    print("-" * 60)
    print()
    
    return {
        'model': model,
        'model_name': model_name,
        'aic': aic_value,
        'bic': bic_value,
        'adj_p_value': adj_p_value
    }

def fit_drilling_models(drilling_changes):
    """
    Fit and evaluate various ARIMA models for drilling data
    """
    print("=== MODEL IDENTIFICATION AND ESTIMATION ===\n")
    print("Fitting various ARMA models and evaluating their performance\n")
    
    model_results = {}
    
    # Model 1: AR(15) - Full model
    print("Fitting AR(15) - All Coefficients...")
    try:
        if AutoReg is not None:
            ar15_model = AutoReg(drilling_changes, lags=15).fit()
            # Convert to ARIMA format for consistency
            ar15_arima = ARIMA(drilling_changes, order=(15, 0, 0)).fit()
        else:
            ar15_arima = ARIMA(drilling_changes, order=(15, 0, 0)).fit()
        
        model_results['AR15'] = evaluate_arima_model(
            ar15_arima, "AR(15) - All Coefficients", drilling_changes
        )
    except Exception as e:
        print(f"Could not fit AR(15) model: {e}")
        model_results['AR15'] = None
    
    # Model 2: Auto-ARIMA
    print("Fitting Auto-ARIMA model...")
    try:
        # Simple automatic selection
        best_aic = np.inf
        best_order = None
        best_model = None
        
        for p in range(6):
            for q in range(6):
                try:
                    model = ARIMA(drilling_changes, order=(p, 0, q)).fit()
                    if model.aic < best_aic:
                        best_aic = model.aic
                        best_order = (p, 0, q)
                        best_model = model
                except:
                    continue
        
        if best_model is not None:
            model_results['Auto'] = evaluate_arima_model(
                best_model, f"Auto-ARIMA: ARIMA{best_order}", drilling_changes
            )
        else:
            print("Could not find suitable auto-ARIMA model")
            model_results['Auto'] = None
            
    except Exception as e:
        print(f"Could not fit Auto-ARIMA model: {e}")
        model_results['Auto'] = None
    
    # Model 3: AR(8) based on visual inspection
    print("Fitting AR(8) model...")
    try:
        ar8_model = ARIMA(drilling_changes, order=(8, 0, 0)).fit()
        model_results['AR8'] = evaluate_arima_model(
            ar8_model, "AR(8)", drilling_changes
        )
    except Exception as e:
        print(f"Could not fit AR(8) model: {e}")
        model_results['AR8'] = None
    
    # Model 4: ARMA(3,3) for comparison
    print("Fitting ARMA(3,3) model...")
    try:
        arma33_model = ARIMA(drilling_changes, order=(3, 0, 3)).fit()
        model_results['ARMA33'] = evaluate_arima_model(
            arma33_model, "ARMA(3,3)", drilling_changes
        )
    except Exception as e:
        print(f"Could not fit ARMA(3,3) model: {e}")
        model_results['ARMA33'] = None
    
    return model_results

def compare_drilling_models(model_results):
    """
    Compare model performance using information criteria
    """
    print("=== MODEL COMPARISON ===\n")
    
    # Create comparison table
    comparison_data = []
    for key, result in model_results.items():
        if result is not None:
            comparison_data.append({
                'Model': result['model_name'],
                'AIC': result['aic'],
                'BIC': result['bic'],
                'Residual_Autocorr': 'No' if result['adj_p_value'] > 0.05 else 'Yes'
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('AIC')
        
        print("Model Comparison (sorted by AIC):")
        print("=" * 60)
        print(comparison_df.to_string(index=False))
        print("=" * 60)
        print()
        
        # Select best model
        best_model_name = comparison_df.iloc[0]['Model']
        print(f"Best model by AIC: {best_model_name}\n")
        
        # Find best model in results
        best_model = None
        for result in model_results.values():
            if result is not None and result['model_name'] == best_model_name:
                best_model = result
                break
        
        return best_model, comparison_df
    else:
        print("No models successfully estimated.")
        return None, None

def generate_forecasts_with_ci(model, h=50, plot_title=None):
    """
    Generate forecasts with confidence intervals and plotting
    """
    print(f"Generating forecasts (h={h})...")
    
    try:
        # Generate forecasts with confidence intervals
        forecast_result = model.get_forecast(steps=h)
        forecast_mean = forecast_result.predicted_mean
        forecast_ci = forecast_result.conf_int()
        
        # Handle confidence intervals robustly
        if hasattr(forecast_ci, 'iloc'):
            ci_lower = forecast_ci.iloc[:, 0]
            ci_upper = forecast_ci.iloc[:, 1]
        else:
            ci_lower = forecast_ci[:, 0]
            ci_upper = forecast_ci[:, 1]
        
        # Plot forecasts
        if plot_title:
            plt.figure(figsize=(12, 6))
            
            # Plot historical data (last 100 points)
            historical = model.fittedvalues[-100:]
            plt.plot(range(len(historical)), historical, label='Historical', alpha=0.8, color='blue')
            
            # Plot forecasts
            forecast_index = range(len(historical), len(historical) + len(forecast_mean))
            plt.plot(forecast_index, forecast_mean, label='Forecast', color='red', linewidth=2)
            plt.fill_between(forecast_index, ci_lower, ci_upper, 
                           color='red', alpha=0.2, label='95% CI')
            
            plt.axvline(x=len(historical), color='k', linestyle='--', alpha=0.5)
            plt.title(plot_title)
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            plt.show()
        
        # Create forecast table
        forecast_table = pd.DataFrame({
            'Forecast': forecast_mean,
            'Lower_95': ci_lower,
            'Upper_95': ci_upper
        })
        
        return forecast_table, forecast_result
        
    except Exception as e:
        print(f"Error generating forecasts: {e}")
        return None, None

def evaluate_forecast_accuracy_holdout(data, model_order, h=50, holdout_indices=None, model_name="Model"):
    """
    Evaluate forecast accuracy using a holdout sample with comprehensive metrics
    """
    print(f"=== HOLDOUT SAMPLE EVALUATION: {model_name.upper()} ===\n")
    
    # Convert to numpy array for easier indexing
    data_array = np.array(data)
    
    # Determine holdout indices if not provided
    if holdout_indices is None:
        n = len(data_array)
        holdout_indices = list(range(n-h, n))
    
    print(f"Holdout period: {len(holdout_indices)} observations")
    print(f"Training period: {len(data_array) - len(holdout_indices)} observations\n")
    
    # Split data into training and holdout
    training_indices = [i for i in range(len(data_array)) if i not in holdout_indices]
    training_data = data_array[training_indices]
    holdout_data = data_array[holdout_indices]
    
    try:
        # Fit model on training data
        if isinstance(model_order, tuple) and len(model_order) == 3:
            p, d, q = model_order
            training_model = ARIMA(training_data, order=(p, d, q)).fit()
        else:
            print(f"Invalid model order: {model_order}")
            return None
        
        # Generate forecasts for holdout period
        forecast_result = training_model.get_forecast(steps=len(holdout_indices))
        forecasts = forecast_result.predicted_mean
        forecast_ci = forecast_result.conf_int()
        
        # Handle confidence intervals
        if hasattr(forecast_ci, 'iloc'):
            ci_lower = forecast_ci.iloc[:, 0]
            ci_upper = forecast_ci.iloc[:, 1]
        else:
            ci_lower = forecast_ci[:, 0]
            ci_upper = forecast_ci[:, 1]
        
        # Plot forecast evaluation
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f'Holdout Sample Evaluation: {model_name}', fontsize=14)
        
        # Top plot: Full series with training/holdout split
        full_series_index = range(len(data_array))
        axes[0].plot(full_series_index, data_array, label='Full Series', alpha=0.7, color='blue')
        
        # Highlight training period
        axes[0].plot(training_indices, training_data, label='Training Data', alpha=0.8, color='green')
        
        # Plot forecasts over holdout period
        holdout_index = range(holdout_indices[0], holdout_indices[0] + len(forecasts))
        axes[0].plot(holdout_index, forecasts, label='Forecasts', color='red', linewidth=2)
        axes[0].fill_between(holdout_index, ci_lower, ci_upper, 
                           color='red', alpha=0.2, label='95% CI')
        
        # Plot actual holdout data
        axes[0].plot(holdout_indices, holdout_data, label='Actual Holdout', 
                    color='black', linewidth=2, linestyle='--')
        
        axes[0].axvline(x=holdout_indices[0], color='k', linestyle=':', alpha=0.7, 
                       label='Holdout Start')
        axes[0].set_title('Forecast vs Actual: Full Time Series View')
        axes[0].legend()
        axes[0].grid(True)
        
        # Bottom plot: Zoomed view of holdout period
        axes[1].plot(range(len(holdout_data)), holdout_data, label='Actual', 
                    color='black', linewidth=2, marker='o')
        axes[1].plot(range(len(forecasts)), forecasts, label='Forecast', 
                    color='red', linewidth=2, marker='s')
        axes[1].fill_between(range(len(forecasts)), ci_lower, ci_upper, 
                           color='red', alpha=0.2, label='95% CI')
        axes[1].set_title('Forecast vs Actual: Holdout Period Detail')
        axes[1].set_xlabel('Holdout Period Index')
        axes[1].set_ylabel('Value')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Calculate forecast accuracy measures
        errors = holdout_data - forecasts
        
        accuracy_metrics = {
            'ME': np.mean(errors),                              # Mean Error
            'RMSE': np.sqrt(np.mean(errors**2)),               # Root Mean Squared Error
            'MAE': np.mean(np.abs(errors)),                    # Mean Absolute Error
            'MPE': np.mean(100 * errors / holdout_data),       # Mean Percentage Error
            'MAPE': np.mean(100 * np.abs(errors / holdout_data)) # Mean Absolute Percentage Error
        }
        
        print("Forecast Accuracy Metrics:")
        print("-" * 30)
        for metric, value in accuracy_metrics.items():
            print(f"{metric:>6}: {value:>8.4f}")
        print()
        
        return {
            'model': training_model,
            'forecasts': forecasts,
            'holdout_data': holdout_data,
            'errors': errors,
            'accuracy_metrics': accuracy_metrics,
            'forecast_ci': forecast_ci
        }
        
    except Exception as e:
        print(f"Error in holdout evaluation: {e}")
        return None

def compare_model_forecast_accuracy(model_results, data, holdout_indices=None):
    """
    Compare forecast accuracy across multiple models using holdout sample
    """
    print("=== COMPARATIVE HOLDOUT SAMPLE EVALUATION ===\n")
    
    if not model_results:
        print("No models available for comparison")
        return None
    
    # Filter valid models
    valid_models = {k: v for k, v in model_results.items() if v is not None}
    
    if len(valid_models) < 2:
        print("Need at least 2 valid models for comparison")
        return None
    
    holdout_results = {}
    accuracy_comparison = {}
    
    # Evaluate each model
    for model_key, model_result in valid_models.items():
        try:
            # Extract model order from the fitted model
            model = model_result['model']
            model_order = None
            
            # Method 1: Try to get from model specification dictionary
            if hasattr(model, 'specification') and hasattr(model.specification, 'order'):
                model_order = model.specification.order
            
            # Method 2: Try direct order attribute
            elif hasattr(model, 'order'):
                model_order = model.order
                
            # Method 3: Try to extract from model_orders dictionary
            elif hasattr(model, 'model_orders') and isinstance(model.model_orders, dict):
                try:
                    p = model.model_orders.get('ar', 0)
                    d = 0  # We're working with stationary series
                    q = model.model_orders.get('ma', 0)
                    model_order = (p, d, q)
                except:
                    model_order = None
            
            # Method 4: Parse from model name as fallback
            if model_order is None:
                model_name = model_result['model_name']
                if 'AR(15)' in model_name:
                    model_order = (15, 0, 0)
                elif 'AR(8)' in model_name:
                    model_order = (8, 0, 0)
                elif 'ARMA(3,3)' in model_name:
                    model_order = (3, 0, 3)
                elif 'Auto-ARIMA' in model_name:
                    # Try to extract from the ARIMA(...) part of the name
                    import re
                    match = re.search(r'ARIMA\((\d+),(\d+),(\d+)\)', model_name)
                    if match:
                        p, d, q = map(int, match.groups())
                        model_order = (p, d, q)
                    else:
                        model_order = (2, 0, 1)  # Default for auto-ARIMA
                else:
                    model_order = (2, 0, 0)  # Default fallback
            
            print(f"Model: {model_result['model_name']}, Order: {model_order}")
            
            # Validate model order before proceeding
            if model_order is None or not isinstance(model_order, tuple) or len(model_order) != 3:
                print(f"❌ Could not determine valid model order for {model_result['model_name']}")
                continue
            
            holdout_result = evaluate_forecast_accuracy_holdout(
                data, model_order, 
                holdout_indices=holdout_indices,
                model_name=model_result['model_name']
            )
            
            if holdout_result is not None:
                holdout_results[model_key] = holdout_result
                accuracy_comparison[model_result['model_name']] = holdout_result['accuracy_metrics']
                print(f"✅ Successfully evaluated {model_result['model_name']}")
            else:
                print(f"❌ Holdout evaluation failed for {model_result['model_name']}")
                
        except Exception as e:
            print(f"❌ Could not evaluate {model_result['model_name']}: {e}")
            continue
    
    # Create comparison table
    if accuracy_comparison:
        print("=== FORECAST ACCURACY COMPARISON ===")
        print("=" * 60)
        
        comparison_df = pd.DataFrame(accuracy_comparison).T
        comparison_df = comparison_df.round(4)
        print(comparison_df)
        print("=" * 60)
        
        # Identify best model for each metric
        print("\nBest model by metric:")
        print("-" * 30)
        for metric in comparison_df.columns:
            if metric in ['RMSE', 'MAE', 'MAPE']:  # Lower is better
                best_model = comparison_df[metric].idxmin()
                best_value = comparison_df.loc[best_model, metric]
                print(f"{metric:>6}: {best_model} ({best_value:.4f})")
        print()
        
        return holdout_results, comparison_df
    
    else:
        print("No successful holdout evaluations")
        return None, None

def analyze_cycles(model, model_name):
    """
    Analyze cyclical components via roots of characteristic polynomials
    """
    print(f"\n=== CYCLE ANALYSIS FOR {model_name.upper()} ===")
    
    try:
        # Extract AR coefficients
        ar_params = []
        for i, param_name in enumerate(model.params.index):
            if 'ar.L' in param_name:
                ar_params.append(model.params.iloc[i])
        
        if not ar_params:
            print("No AR parameters found for cycle analysis")
            return None
        
        # Create characteristic polynomial: 1 - φ₁z - φ₂z² - ... = 0
        # Convert to polynomial form for numpy: [1, -φ₁, -φ₂, ...]
        char_poly = np.array([1] + [-coef for coef in ar_params])
        
        # Find roots
        roots = np.roots(char_poly)
        
        # Calculate moduli
        moduli = np.abs(roots)
        
        # Identify complex roots
        is_complex = np.abs(np.imag(roots)) > 1e-10
        complex_roots = roots[is_complex]
        
        print(f"Number of roots: {len(roots)}")
        print(f"All roots stationary (|root| > 1): {np.all(moduli > 1)}")
        
        if len(complex_roots) > 0:
            # Process complex conjugate pairs
            n_pairs = len(complex_roots) // 2
            print(f"Number of complex conjugate pairs: {n_pairs}")
            
            cycle_lengths = []
            for i in range(n_pairs):
                root = complex_roots[i]
                modulus = abs(root)
                real_part = np.real(root)
                
                # Calculate cycle length: 2π/arccos(Re(z)/|z|)
                if modulus > 0:
                    cycle_length = 2 * np.pi / np.arccos(real_part / modulus)
                    cycle_lengths.append(cycle_length)
                    
                    print(f"\nPair {i+1}:")
                    print(f"  Root: {root:.4f}")
                    print(f"  Modulus: {modulus:.4f}")
                    print(f"  Cycle length: {cycle_length:.2f} periods")
            
            return cycle_lengths
        else:
            print("No complex roots found - no cyclical components")
            return []
            
    except Exception as e:
        print(f"Error in cycle analysis: {e}")
        return None

# ===============================================================================
# CASE STUDY 2: UNEMPLOYMENT RATES
# ===============================================================================

def load_unemployment_data():
    """
    Load and preprocess unemployment rate data from FRED
    """
    print("\n=== CASE STUDY 2: UNEMPLOYMENT RATES ===\n")
    print("Data Acquisition - Retrieve unemployment rate data from FRED")
    
    if not FRED_AVAILABLE or web is None:
        print("✗ pandas_datareader not available.")
        return None, None
    
    try:
        # Get unemployment rate data
        print("Downloading US Unemployment Rate (UNRATE)")
        start_date = "1948-01-01"
        end_date = "2020-08-01"
        
        unemployment_data = web.DataReader('UNRATE', 'fred', start_date, end_date)['UNRATE'].dropna()
        print(f"✓ Data downloaded successfully: {len(unemployment_data)} observations")
        print(f"Date range: {unemployment_data.index[0]} to {unemployment_data.index[-1]}\n")
        
        # Calculate first differences
        unemployment_changes = unemployment_data.diff().dropna()
        
        return unemployment_data, unemployment_changes
        
    except Exception as e:
        print(f"✗ Error downloading unemployment data: {e}")
        return None, None

def analyze_unemployment_models(unemployment_data, unemployment_changes):
    """
    Fit and evaluate various models for unemployment data
    """
    print("=== UNEMPLOYMENT MODEL ESTIMATION ===\n")
    
    model_results = {}
    
    # Model 1: AR(4) with constraints (based on PACF analysis)
    print("Fitting AR(4) model...")
    try:
        ar4_model = ARIMA(unemployment_changes, order=(4, 0, 0)).fit()
        model_results['AR4'] = evaluate_arima_model(
            ar4_model, "AR(4)", unemployment_changes
        )
    except Exception as e:
        print(f"Could not fit AR(4) model: {e}")
        model_results['AR4'] = None
    
    # Model 2: MA(1)
    print("Fitting MA(1) model...")
    try:
        ma1_model = ARIMA(unemployment_changes, order=(0, 0, 1)).fit()
        model_results['MA1'] = evaluate_arima_model(
            ma1_model, "MA(1)", unemployment_changes
        )
    except Exception as e:
        print(f"Could not fit MA(1) model: {e}")
        model_results['MA1'] = None
    
    # Model 3: ARMA(1,1)
    print("Fitting ARMA(1,1) model...")
    try:
        arma11_model = ARIMA(unemployment_changes, order=(1, 0, 1)).fit()
        model_results['ARMA11'] = evaluate_arima_model(
            arma11_model, "ARMA(1,1)", unemployment_changes
        )
    except Exception as e:
        print(f"Could not fit ARMA(1,1) model: {e}")
        model_results['ARMA11'] = None
    
    # Compare models
    print("=== UNEMPLOYMENT MODEL COMPARISON ===\n")
    comparison_data = []
    for key, result in model_results.items():
        if result is not None:
            comparison_data.append({
                'Model': result['model_name'],
                'AIC': result['aic'],
                'BIC': result['bic'],
                'Residual_Autocorr': 'No' if result['adj_p_value'] > 0.05 else 'Yes'
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('AIC')
        
        print("Unemployment Model Comparison:")
        print("=" * 60)
        print(comparison_df.to_string(index=False))
        print("=" * 60)
        print()
        
        return model_results, comparison_df
    else:
        return model_results, None

# ===============================================================================
# MAIN EXECUTION
# ===============================================================================

def main():
    """
    Execute the complete ARIMA modeling tutorial
    """
    print("Starting comprehensive ARIMA modeling tutorial...\n")
    
    # Case Study 1: Drilling Activity
    drilling_data, drilling_changes = load_drilling_data()
    
    if drilling_data is not None:
        # Visualize data
        visualize_drilling_data(drilling_data, drilling_changes)
        
        # Fit models
        model_results = fit_drilling_models(drilling_changes)
        
        # Compare models
        best_model, comparison_df = compare_drilling_models(model_results)
        
        if best_model is not None:
            # Generate forecasts
            forecasts, forecast_result = generate_forecasts_with_ci(
                best_model['model'], h=50, 
                plot_title=f"Forecasts from {best_model['model_name']}"
            )
            
            if forecasts is not None:
                print("First 10 forecast periods:")
                print(forecasts.head(10))
                print()
            
            # Holdout sample evaluation and model comparison
            print("=== HOLDOUT SAMPLE EVALUATION ===\n")
            print("Evaluating models using holdout sample for forecast accuracy...")
            
            # Define holdout period (last 50 observations)
            n_obs = len(drilling_changes)
            holdout_indices = list(range(n_obs-50, n_obs))
            print(f"Holdout indices: {holdout_indices[0]} to {holdout_indices[-1]}\n")
            
            # Compare forecast accuracy across models
            holdout_results, accuracy_df = compare_model_forecast_accuracy(
                model_results, drilling_changes, holdout_indices
            )
            
            # Cycle analysis
            cycle_lengths = analyze_cycles(best_model['model'], best_model['model_name'])
    
    # Case Study 2: Unemployment
    unemployment_data, unemployment_changes = load_unemployment_data()
    
    if unemployment_data is not None:
        # Analyze unemployment models
        unemp_results, unemp_comparison = analyze_unemployment_models(
            unemployment_data, unemployment_changes
        )
        
        # Find best unemployment model
        if unemp_comparison is not None:
            best_unemp_name = unemp_comparison.iloc[0]['Model']
            best_unemp_model = None
            for result in unemp_results.values():
                if result is not None and result['model_name'] == best_unemp_name:
                    best_unemp_model = result
                    break
            
            if best_unemp_model is not None:
                # Cycle analysis for unemployment
                unemp_cycles = analyze_cycles(best_unemp_model['model'], best_unemp_name)
                
                # Holdout evaluation for unemployment
                print("\n=== UNEMPLOYMENT HOLDOUT EVALUATION ===\n")
                unemp_n_obs = len(unemployment_changes)
                unemp_holdout_indices = list(range(unemp_n_obs-12, unemp_n_obs))  # Last 12 months
                
                # Evaluate best unemployment model
                if 'AR(4)' in best_unemp_name:
                    unemp_holdout_result = evaluate_forecast_accuracy_holdout(
                        unemployment_changes, (4, 0, 0), 
                        holdout_indices=unemp_holdout_indices,
                        model_name=best_unemp_name
                    )
                
                # Generate unemployment forecasts (ARIMA with integration)
                try:
                    # Fit ARIMA(4,1,0) for levels
                    print("Generating integrated forecasts for unemployment levels...")
                    arima_model = ARIMA(unemployment_data, order=(4, 1, 0)).fit()
                    unemp_forecasts, _ = generate_forecasts_with_ci(
                        arima_model, h=48, 
                        plot_title="Unemployment Rate Forecasts (48 months)"
                    )
                    
                    # Evaluate ARIMA model on levels data
                    levels_holdout_indices = list(range(len(unemployment_data)-12, len(unemployment_data)))
                    levels_holdout_result = evaluate_forecast_accuracy_holdout(
                        unemployment_data, (4, 1, 0),
                        holdout_indices=levels_holdout_indices,
                        model_name="ARIMA(4,1,0) - Levels"
                    )
                    
                except Exception as e:
                    print(f"Could not generate integrated forecasts: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY OF FINDINGS")
    print("=" * 60)
    
    if drilling_data is not None and best_model is not None:
        print(f"1. For drilling activity:")
        print(f"   - Best model: {best_model['model_name']}")
        if cycle_lengths:
            print(f"   - Cycles of approx. {np.mean(cycle_lengths):.1f} periods")
        print(f"   - AIC: {best_model['aic']:.2f}")
    
    if unemployment_data is not None and 'best_unemp_model' in locals() and best_unemp_model is not None:
        print(f"\n2. For unemployment:")
        print(f"   - Best model: {best_unemp_model['model_name']}")
        if 'unemp_cycles' in locals() and unemp_cycles:
            print(f"   - Cycle of approx. {unemp_cycles[0]:.1f} periods")
        print(f"   - AIC: {best_unemp_model['aic']:.2f}")
    
    print("-" * 60)
    print("\n=== TUTORIAL COMPLETE ===")

if __name__ == "__main__":
    main()