# ===============================================================================
# AR AND MA SIMULATION EXAMPLES: UNDERSTANDING TIME SERIES BEHAVIOR
# ===============================================================================
#
# LEARNING OBJECTIVES:
# 1. Simulate AR and MA processes with known parameters
# 2. Distinguish between real and complex characteristic polynomial roots
# 3. Interpret cyclical behavior from complex roots
# 4. Compare forecasting properties of AR, MA, and White Noise processes
# 5. Understand how different model structures affect time series patterns
#
# THEORETICAL BACKGROUND:
# - AR processes: Current value depends on past values (persistence)
# - MA processes: Current value depends on past error terms (short memory)
# - Complex roots: Create cyclical/seasonal patterns in time series
# - Real roots: Create exponential decay or explosive behavior
#
# PRACTICAL IMPORTANCE:
# Understanding these simulation exercises helps in:
# - Model identification from real data
# - Forecast interpretation
# - Economic interpretation of estimated models
# ===============================================================================

# Import required packages with compatibility handling
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Robust imports for statsmodels (handle different versions)
try:
    from statsmodels.tsa.arima_process import ArmaProcess, arma_generate_sample
except ImportError:
    try:
        from statsmodels.tsa.arima_process import ArmaProcess
        from statsmodels.tsa.arima_process import arma_generate_sample
    except ImportError:
        print("Warning: Could not import ArmaProcess. Some functionality may be limited.")
        ArmaProcess = None
        arma_generate_sample = None

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    try:
        from statsmodels.tsa.arima_model import ARIMA
    except ImportError:
        print("Warning: Could not import ARIMA model.")
        ARIMA = None

try:
    from statsmodels.tsa.ar_model import AutoReg, ar_select_order
except ImportError:
    try:
        from statsmodels.tsa.ar_model import AutoReg
        from statsmodels.tsa.ar_model import ar_select_order
    except ImportError:
        print("Warning: Could not import AR model functions.")
        AutoReg = None
        ar_select_order = None

try:
    from statsmodels.tsa.stattools import acf, pacf
except ImportError:
    print("Warning: Could not import ACF/PACF functions.")
    acf = None
    pacf = None

# Import Ljung-Box test
from statsmodels.stats.diagnostic import acorr_ljungbox

# Set plotting style with fallbacks
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        try:
            import seaborn as sns
            sns.set_style("whitegrid")
            print("Using seaborn default style")
        except ImportError:
            print("Using matplotlib default style")

np.random.seed(42)

# Compatibility check function
def check_required_functions():
    """Check if required functions are available and provide alternatives"""
    missing_functions = []
    
    if ArmaProcess is None:
        missing_functions.append("ArmaProcess (for ARMA simulation)")
    if ARIMA is None:
        missing_functions.append("ARIMA (for model fitting)")  
    if acf is None:
        missing_functions.append("ACF (for autocorrelation analysis)")
    if pacf is None:
        missing_functions.append("PACF (for partial autocorrelation analysis)")
    
    if missing_functions:
        print("WARNING: Some required functions are not available:")
        for func in missing_functions:
            print(f"  - {func}")
        print("\nTo fix this, try:")
        print("  pip install --upgrade statsmodels")
        print("  conda update statsmodels")
        print("\nSome functionality may be limited.\n")
        return False
    return True

print("=== AR AND MA SIMULATION TUTORIAL ===")
print("Understanding Time Series Behavior Through Simulation\n")

# Check compatibility
compatibility_ok = check_required_functions()

# ===============================================================================
# PART 1: AR(2) PROCESSES WITH DIFFERENT ROOT STRUCTURES
# ===============================================================================

def simulate_ar2_processes():
    """
    Simulate two AR(2) processes to demonstrate how characteristic polynomial
    roots affect the time series behavior:
    1. AR(2) with real roots → exponential decay/growth patterns
    2. AR(2) with complex roots → cyclical/oscillatory patterns
    """
    print("=== PART 1: AR(2) SIMULATION EXAMPLES ===\n")
    
    # Check if required functions are available
    if ArmaProcess is None:
        print("Error: ArmaProcess not available. Cannot simulate ARMA processes.")
        print("Please install/update statsmodels: pip install --upgrade statsmodels")
        return None, None
    
    n_obs = 1000
    
    # ------------------- AR(2) PROCESS WITH REAL ROOTS -------------------
    print("--- AR(2) Process with Real Roots ---")
    print("Model: y_t = 1.3*y_{t-1} - 0.4*y_{t-2} + ε_t")
    print("Parameters: φ₁ = 1.3, φ₂ = -0.4")
    
    # AR coefficients for real roots
    ar_real = [1, -1.3, 0.4]  # 1 - 1.3L + 0.4L² = 0
    ma_real = [1]
    
    # Create ARMA process object
    ar2_real_process = ArmaProcess(ar_real, ma_real)
    
    # Generate time series
    ar2_real_roots = ar2_real_process.generate_sample(n_obs, scale=1.0)
    
    print(f"Simulated {len(ar2_real_roots)} observations")
    print("Expected behavior: Exponential decay pattern in ACF\n")
    
    # ------------------- AR(2) PROCESS WITH COMPLEX ROOTS -------------------
    print("--- AR(2) Process with Complex Roots ---")
    print("Model: y_t = 0.8*y_{t-1} - 0.7*y_{t-2} + ε_t")
    print("Parameters: φ₁ = 0.8, φ₂ = -0.7")
    
    # AR coefficients for complex roots
    ar_complex = [1, -0.8, 0.7]  # 1 - 0.8L + 0.7L² = 0
    ma_complex = [1]
    
    # Create ARMA process object
    ar2_complex_process = ArmaProcess(ar_complex, ma_complex)
    
    # Generate time series
    ar2_complex_roots = ar2_complex_process.generate_sample(n_obs, scale=1.0)
    
    print("Expected behavior: Damped oscillating pattern in ACF\n")
    
    print("COMPARISON OF ACF PATTERNS:")
    print("- Real roots: Smooth exponential decay")
    print("- Complex roots: Oscillating (sine/cosine-like) decay\n")
    
    # ------------------- VISUAL COMPARISON -------------------
    print("--- Visual Comparison of Time Series Patterns ---")
    
    # Plot both processes
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Time series plots
    axes[0, 0].plot(ar2_real_roots, color='blue', alpha=0.8)
    axes[0, 0].set_title('AR(2) with Real Roots: Smooth Exponential Behavior')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(ar2_complex_roots, color='red', alpha=0.8)
    axes[0, 1].set_title('AR(2) with Complex Roots: Cyclical/Oscillatory Behavior')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].grid(True)
    
    # ACF plots (with safety checks)
    if acf is not None:
        try:
            acf_real = acf(ar2_real_roots, nlags=30, fft=True)
            axes[1, 0].plot(range(len(acf_real)), acf_real, 'o-', color='blue', alpha=0.8)
            axes[1, 0].axhline(y=0, color='k', linestyle='-')
            axes[1, 0].axhline(y=1.96/np.sqrt(len(ar2_real_roots)), color='r', linestyle='--')
            axes[1, 0].axhline(y=-1.96/np.sqrt(len(ar2_real_roots)), color='r', linestyle='--')
            axes[1, 0].set_title('ACF: AR(2) with Real Roots')
            axes[1, 0].set_xlabel('Lag')
            axes[1, 0].set_ylabel('ACF')
            axes[1, 0].grid(True)
            
            acf_complex = acf(ar2_complex_roots, nlags=30, fft=True)
            axes[1, 1].plot(range(len(acf_complex)), acf_complex, 'o-', color='red', alpha=0.8)
            axes[1, 1].axhline(y=0, color='k', linestyle='-')
            axes[1, 1].axhline(y=1.96/np.sqrt(len(ar2_complex_roots)), color='r', linestyle='--')
            axes[1, 1].axhline(y=-1.96/np.sqrt(len(ar2_complex_roots)), color='r', linestyle='--')
            axes[1, 1].set_title('ACF: AR(2) with Complex Roots')
            axes[1, 1].set_xlabel('Lag')
            axes[1, 1].set_ylabel('ACF')
            axes[1, 1].grid(True)
        except Exception as e:
            axes[1, 0].text(0.5, 0.5, f'ACF calculation failed:\n{str(e)}', 
                           transform=axes[1, 0].transAxes, ha='center', va='center')
            axes[1, 1].text(0.5, 0.5, f'ACF calculation failed:\n{str(e)}', 
                           transform=axes[1, 1].transAxes, ha='center', va='center')
    else:
        axes[1, 0].text(0.5, 0.5, 'ACF function not available\nPlease update statsmodels', 
                       transform=axes[1, 0].transAxes, ha='center', va='center')
        axes[1, 1].text(0.5, 0.5, 'ACF function not available\nPlease update statsmodels', 
                       transform=axes[1, 1].transAxes, ha='center', va='center')
    
    plt.tight_layout()
    plt.show()
    
    print("VISUAL INTERPRETATION:")
    print("- Real roots: Smooth transitions between values")
    print("- Complex roots: Regular oscillatory patterns (pseudo-cycles)\n")
    
    return ar2_real_roots, ar2_complex_roots

# ===============================================================================
# PART 2: CHARACTERISTIC POLYNOMIAL ROOTS ANALYSIS
# ===============================================================================

def analyze_characteristic_roots():
    """
    Analyze characteristic polynomial roots to understand dynamic properties
    """
    print("=== PART 2: CHARACTERISTIC POLYNOMIAL ROOTS ANALYSIS ===\n")
    
    print("The characteristic polynomial helps us understand the dynamic properties")
    print("of AR processes. For an AR(2) process: y_t = φ₁y_{t-1} + φ₂y_{t-2} + ε_t")
    print("The characteristic polynomial is: 1 - φ₁z - φ₂z² = 0\n")
    
    print("Root interpretation:")
    print("- Real roots: Exponential behavior (growth/decay)")
    print("- Complex roots: Cyclical behavior (pseudo-periodic oscillations)")
    print("- Modulus > 1: Stationary process")
    print("- Modulus = 1: Unit root non-stationary process\n")
    print("- Modulus < 1: Explosive process\n")
    
    # ------------------- REAL ROOTS ANALYSIS -------------------
    print("--- Analysis of AR(2) with Real Roots ---")
    print("Characteristic polynomial: 1 - 1.3z + 0.4z² = 0")
    
    # Coefficients for characteristic polynomial (note the sign convention)
    char_poly_real = [0.4, -1.3, 1]  # 0.4z² - 1.3z + 1 = 0
    roots_real = np.roots(char_poly_real)
    
    print("Characteristic roots:")
    print(f"Root 1: {roots_real[0]:.4f}")
    print(f"Root 2: {roots_real[1]:.4f}")
    print("Both roots are real numbers")
    print(f"Moduli: {np.abs(roots_real)}")
    
    if np.all(np.abs(roots_real) > 1):
        print("CONCLUSION: Process is stationary (all roots outside unit circle)\n")
    else:
        print("WARNING: Process may be non-stationary\n")
    
    # ------------------- COMPLEX ROOTS ANALYSIS -------------------
    print("--- Analysis of AR(2) with Complex Roots ---")
    print("Characteristic polynomial: 1 - 0.8z + 0.7z² = 0")
    
    # Coefficients for characteristic polynomial
    char_poly_complex = [0.7, -0.8, 1]  # 0.7z² - 0.8z + 1 = 0
    roots_complex = np.roots(char_poly_complex)
    
    print("Characteristic roots:")
    print(f"Root 1: {roots_complex[0]:.4f}")
    print(f"Root 2: {roots_complex[1]:.4f}")
    
    # Extract real and imaginary parts for cycle calculation
    real_part = np.real(roots_complex[0])    # Real part (same for both roots)
    imag_part = np.imag(roots_complex[0])    # Imaginary part
    modulus = np.abs(roots_complex[0])       # Modulus (same for both roots)
    
    print(f"\nDetailed analysis of complex roots:")
    print(f"Real part (a): {real_part:.6f}")
    print(f"Imaginary part (b): {np.abs(imag_part):.6f}")
    print(f"Modulus: {modulus:.6f}")
    
    # ------------------- CYCLE LENGTH CALCULATION -------------------
    print("\n--- Cycle Length Calculation ---")
    print("For complex roots, we can calculate the average cycle length:")
    print("Formula: k = 2π / arccos(real_part / modulus)\n")
    
    # Calculate cycle length using the formula
    cycle_length = 2 * np.pi / np.arccos(real_part / modulus)
    print(f"Calculated cycle length: {cycle_length:.2f} periods")
    
    # Alternative calculation (verification)
    print("\nVerification calculation:")
    modulus_check = np.sqrt(real_part**2 + imag_part**2)
    cycle_length_check = 2 * np.pi / np.arccos(real_part / modulus_check)
    print(f"Alternative cycle length: {cycle_length_check:.2f} periods")
    
    print("\nINTERPRETATION:")
    print("- The process exhibits pseudo-cyclical behavior")
    print(f"- Average cycle length is approximately {cycle_length:.1f} periods")
    print("- This creates the oscillating pattern we see in the ACF")
    print("- No need for sine/cosine functions - AR coefficients capture cyclical dynamics\n")
    
    # ------------------- STATIONARITY CHECK -------------------
    if modulus > 1:
        print("STATIONARITY: Process is stationary (modulus > 1)")
    else:
        print("WARNING: Process is non-stationary (modulus <= 1)")
    
    return cycle_length

# ===============================================================================
# PART 3: MODEL IDENTIFICATION AND ESTIMATION FROM SIMULATED DATA
# ===============================================================================

def model_identification_estimation(ar2_real_roots, ar2_complex_roots):
    """
    Demonstrate how to recover true model structure using identification techniques
    """
    print("\n=== PART 3: MODEL IDENTIFICATION FROM SIMULATED DATA ===\n")
    
    print("In practice, we observe time series data but don't know the true underlying")
    print("parameters. This section demonstrates how to estimate the model structure")
    print("using standard model selection techniques.\n")
    
    # ------------------- MODEL SELECTION FOR REAL ROOTS PROCESS -------------------
    print("--- Model Selection: AR(2) with Real Roots ---")
    print("True parameters: φ₁ = 1.3, φ₂ = -0.4")
    print("Attempting to recover these parameters from the data...\n")
    
    # Use automatic AR model selection
    try:
        # Fit AR model with automatic order selection
        ar_real_model = ar_select_order(ar2_real_roots, maxlag=10, ic='aic')
        selected_order_real = ar_real_model.ar_lags[-1]
        
        # Fit the selected model
        ar_fitted_real = AutoReg(ar2_real_roots, lags=selected_order_real).fit()
        
        print("Automatic model selection results:")
        print(f"Selected AR order: {selected_order_real}")
        print(f"Estimated coefficients: {ar_fitted_real.params[1:]}")  # Exclude constant
        print("Compare to true values: φ₁ = 1.3, φ₂ = -0.4\n")
        
        # Plot PACF
        pacf_real = pacf(ar2_real_roots, nlags=10)
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(range(len(pacf_real)), pacf_real, 'o-')
        plt.axhline(y=0, color='k', linestyle='-')
        plt.axhline(y=1.96/np.sqrt(len(ar2_real_roots)), color='r', linestyle='--')
        plt.axhline(y=-1.96/np.sqrt(len(ar2_real_roots)), color='r', linestyle='--')
        plt.title('PACF: AR(2) with Real Roots')
        plt.xlabel('Lag')
        plt.ylabel('PACF')
        plt.grid(True)
        
        # Diagnostic tests
        print("Diagnostic Tests:")
        mean_test = stats.ttest_1samp(ar2_real_roots, 0)
        print(f"Mean test (H₀: μ = 0): Sample mean = {np.mean(ar2_real_roots):.4f}, p-value = {mean_test.pvalue:.4f}")
        
        ljung_test = acorr_ljungbox(ar2_real_roots, lags=10, return_df=True)
        print(f"Ljung-Box test (H₀: no autocorrelation): min p-value = {ljung_test['lb_pvalue'].min():.4f}")
        if ljung_test['lb_pvalue'].min() < 0.05:
            print("CONCLUSION: Significant autocorrelation detected (as expected for AR process)\n")
        
    except Exception as e:
        print(f"Error in model identification: {e}\n")
    
    # ------------------- MODEL IDENTIFICATION FOR COMPLEX ROOTS PROCESS -------------------
    print("--- Model Identification: AR(2) with Complex Roots ---")
    print("True parameters: φ₁ = 0.8, φ₂ = -0.7")
    
    try:
        # Use automatic AR model selection
        ar_complex_model = ar_select_order(ar2_complex_roots, maxlag=10, ic='aic')
        selected_order_complex = ar_complex_model.ar_lags[-1]
        
        # Fit the selected model
        ar_fitted_complex = AutoReg(ar2_complex_roots, lags=selected_order_complex).fit()
        
        print("Automatic model selection results:")
        print(f"Selected AR order: {selected_order_complex}")
        print(f"Estimated coefficients: {ar_fitted_complex.params[1:]}")  # Exclude constant
        print("Compare to true values: φ₁ = 0.8, φ₂ = -0.7\n")
        
        # Plot PACF
        pacf_complex = pacf(ar2_complex_roots, nlags=10)
        plt.subplot(1, 2, 2)
        plt.plot(range(len(pacf_complex)), pacf_complex, 'o-')
        plt.axhline(y=0, color='k', linestyle='-')
        plt.axhline(y=1.96/np.sqrt(len(ar2_complex_roots)), color='r', linestyle='--')
        plt.axhline(y=-1.96/np.sqrt(len(ar2_complex_roots)), color='r', linestyle='--')
        plt.title('PACF: AR(2) with Complex Roots')
        plt.xlabel('Lag')
        plt.ylabel('PACF')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Diagnostic tests
        mean_test_2 = stats.ttest_1samp(ar2_complex_roots, 0)
        print("Diagnostic Tests:")
        print(f"Mean test (H₀: μ = 0): Sample mean = {np.mean(ar2_complex_roots):.4f}, p-value = {mean_test_2.pvalue:.4f}")
        
        ljung_test_2 = acorr_ljungbox(ar2_complex_roots, lags=10, return_df=True)
        print(f"Ljung-Box test (H₀: no autocorrelation): min p-value = {ljung_test_2['lb_pvalue'].min():.4f}\n")
        
    except Exception as e:
        print(f"Error in model identification: {e}\n")
    
    print("INTERPRETATION OF IDENTIFICATION RESULTS:")
    print("- Both processes should be correctly identified as AR(2)")
    print("- Estimated coefficients should be close to true values")
    print("- PACF should show significant values at lags 1 and 2, then cut off")
    print("- Large samples provide more accurate parameter estimates\n")

# ===============================================================================
# PART 4: FORECASTING BEHAVIOR
# ===============================================================================

def ar2_forecasting_comparison(ar2_real, ar2_complex):
    """
    Compare forecasting behavior between AR(2) with real vs complex roots
    """
    print("=== PART 4: AR(2) FORECASTING COMPARISON ===\n")
    
    print("Demonstrates how different root structures affect forecasting behavior:")
    print("- Real roots: Smooth convergence to long-run mean")
    print("- Complex roots: Cyclical approach to long-run mean\n")
    
    try:
        # Fit AR models to both series
        print("--- Forecasting AR(2) with Real Roots ---")
        real_model = AutoReg(ar2_real, lags=2).fit()
        print("Estimated coefficients:", real_model.params[1:])
        print("True coefficients: [1.3, -0.4]\n")
        
        # Generate forecasts for real roots
        real_forecast = real_model.forecast(steps=100)
        real_forecast_std = np.sqrt(real_model.sigma2) * np.ones(len(real_forecast))
        z_score = 1.96
        real_ci_lower = real_forecast - z_score * real_forecast_std
        real_ci_upper = real_forecast + z_score * real_forecast_std
        
        print("--- Forecasting AR(2) with Complex Roots ---") 
        complex_model = AutoReg(ar2_complex, lags=2).fit()
        print("Estimated coefficients:", complex_model.params[1:])
        print("True coefficients: [0.8, -0.7]\n")
        
        # Generate forecasts for complex roots
        complex_forecast = complex_model.forecast(steps=100)
        complex_forecast_std = np.sqrt(complex_model.sigma2) * np.ones(len(complex_forecast))
        complex_ci_lower = complex_forecast - z_score * complex_forecast_std
        complex_ci_upper = complex_forecast + z_score * complex_forecast_std
        
        # Plot forecasting comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('AR(2) Forecasting Comparison: Real vs Complex Roots', fontsize=16)
        
        # Real roots - full forecast
        axes[0, 0].plot(range(len(ar2_real)), ar2_real, label='Observed', color='blue', alpha=0.7)
        forecast_index = range(len(ar2_real), len(ar2_real) + len(real_forecast))
        axes[0, 0].plot(forecast_index, real_forecast, label='Forecast', color='red', linewidth=2)
        axes[0, 0].fill_between(forecast_index, real_ci_lower, real_ci_upper, 
                               color='red', alpha=0.2, label='95% CI')
        axes[0, 0].axvline(x=len(ar2_real), color='k', linestyle='--', alpha=0.5)
        axes[0, 0].set_title('AR(2) Real Roots: Smooth Exponential Convergence')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Real roots - zoom on forecast
        recent_data = ar2_real[-50:]
        axes[0, 1].plot(range(50), recent_data, label='Recent Observed', color='blue', alpha=0.7)
        forecast_index_zoom = range(50, 50 + 50)  # First 50 forecast steps
        axes[0, 1].plot(forecast_index_zoom, real_forecast[:50], label='Forecast', color='red', linewidth=2)
        axes[0, 1].fill_between(forecast_index_zoom, real_ci_lower[:50], real_ci_upper[:50],
                               color='red', alpha=0.2, label='95% CI')
        axes[0, 1].axvline(x=50, color='k', linestyle='--', alpha=0.5)
        axes[0, 1].set_title('Real Roots: Forecast Detail')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Complex roots - full forecast  
        axes[1, 0].plot(range(len(ar2_complex)), ar2_complex, label='Observed', color='blue', alpha=0.7)
        forecast_index = range(len(ar2_complex), len(ar2_complex) + len(complex_forecast))
        axes[1, 0].plot(forecast_index, complex_forecast, label='Forecast', color='red', linewidth=2)
        axes[1, 0].fill_between(forecast_index, complex_ci_lower, complex_ci_upper,
                               color='red', alpha=0.2, label='95% CI')
        axes[1, 0].axvline(x=len(ar2_complex), color='k', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('AR(2) Complex Roots: Cyclical Convergence')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Complex roots - zoom on forecast
        recent_data_complex = ar2_complex[-50:]
        axes[1, 1].plot(range(50), recent_data_complex, label='Recent Observed', color='blue', alpha=0.7)
        axes[1, 1].plot(forecast_index_zoom, complex_forecast[:50], label='Forecast', color='red', linewidth=2)
        axes[1, 1].fill_between(forecast_index_zoom, complex_ci_lower[:50], complex_ci_upper[:50],
                               color='red', alpha=0.2, label='95% CI')
        axes[1, 1].axvline(x=50, color='k', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Complex Roots: Cyclical Forecast Detail')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        print("COMPARISON OF FORECASTING BEHAVIOR:")
        print("- Complex roots: Cyclical/oscillatory convergence to mean")
        print("- Real roots: Smooth exponential convergence to mean") 
        print("- Both converge to long-run mean in the limit")
        print("- Complex roots preserve cyclical behavior in short-term forecasts\n")
        
    except Exception as e:
        print(f"AR(2) forecasting comparison encountered an issue: {e}")
        print("This is common with certain model specifications.\n")

def higher_order_comparison():
    """
    Compare AR(3) and MA(3) processes to demonstrate higher-order dynamics
    """
    print("=== PART 6: HIGHER-ORDER PROCESSES ===\n")
    
    print("--- Higher-Order Process Comparison ---")
    print("Simulating AR(3) and MA(3) with complex dynamics\n")
    
    np.random.seed(999)
    n_obs = 2000
    
    try:
        # ------------------- AR(3) PROCESS -------------------
        print("--- AR(3) Process Analysis ---")
        
        # Simulate AR(3): φ₁=0.8, φ₂=-0.3, φ₃=0.3
        ar3_coef = [1, -0.8, 0.3, -0.3]  # Note: ArmaProcess uses negative signs
        ma3_coef = [1]
        ar3_process = ArmaProcess(ar3_coef, ma3_coef)
        ar3_series = ar3_process.generate_sample(n_obs)
        
        print("True AR(3) coefficients: [0.8, -0.3, 0.3]")
        
        # Fit AR(3) model
        ar3_model = AutoReg(ar3_series, lags=3).fit()
        print("Estimated coefficients:", ar3_model.params[1:])
        
        # AR(3) forecasting with confidence intervals
        print("\nGenerating AR(3) forecasts...")
        ar3_forecast = ar3_model.forecast(steps=200)
        ar3_forecast_std = np.sqrt(ar3_model.sigma2) * np.ones(len(ar3_forecast))
        z_score = 1.96
        ar3_ci_lower = ar3_forecast - z_score * ar3_forecast_std
        ar3_ci_upper = ar3_forecast + z_score * ar3_forecast_std
        
        # ------------------- MA(3) PROCESS -------------------
        print("\n--- MA(3) Process Analysis ---")
        
        # Simulate MA(3): θ₁=0.8, θ₂=-0.3, θ₃=0.3
        ma3_ar_coef = [1]
        ma3_ma_coef = [1, 0.8, -0.3, 0.3]
        ma3_process = ArmaProcess(ma3_ar_coef, ma3_ma_coef)
        ma3_series = ma3_process.generate_sample(n_obs)
        
        print("True MA(3) coefficients: [0.8, -0.3, 0.3]")
        
        # Fit MA(3) model using ARIMA
        ma3_model = ARIMA(ma3_series, order=(0, 0, 3)).fit()
        print("Estimated coefficients:", ma3_model.params[:-1])  # Exclude sigma2
        
        # MA(3) forecasting with confidence intervals
        print("\nGenerating MA(3) forecasts...")
        ma3_forecast_result = ma3_model.get_forecast(steps=10)  # MA converges quickly
        ma3_forecast = ma3_forecast_result.predicted_mean
        ma3_forecast_ci = ma3_forecast_result.conf_int()
        
        # ------------------- VISUALIZATION -------------------
        
        # Plot processes and their forecasts
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Higher-Order Process Comparison: AR(3) vs MA(3)', fontsize=16)
        
        # AR(3) time series
        axes[0, 0].plot(ar3_series[-200:], alpha=0.8, color='blue')
        axes[0, 0].set_title('AR(3): Complex Long-Memory Dynamics')
        axes[0, 0].set_ylabel('AR(3) Value')
        axes[0, 0].grid(True)
        
        # MA(3) time series  
        axes[0, 1].plot(ma3_series[-200:], alpha=0.8, color='red')
        axes[0, 1].set_title('MA(3): Complex Short-Memory Dynamics') 
        axes[0, 1].set_ylabel('MA(3) Value')
        axes[0, 1].grid(True)
        
        # AR(3) forecast
        recent_ar3 = ar3_series[-100:]
        axes[1, 0].plot(range(100), recent_ar3, label='Observed', color='blue', alpha=0.7)
        forecast_index = range(100, 100 + 50)  # Show first 50 forecast steps
        axes[1, 0].plot(forecast_index, ar3_forecast[:50], label='Forecast', color='red', linewidth=2)
        axes[1, 0].fill_between(forecast_index, ar3_ci_lower[:50], ar3_ci_upper[:50],
                               color='red', alpha=0.2, label='95% CI')
        axes[1, 0].axvline(x=100, color='k', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('AR(3) Forecast: Gradual Complex Convergence')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # MA(3) forecast
        recent_ma3 = ma3_series[-50:]
        axes[1, 1].plot(range(50), recent_ma3, label='Observed', color='blue', alpha=0.7)
        forecast_index_ma = range(50, 50 + len(ma3_forecast))
        axes[1, 1].plot(forecast_index_ma, ma3_forecast, label='Forecast', color='red', 
                       linewidth=2, marker='o')
        
        # Handle MA confidence intervals
        if hasattr(ma3_forecast_ci, 'iloc'):
            ci_lower = ma3_forecast_ci.iloc[:, 0]
            ci_upper = ma3_forecast_ci.iloc[:, 1]
        else:
            ci_lower = ma3_forecast_ci[:, 0]
            ci_upper = ma3_forecast_ci[:, 1]
            
        axes[1, 1].fill_between(forecast_index_ma, ci_lower, ci_upper,
                               color='red', alpha=0.2, label='95% CI')
        axes[1, 1].axvline(x=50, color='k', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('MA(3) Forecast: Quick Convergence after Lag 3')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        print("HIGHER-ORDER PROCESS INSIGHTS:")
        print("- AR(3): Complex long-term dynamics with gradual forecast convergence") 
        print("- MA(3): Complex short-term dynamics but quick forecast convergence")
        print("- Higher-order models capture more sophisticated temporal patterns")
        print("- Parameter estimation becomes more challenging with higher orders")
        print("- Sample size requirements increase with model complexity\n")
        
    except Exception as e:
        print(f"Higher-order comparison encountered an issue: {e}")
        print("This is common with complex model specifications.\n")

def simple_process_forecasting_comparison():
    """
    Demonstrate forecasting behavior for different process types
    """
    print("=== PART 4: COMPARING AR, MA, AND WHITE NOISE PROCESSES ===\n")
    
    print("This section demonstrates fundamental differences between:")
    print("- AR(1): Autoregressive - persistence and mean reversion")
    print("- MA(1): Moving Average - short memory, larger variance")
    print("- White Noise: No memory, constant variance\n")
    
    n_obs = 200
    np.random.seed(123)
    
    # ------------------- SIMULATION -------------------
    print("--- Simulating Different Process Types ---")
    print("All processes have the same coefficient (0.8) for comparison\n")
    
    # AR(1) process: y_t = 0.8*y_{t-1} + ε_t
    ar_coef = [1, -0.8]
    ma_coef = [1]
    ar1_process = ArmaProcess(ar_coef, ma_coef)
    ar1_series = ar1_process.generate_sample(n_obs)
    
    # MA(1) process: y_t = ε_t + 0.8*ε_{t-1}
    ar_coef_ma = [1]
    ma_coef_ma = [1, 0.8]
    ma1_process = ArmaProcess(ar_coef_ma, ma_coef_ma)
    ma1_series = ma1_process.generate_sample(n_obs)
    
    # White Noise: y_t = ε_t
    wn_series = np.random.normal(0, 1, n_obs)
    
    print(f"Sample sizes: {n_obs} observations each")
    print("AR(1): y_t = 0.8*y_{t-1} + ε_t")
    print("MA(1): y_t = ε_t + 0.8*ε_{t-1}")
    print("White Noise: y_t = ε_t\n")
    
    # ------------------- VISUAL COMPARISON -------------------
    print("--- Visual Comparison of Time Series ---")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Time series plots
    axes[0, 0].plot(ar1_series, color='blue', alpha=0.8)
    axes[0, 0].set_title('AR(1): Persistent, Mean-Reverting')
    axes[0, 0].set_ylabel('AR(1) Value')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(ma1_series, color='red', alpha=0.8)
    axes[0, 1].set_title('MA(1): Short Memory, Higher Variance')
    axes[0, 1].set_ylabel('MA(1) Value')
    axes[0, 1].grid(True)
    
    axes[0, 2].plot(wn_series, color='green', alpha=0.8)
    axes[0, 2].set_title('White Noise: No Memory, Constant Variance')
    axes[0, 2].set_ylabel('WN Value')
    axes[0, 2].grid(True)
    
    # ACF plots
    acf_ar1 = acf(ar1_series, nlags=20, fft=True)
    axes[1, 0].plot(range(len(acf_ar1)), acf_ar1, 'o-', color='blue')
    axes[1, 0].axhline(y=0, color='k', linestyle='-')
    axes[1, 0].set_title('ACF: AR(1)')
    axes[1, 0].set_xlabel('Lag')
    axes[1, 0].set_ylabel('ACF')
    axes[1, 0].grid(True)
    
    acf_ma1 = acf(ma1_series, nlags=20, fft=True)
    axes[1, 1].plot(range(len(acf_ma1)), acf_ma1, 'o-', color='red')
    axes[1, 1].axhline(y=0, color='k', linestyle='-')
    axes[1, 1].set_title('ACF: MA(1)')
    axes[1, 1].set_xlabel('Lag')
    axes[1, 1].set_ylabel('ACF')
    axes[1, 1].grid(True)
    
    acf_wn = acf(wn_series, nlags=20, fft=True)
    axes[1, 2].plot(range(len(acf_wn)), acf_wn, 'o-', color='green')
    axes[1, 2].axhline(y=0, color='k', linestyle='-')
    axes[1, 2].set_title('ACF: White Noise')
    axes[1, 2].set_xlabel('Lag')
    axes[1, 2].set_ylabel('ACF')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("BEHAVIORAL DIFFERENCES:")
    print("- AR(1): Drifts away from mean, then drifts back (persistence)")
    print("- MA(1): Similar mean reversion to WN, but wider variance")
    print("- White Noise: Random fluctuations around mean")
    print("- MA(1) has more extreme values outside ±2 standard deviations\n")
    
    # ------------------- FORECASTING COMPARISON -------------------
    print("--- Forecasting Comparison ---")
    
    try:
        # Fit AR(1) model and generate forecasts with confidence intervals
        ar1_model = AutoReg(ar1_series, lags=1).fit()
        print(ar1_model.summary())
        # AutoReg uses forecast method, not get_forecast
        ar1_forecast = ar1_model.forecast(steps=50)
        # Calculate confidence intervals manually for AutoReg
        forecast_std = np.sqrt(ar1_model.sigma2) * np.ones(len(ar1_forecast))
        z_score = 1.96  # 95% confidence interval
        ar1_forecast_ci_lower = ar1_forecast - z_score * forecast_std
        ar1_forecast_ci_upper = ar1_forecast + z_score * forecast_std
        
        # Fit MA(1) model and generate forecasts with confidence intervals
        ma1_model = ARIMA(ma1_series, order=(0, 0, 1)).fit()
        print(ma1_model.summary())
        ma1_forecast_result = ma1_model.get_forecast(steps=5)
        ma1_forecast_obj = ma1_forecast_result.predicted_mean
        ma1_forecast_ci = ma1_forecast_result.conf_int()
        
        # Plot forecasting behavior
        plt.figure(figsize=(12, 8))
        
        # AR(1) forecasts with confidence intervals
        plt.subplot(2, 1, 1)
        plt.plot(range(len(ar1_series)), ar1_series, label='Observed', color='blue', alpha=0.8)
        forecast_index = range(len(ar1_series), len(ar1_series) + len(ar1_forecast))
        plt.plot(forecast_index, ar1_forecast, label='Forecast', color='red', linewidth=2)
        plt.fill_between(forecast_index, ar1_forecast_ci_lower, ar1_forecast_ci_upper, 
                        color='red', alpha=0.2, label='95% CI')
        plt.axvline(x=len(ar1_series), color='k', linestyle='--', alpha=0.5)
        plt.title('AR(1) Forecast: Gradual Convergence to Mean')
        plt.legend()
        plt.grid(True)
        
        # MA(1) forecasts with confidence intervals (shorter horizon)
        plt.subplot(2, 1, 2)
        plt.plot(range(len(ma1_series[-50:])), ma1_series[-50:], label='Observed', color='blue', alpha=0.8)
        forecast_index_ma = range(50, 50 + len(ma1_forecast_obj))
        plt.plot(forecast_index_ma, ma1_forecast_obj, label='Forecast', color='red', linewidth=2, marker='o')
        
        # Handle confidence intervals robustly (pandas DataFrame or numpy array)
        if hasattr(ma1_forecast_ci, 'iloc'):
            # pandas DataFrame
            ci_lower = ma1_forecast_ci.iloc[:, 0]
            ci_upper = ma1_forecast_ci.iloc[:, 1]
        else:
            # numpy array
            ci_lower = ma1_forecast_ci[:, 0]
            ci_upper = ma1_forecast_ci[:, 1]
        
        plt.fill_between(forecast_index_ma, ci_lower, ci_upper, 
                        color='red', alpha=0.2, label='95% CI')
        plt.axvline(x=50, color='k', linestyle='--', alpha=0.5)
        plt.title('MA(1) Forecast: Quick Convergence to Mean')
        plt.xlabel('Time')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        print("FORECASTING INSIGHTS:")
        print("- MA models: Forecast converges to mean quickly after lag q")
        print("- AR models: Gradual convergence with exponential decay")
        print("- Proper model specification improves short-term forecast precision")
        print("- Ignoring model structure increases forecast uncertainty\n")
        
    except Exception as e:
        print(f"Forecasting comparison encountered an issue: {e}")
        print("This is common with certain model specifications.\n")

# ===============================================================================
# SUMMARY AND KEY TAKEAWAYS
# ===============================================================================

def print_summary():
    """
    Print summary of key insights from the simulation tutorial
    """
    print("=== SUMMARY OF KEY INSIGHTS ===\n")
    
    print("1. ROOT STRUCTURE EFFECTS:")
    print("   - Real roots → Smooth exponential patterns")
    print("   - Complex roots → Cyclical/oscillatory patterns")
    print("   - Modulus > 1 → Stationary process\n")
    
    print("2. PROCESS TYPE CHARACTERISTICS:")
    print("   - AR processes: Long memory, gradual mean reversion")
    print("   - MA processes: Short memory, quick mean reversion")
    print("   - Higher variance in MA than AR with same coefficients\n")
    
    print("3. IDENTIFICATION PRINCIPLES:")
    print("   - ACF/PACF patterns help distinguish AR from MA")
    print("   - Automatic selection methods are helpful but not infallible")
    print("   - Larger samples improve parameter estimation accuracy\n")
    
    print("4. FORECASTING IMPLICATIONS:")
    print("   - AR forecasts show gradual convergence")
    print("   - MA forecasts converge quickly after lag q")
    print("   - Model structure affects forecast uncertainty")
    print("   - Complex roots create cyclical forecast patterns\n")
    
    print("=== END OF SIMULATION TUTORIAL ===")

# ===============================================================================
# MAIN EXECUTION
# ===============================================================================

def main():
    """
    Execute the complete simulation tutorial
    """
    # Part 1: Simulate AR(2) processes
    ar2_real, ar2_complex = simulate_ar2_processes()
    
    # Part 2: Analyze characteristic roots
    cycle_length = analyze_characteristic_roots()
    
    # Part 3: Model identification and estimation
    model_identification_estimation(ar2_real, ar2_complex)
    
    # Part 4: AR(2) forecasting comparison (real vs complex roots)
    ar2_forecasting_comparison(ar2_real, ar2_complex)
    
    # Part 5: Simple process forecasting comparison (AR(1), MA(1), WN)
    simple_process_forecasting_comparison()
    
    # Part 6: Higher-order processes (AR(3) vs MA(3))
    higher_order_comparison()
    
    # Summary
    print_summary()

if __name__ == "__main__":
    main()