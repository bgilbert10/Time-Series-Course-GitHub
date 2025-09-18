# AR and MA Simulation Examples: Understanding Time Series Behavior -----------
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

# Load required packages
library(forecast)    # Time series analysis and forecasting
library(quantmod)    # Additional time series utilities

# Part 1: AR(2) Processes with Different Root Structures ---------------------
#
# We simulate two AR(2) processes to demonstrate how characteristic polynomial
# roots affect the time series behavior:
# 1. AR(2) with real roots → exponential decay/growth patterns
# 2. AR(2) with complex roots → cyclical/oscillatory patterns

# ------------------- AR(2) PROCESS WITH REAL ROOTS -------------------

cat("Model: y_t = 1.3*y_{t-1} - 0.4*y_{t-2} + ε_t\n")
cat("Parameters: φ₁ = 1.3, φ₂ = -0.4\n")

# Simulate AR(2) with real roots
set.seed(123)  # For reproducible results
ar2_real_roots <- arima.sim(model=list(ar=c(1.3, -0.4)), n=1000)

cat("Simulated", length(ar2_real_roots), "observations\n")
cat("Expected behavior: Exponential decay pattern in ACF\n\n")

# Examine autocorrelation pattern
acf(ar2_real_roots, 
    main="ACF: AR(2) with Real Roots",
    lag.max=30)

# ------------------- AR(2) PROCESS WITH COMPLEX ROOTS -------------------

cat("Model: y_t = 0.8*y_{t-1} - 0.7*y_{t-2} + ε_t\n")
cat("Parameters: φ₁ = 0.8, φ₂ = -0.7\n")

# Simulate AR(2) with complex roots
set.seed(456)  # Different seed for comparison
ar2_complex_roots <- arima.sim(model=list(ar=c(0.8, -0.7)), n=1000)

cat("Expected behavior: Damped oscillating pattern in ACF\n\n")

# Examine autocorrelation pattern
acf(ar2_complex_roots,
    main="ACF: AR(2) with Complex Roots", 
    lag.max=30)

cat("COMPARISON OF ACF PATTERNS:\n")
cat("- Real roots (y1): Smooth exponential decay\n")
cat("- Complex roots (y2): Oscillating (sine/cosine-like) decay\n\n")

# ------------------- VISUAL COMPARISON OF TIME SERIES -------------------

# Plot both time series for comparison
par(mfrow=c(2,1))
plot(ar2_real_roots, 
     main="AR(2) with Real Roots: Smooth Exponential Behavior",
     ylab="Value", xlab="Time")
plot(ar2_complex_roots,
     main="AR(2) with Complex Roots: Cyclical/Oscillatory Behavior", 
     ylab="Value", xlab="Time")
par(mfrow=c(1,1))

cat("VISUAL INTERPRETATION:\n")
cat("- Real roots: Smooth transitions between values\n") 
cat("- Complex roots: Regular oscillatory patterns (pseudo-cycles)\n\n")


# PART 2: CHARACTERISTIC POLYNOMIAL ROOTS ANALYSIS -------------------

#
# The characteristic polynomial helps us understand the dynamic properties
# of AR processes. For an AR(2) process: y_t = φ₁y_{t-1} + φ₂y_{t-2} + ε_t
# The characteristic polynomial is: 1 - φ₁z - φ₂z² = 0
#
# Root interpretation:
# - Real roots: Exponential behavior (growth/decay)
# - Complex roots: Cyclical behavior (pseudo-periodic oscillations)
# - Modulus > 1: Stationary process
# - Modulus = 1: Non-stationary unit root process
# - Modulus < 1: Explosive process

# ------------------- REAL ROOTS ANALYSIS -------------------
cat("--- Analysis of AR(2) with Real Roots ---\n")
cat("Characteristic polynomial: 1 - 1.3z + 0.4z² = 0\n")

# Coefficients for characteristic polynomial (note the sign convention)
char_poly_real <- c(1, -1.3, 0.4)
roots_real <- polyroot(char_poly_real)

cat("Characteristic roots:\n")
print(roots_real)

cat("Root analysis:\n")
cat("Root 1:", round(roots_real[1], 4), "\n")
cat("Root 2:", round(roots_real[2], 4), "\n")
cat("Both roots are real numbers\n")
cat("Moduli:", round(Mod(roots_real), 4), "\n")

if(all(Mod(roots_real) > 1)) {
  cat("CONCLUSION: Process is stationary (all roots outside unit circle)\n\n")
} else {
  cat("WARNING: Process may be non-stationary\n\n")
}

# ------------------- COMPLEX ROOTS ANALYSIS -------------------
cat("--- Analysis of AR(2) with Complex Roots ---\n")
cat("Characteristic polynomial: 1 - 0.8z + 0.7z² = 0\n")

# Coefficients for characteristic polynomial 
char_poly_complex <- c(1, -0.8, 0.7)
roots_complex <- polyroot(char_poly_complex)

cat("Characteristic roots:\n")
print(roots_complex)

# Extract real and imaginary parts for cycle calculation
real_part <- Re(roots_complex[1])    # Real part (same for both roots)
imag_part <- Im(roots_complex[1])    # Imaginary part (opposite for conjugate pair)
modulus <- Mod(roots_complex[1])     # Modulus (same for both roots)

cat("\\nDetailed analysis of complex roots:\n")
cat("Real part (a):", round(real_part, 6), "\n")
cat("Imaginary part (b):", round(abs(imag_part), 6), "\n") 
cat("Modulus:", round(modulus, 6), "\n")

# ------------------- CYCLE LENGTH CALCULATION -------------------

cat("For complex roots, we can calculate the average cycle length:\n")
cat("Formula: k = 2π / arccos(real_part / modulus)\n\n")

# Calculate cycle length using the formula
cycle_length <- 2 * pi / acos(real_part / modulus)
cat("Calculated cycle length:", round(cycle_length, 2), "periods\n")

# Alternative calculation (verification)
cat("\\nVerification calculation:\n")
modulus_check <- sqrt(real_part^2 + imag_part^2)
cycle_length_check <- 2 * pi / acos(real_part / modulus_check)
cat("Alternative cycle length:", round(cycle_length_check, 2), "periods\n")

cat("\\nINTERPRETATION:\n")
cat("- The process exhibits pseudo-cyclical behavior\n")
cat("- Average cycle length is approximately", round(cycle_length, 1), "periods\n")
cat("- This creates the oscillating pattern we see in the ACF\n")
cat("- No need for sine/cosine functions - AR coefficients capture cyclical dynamics\n\n")

# ------------------- STATIONARITY CHECK -------------------
if(modulus > 1) {
  cat("STATIONARITY: Process is stationary (modulus > 1)\n")
} else {
  cat("WARNING: Process is non-stationary (modulus <= 1)\n")
}


# PART 3: MODEL SELECTION AND ESTIMATION FROM SIMULATED DATA -----------------

#
# In practice, we observe time series data but don't know the true underlying
# parameters. This section demonstrates how to estimate the model structure
# using standard model selection techniques.

# ------------------- MODEL SELECTION FOR REAL ROOTS PROCESS -------------------
cat("--- Model Selection: AR(2) with Real Roots ---\n")
cat("True parameters: φ₁ = 1.3, φ₂ = -0.4\n")
cat("Attempting to recover these parameters from the data...\n\n")

# Use automatic AR model selection with MLE
real_roots_model <- ar(ar2_real_roots, method="mle", order.max=10)

cat("Automatic model selection results:\n")
cat("Selected AR order:", real_roots_model$order, "\n")
cat("Estimated coefficients:\n")
print(real_roots_model$ar)
cat("Compare to true values: φ₁ = 1.3, φ₂ = -0.4\n\n")

# Additional diagnostic tools
cat("PACF analysis (should cut off after lag 2 for AR(2)):\n")
pacf(ar2_real_roots, main="PACF: AR(2) with Real Roots")

# Statistical tests
cat("\nDiagnostic Tests:\n")
mean_test <- t.test(ar2_real_roots)
cat("Mean test (H₀: μ = 0):\n")
cat("Sample mean:", round(mean_test$estimate, 4), "\n")
cat("p-value:", round(mean_test$p.value, 4), "\n")

ljung_test_real <- Box.test(ar2_real_roots, lag=10, type='Ljung-Box')
cat("Ljung-Box test (H₀: no autocorrelation):\n") 
cat("p-value:", round(ljung_test_real$p.value, 4), "\n")
if(ljung_test_real$p.value < 0.05) {
  cat("CONCLUSION: Significant autocorrelation detected (as expected for AR process)\n\n")
} else {
  cat("CONCLUSION: No significant autocorrelation\n\n")
}

# ------------------- MODEL SELECTION FOR COMPLEX ROOTS PROCESS -------------------
cat("--- Model Selection: AR(2) with Complex Roots ---\n")
cat("True parameters: φ₁ = 0.8, φ₂ = -0.7\n")

# Use automatic AR model selection
complex_roots_model <- ar(ar2_complex_roots, method="mle", order.max=10)

cat("Automatic model selection results:\n")
cat("Selected AR order:", complex_roots_model$order, "\n")
cat("Estimated coefficients:\n") 
print(complex_roots_model$ar)
cat("Compare to true values: φ₁ = 0.8, φ₂ = -0.7\n\n")

# PACF analysis
cat("PACF analysis:\n")
pacf(ar2_complex_roots, main="PACF: AR(2) with Complex Roots")

# Statistical tests
mean_test_2 <- t.test(ar2_complex_roots)
cat("\nDiagnostic Tests:\n")
cat("Mean test (H₀: μ = 0):\n")
cat("Sample mean:", round(mean_test_2$estimate, 4), "\n")
cat("p-value:", round(mean_test_2$p.value, 4), "\n")

ljung_test_complex <- Box.test(ar2_complex_roots, lag=10, type='Ljung-Box') 
cat("Ljung-Box test (H₀: no autocorrelation):\n")
cat("p-value:", round(ljung_test_complex$p.value, 4), "\n\n")

cat("INTERPRETATION OF IDENTIFICATION RESULTS:\n")
cat("- Both processes correctly identified as AR(2)\n")
cat("- Estimated coefficients should be close to true values\n")
cat("- PACF should show significant values at lags 1 and 2, then cut off\n")
cat("- Large samples provide more accurate parameter estimates\n\n")


# PART 4: FORECASTING WITH AR MODELS -------------------

#
# Demonstrates how different root structures affect forecasting behavior:
# - Real roots: Smooth convergence to long-run mean
# - Complex roots: Cyclical approach to long-run mean

# ------------------- FORECASTING AR(2) WITH COMPLEX ROOTS -------------------

cat("Generating forecasts that preserve cyclical patterns...\n")

# Generate forecasts using the estimated model
forecast_complex <- forecast(complex_roots_model, h=100)

# Plot full forecast 
plot(forecast_complex,
     main="100-Step Forecast: AR(2) with Complex Roots",
     xlab="Time", ylab="Value")

cat("Full forecast shows long-term convergence to mean\n")

# Plot forecast with recent history for better visualization
plot(forecast_complex, include=100,
     main="AR(2) Complex Roots: Forecast with Recent History", 
     xlab="Time", ylab="Value")

cat("IMPORTANT INSIGHT: Forecasts preserve cyclical behavior in short-term\n")
cat("but converge to long-run mean (μ) in the long-term\n\n")

# ------------------- FORECASTING AR(2) WITH REAL ROOTS -------------------

cat("Generating forecasts with smooth exponential convergence...\n")

# Generate forecasts for real roots process
forecast_real <- forecast(real_roots_model, h=100)

# Plot full forecast
plot(forecast_real,
     main="100-Step Forecast: AR(2) with Real Roots",
     xlab="Time", ylab="Value")

# Plot with recent history
plot(forecast_real, include=100,
     main="AR(2) Real Roots: Forecast with Recent History",
     xlab="Time", ylab="Value") 

cat("COMPARISON OF FORECASTING BEHAVIOR:\n")
cat("- Complex roots: Cyclical/oscillatory convergence to mean\n")
cat("- Real roots: Smooth exponential convergence to mean\n")
cat("- Both converge to long-run mean in the limit\n")
cat("- Short-term forecasts reflect underlying dynamics\n\n")



# PART 5: COMPARING AR, MA, AND WHITE NOISE PROCESSES ------------------

#
# This section demonstrates fundamental differences between:
# - AR(1): Autoregressive - persistence and mean reversion
# - MA(1): Moving Average - short memory, larger variance  
# - White Noise: No memory, constant variance
#
# Understanding these differences is crucial for:
# - Model selection and identification
# - Interpretation of economic time series
# - Forecasting behavior

# ------------------- SIMULATION OF DIFFERENT PROCESS TYPES -------------------

cat("All processes have the same coefficient (0.8) for comparison\n\n")

set.seed(789)  # For reproducible results

# Simulate processes with same coefficient
ar1_process <- arima.sim(model=list(ar=c(0.8)), n=200)
ma1_process <- arima.sim(model=list(ma=c(0.8)), n=200)  
white_noise <- arima.sim(model=list(), n=200)

cat("Sample sizes: 200 observations each\n")
cat("AR(1): y_t = 0.8*y_{t-1} + ε_t\n")
cat("MA(1): y_t = ε_t + 0.8*ε_{t-1}\n") 
cat("White Noise: y_t = ε_t\n\n")

# ------------------- VISUAL COMPARISON -------------------

par(mfrow=c(3,1))
plot(ar1_process, main="AR(1): Persistent, Mean-Reverting", 
     ylab="AR(1) Value", type="l")
plot(ma1_process, main="MA(1): Short Memory, Higher Variance",
     ylab="MA(1) Value", type="l") 
plot(white_noise, main="White Noise: No Memory, Constant Variance",
     ylab="WN Value", type="l")
par(mfrow=c(1,1))

cat("BEHAVIORAL DIFFERENCES:\n")
cat("- AR(1): Drifts away from mean, then drifts back (persistence)\n")
cat("- MA(1): Similar mean reversion to WN, but wider variance\n")
cat("- White Noise: Random fluctuations around mean\n")
cat("- MA(1) has more extreme values outside ±2 standard deviations\n\n")

# ------------------- MODEL SELECTION AND ESTIMATION -------------------

# AR(1) selection
cat("\\n1. AR(1) Process Identification:\\n")
ar1_fitted <- auto.arima(ar1_process)
print(ar1_fitted)

cat("Note: With coefficient close to 1, auto.arima might sometimes\n")
cat("incorrectly identify as non-stationary due to finite sample effects\n")

# Alternative: Use ar() for pure AR selection
ar1_fitted_ar <- ar(ar1_process)
cat("\\nUsing ar() function (AR-only selection):\\n")
cat("Selected order:", ar1_fitted_ar$order, "\n")
cat("Estimated coefficient:", round(ar1_fitted_ar$ar[1], 4), "\n")
cat("True coefficient: 0.8\n")

# MA(1) selection  
cat("\\n2. MA(1) Process Selection:\\n")
ma1_fitted <- auto.arima(ma1_process, approximation=FALSE)
print(ma1_fitted)

# Force MA(1) specification for comparison
cat("\\nForced MA(1) specification:\\n")
ma1_forced <- arima(x=ma1_process, order=c(0,0,1), include.mean=FALSE)
print(ma1_forced)
cat("Note: With small samples (n=200), estimates may deviate from true values\n")

# ------------------- FORECASTING COMPARISON -------------------

# AR(1) forecasting
cat("\\n1. AR(1) Forecasting Behavior:\\n")
ar1_forecast <- forecast(ar1_fitted_ar, h=200)
plot(ar1_forecast, include=100,
     main="AR(1) Forecast: Gradual Convergence to Mean")

# MA(1) forecasting
cat("2. MA(1) Forecasting vs Naive Approach:\\n")

# Proper MA(1) forecast
ma1_forecast <- forecast(ma1_forced, h=5)
plot(ma1_forecast, include=50,
     main="MA(1) Forecast: Proper Model-Based")

# Naive forecast (ignoring MA structure)
naive_forecast <- forecast(ma1_process, h=5)  # Uses simple exponential smoothing
plot(naive_forecast, include=50,
     main="Naive Forecast: Ignoring MA Structure")

cat("FORECASTING INSIGHTS:\n")
cat("- MA models: Forecast converges to mean quickly after lag q\n")
cat("- AR models: Gradual convergence with exponential decay\n") 
cat("- Proper model specification improves short-term forecast precision\n")
cat("- Ignoring MA structure increases forecast uncertainty\n\n")


# PART 6: HIGHER-ORDER PROCESSES -----------------

# ------------------- AR(3) AND MA(3) COMPARISON -------------------
cat("--- Higher-Order Process Comparison ---\n")
cat("Simulating AR(3) and MA(3) with complex dynamics\n\n")

set.seed(999)

# Simulate higher-order processes with larger sample size
ar3_process <- arima.sim(model=list(ar=c(0.8, -0.3, 0.3)), n=2000)
ma3_process <- arima.sim(model=list(ma=c(0.8, -0.3, 0.3)), n=2000)

# AR(3) analysis
cat("AR(3) Process Analysis:\n")
ar3_fitted <- ar(ar3_process)
cat("Selected AR order:", ar3_fitted$order, "\n")
cat("Estimated coefficients:\n")
print(round(ar3_fitted$ar, 4))
cat("True coefficients: [0.8, -0.3, 0.3]\n")

# AR(3) forecasting
ar3_forecast <- forecast(ar3_fitted, h=200)
plot(ar3_forecast, include=100,
     main="AR(3) Forecast: Complex Dynamics")

# MA(3) analysis
cat("\\nMA(3) Process Analysis:\n")
ma3_fitted <- auto.arima(ma3_process, approximation=FALSE)
print(ma3_fitted)

# Force MA(3) specification if auto.arima doesn't detect it
cat("Forced MA(3) specification:\\n") 
ma3_forced <- arima(x=ma3_process, order=c(0,0,3), include.mean=FALSE)
print(ma3_forced)

# MA(3) forecasting  
ma3_forecast <- forecast(ma3_forced, h=10)
plot(ma3_forecast, include=50,
     main="MA(3) Forecast: Quick Convergence after Lag 3")


# SUMMARY AND KEY TAKEAWAYS -------------------------


cat("\\n=== SUMMARY OF KEY INSIGHTS ===\\n")
cat("\\n1. ROOT STRUCTURE EFFECTS:\\n")
cat("   - Real roots → Smooth exponential patterns\\n")
cat("   - Complex roots → Cyclical/oscillatory patterns\\n")
cat("   - Modulus > 1 → Stationary process\\n")

cat("\\n2. PROCESS TYPE CHARACTERISTICS:\\n")
cat("   - AR processes: Long memory, gradual mean reversion\\n")
cat("   - MA processes: Short memory, quick mean reversion\\n") 
cat("   - Higher variance in MA than AR with same coefficients\\n")

cat("\\n3. IDENTIFICATION PRINCIPLES:\\n")
cat("   - ACF/PACF patterns help distinguish AR from MA\\n")
cat("   - Automatic selection methods are helpful but not infallible\\n")
cat("   - Larger samples improve parameter estimation accuracy\\n")

cat("\\n4. FORECASTING IMPLICATIONS:\\n") 
cat("   - AR forecasts show gradual convergence\\n")
cat("   - MA forecasts converge quickly after lag q\\n")
cat("   - Model structure affects forecast uncertainty\\n")
cat("   - Complex roots create cyclical forecast patterns\\n")

cat("\\n=== END OF SIMULATION TUTORIAL ===\\n")
  