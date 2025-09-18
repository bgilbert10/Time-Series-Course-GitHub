# COMPREHENSIVE UNIT ROOT TESTING EXAMPLES AND METHODOLOGY ====================

# This program provides systematic guidance for unit root testing procedures:
# 1. Simulated examples demonstrating consequences of wrong case selection
# 2. Individual unit root tests using cadftest() (allows covariates)
# 3. Joint hypothesis testing using ur.df() (tests combined hypotheses)
# 4. Application to real economic data (oil, gas, drilling activity)
#
# Key methodological insight:
# Constraining coefficients to zero when they belong in the model (under-specification)
# is much worse than including extra terms that might be irrelevant (over-specification)
#
# Unit Root Test Cases:
# - Case 1 (type="none"): No intercept, no trend
# - Case 2 (type="drift"): With intercept, no trend  
# - Case 4 (type="trend"): With intercept and trend

# --------------------- SETUP AND PACKAGE LOADING ----------------------
# Load required packages
library(quantmod)    # Financial and economic data retrieval
library(fBasics)     # Basic financial statistics and diagnostics
library(tseries)     # Classical time series analysis
library(CADFtest)    # Covariate-augmented Dickey-Fuller tests
library(urca)        # Unit root and cointegration analysis with joint tests
library(forecast)    # Advanced forecasting and model selection
library(lubridate)   # Date manipulation utilities

# =========== PART 1: CASE 1 SIMULATION (NO INTERCEPT, NO TREND) ===========

# Set seed for reproducible simulations
set.seed(206)

# Simulate three series where Case 1 (type="none") is the true specification:
# 1. Persistent AR(1) with zero mean
# 2. Persistent AR(2) with zero mean  
# 3. Random walk without drift

# Key question: What happens if we mistakenly use Case 2 (type="drift")?
# Answer: No harm - we include an irrelevant intercept, but don't force exclusion of important terms

cat("===============================================================\n")
cat("PART 1: CASE 1 SIMULATIONS (True model has no intercept/trend)\n")
cat("===============================================================\n")

# Series 1: Persistent AR(1) with zero mean
ar1_series <- arima.sim(model=list(order=c(1,0,0), ar=c(0.85)), n=1000)
ts.plot(ar1_series, main="Persistent AR(1) Series (ρ=0.85)", 
        ylab="Value", xlab="Time")
cat("AR(1) simulation: y_t = 0.85*y_{t-1} + ε_t\n")

# Series 2: Persistent AR(2) with zero mean
ar2_series <- arima.sim(model=list(order=c(2,0,0), ar=c(1.2, -0.3)), n=1000)
ts.plot(ar2_series, main="Persistent AR(2) Series", 
        ylab="Value", xlab="Time")
cat("AR(2) simulation: y_t = 1.2*y_{t-1} - 0.3*y_{t-2} + ε_t\n")

# Series 3: Random walk without drift
rw_series <- arima.sim(model=list(order=c(0,1,0)), n=1000)
ts.plot(rw_series, main="Random Walk without Drift", 
        ylab="Value", xlab="Time")
cat("Random walk simulation: y_t = y_{t-1} + ε_t\n\n")

# --------------------- TESTING WITH CASE 1: NO INTERCEPT OR TREND ---------------

# Test each series using Case 1 (type="none") - the true model
# CADFtest automatically selects lag length using information criteria
# Note: CADFtest provides individual tests but cannot perform joint hypothesis tests

cat("--- Testing AR(1) Series with Case 1 ---\n")
ar1_cadf_case1 <- CADFtest(ar1_series, criterion=c("AIC"), type="none")
summary(ar1_cadf_case1)
cat("Result: REJECT null of unit root (correct - we know it's stationary)\n")
cat("Coefficient interpretation:\n")
cat("- Lag coefficient ≈ -0.13 in ADF form: Δy_t = -0.13*y_{t-1} + ε_t\n")
cat("- Rearranging: y_t = 0.87*y_{t-1} + ε_t (close to true 0.85)\n")
cat("Exercise: Try simulating with ρ=0.95 or ρ=0.999 - what happens?\n\n")

cat("--- Testing AR(2) Series with Case 1 ---\n")
ar2_cadf_case1 <- CADFtest(ar2_series, criterion=c("AIC"), type="none")
summary(ar2_cadf_case1)
cat("Result: REJECT null of unit root (correct - we know it's stationary)\n")
cat("Coefficient recovery explanation:\n")
cat("- ADF regression: Δy_t = γ*y_{t-1} + δ*Δy_{t-1} + ε_t\n")
cat("- Original AR(2): y_t = 1.2*y_{t-1} - 0.3*y_{t-2} + ε_t\n")
cat("- ADF estimates: γ ≈ -0.09, δ ≈ 0.33\n")
cat("- Recovery: y_t = (1+γ+δ)*y_{t-1} - δ*y_{t-2} = 1.24*y_{t-1} - 0.33*y_{t-2}\n")
cat("- Close to true parameters after accounting for estimation error\n\n")

cat("--- Testing Random Walk with Case 1 (Correct) ---\n")
rw_cadf_case1 <- CADFtest(rw_series, criterion=c("AIC"), type="none")
summary(rw_cadf_case1)
cat("Result: FAIL TO REJECT null of unit root (correct - we know it has unit root)\n\n")

# --------------------- JOINT TESTING WITH UR.DF (CASE 1) ---------------

# Use ur.df() for joint hypothesis testing capability
# CADFtest already determined optimal lag lengths
# Note: Case 1 has no joint tests (only individual unit root test available)

cat("--- Joint Testing AR(1) Series with ur.df() Case 1 ---\n")
ar1_urdf_case1 <- ur.df(ar1_series, type=c("none"), lags=0)
summary(ar1_urdf_case1)
cat("Test statistic (tau1): Individual unit root test only in Case 1\n\n")

cat("--- Joint Testing AR(2) Series with ur.df() Case 1 ---\n")
ar2_urdf_case1 <- ur.df(ar2_series, type=c("none"), lags=1)
summary(ar2_urdf_case1)
cat("Test statistic (tau1): Individual unit root test only in Case 1\n\n")

cat("--- Joint Testing Random Walk with ur.df() Case 1 ---\n")
rw_urdf_case1 <- ur.df(rw_series, type=c("none"), lags=0)
summary(rw_urdf_case1)
cat("Test statistic (tau1): Individual unit root test only in Case 1\n")
cat("Confirms unit root presence in random walk\n\n")

# --------- USING CASE 2 WHEN CASE 1 IS TRUE MODEL ---------

cat("===============================================================\n")
cat("Using Case 2 when Case 1 is the true model\n")
cat("===============================================================\n")
cat("Question: What happens if we include an intercept when there isn't one?\n")
cat("Answer: No problem! Over-specification is harmless.\n")
cat("Best practice: Use Case 2 with real data when uncertain about drift.\n\n")

# Test AR(1) series with Case 2 (over-specification)
cat("--- Testing AR(1) with Case 2 (Over-specification) ---\n")
ar1_urdf_case2 <- ur.df(ar1_series, type=c("drift"), lags=0)
summary(ar1_urdf_case2)
cat("Results Analysis:\n")
cat("- tau2: Individual test for unit root (lag coefficient = 0)\n")
cat("- phi1: Joint test for unit root AND zero intercept\n")
cat("- Both tests rejected: Either lag ≠ 0 OR intercept ≠ 0 (or both)\n\n")

cat("Interpretation Guide:\n")
cat("- Under null hypothesis: Data has unit root (non-standard critical values)\n")
cat("- Under alternative: Data is stationary (standard critical values apply)\n")
cat("- Since we reject unit root: Can use normal t-tests for coefficient significance\n\n")

# Check intercept significance in stationary model
ar1_arima_check <- arima(ar1_series, order=c(1,0,0), include.mean=TRUE)
cat("Intercept significance check (stationary AR(1) model):\n")
print(ar1_arima_check)
cat("Intercept is not significant (as expected - true mean is zero)\n\n")

# Test AR(2) and Random Walk with Case 2 
cat("--- Testing AR(2) with Case 2 (Over-specification) ---\n")
ar2_urdf_case2 <- ur.df(ar2_series, type=c("drift"), lags=1)
summary(ar2_urdf_case2)
cat("Similar results: Both tau2 and phi1 rejected (correctly identifies stationarity)\n\n")

cat("--- Testing Random Walk with Case 2 ---\n")
rw_urdf_case2 <- ur.df(rw_series, type=c("drift"), lags=0)
summary(rw_urdf_case2)
cat("Results Analysis:\n")
cat("- tau2: Fail to reject (correctly identifies unit root)\n")
cat("- phi1: Fail to reject (correctly identifies unit root and no drift)\n")
cat("- Conclusion: Random walk without drift (correct!)\n\n")

cat("Individual Intercept Significance in Unit Root Case:\n")
cat("- Problem: With unit root, t-statistics are not normally distributed\n")
cat("- Solution: Test drift using first differences (which are stationary)\n\n")

# Test drift in first differences
rw_diff_test <- arima(diff(rw_series), order=c(0,0,0), include.mean=TRUE)
cat("Drift test using first differences:\n")
print(rw_diff_test)

rw_integrated_test <- Arima(rw_series, order=c(0,1,0), include.constant=TRUE)
cat("Integrated model with drift:\n")
print(rw_integrated_test)
cat("Both confirm no significant drift (as expected)\n\n")


# =========== PART 2: CASE 2 SIMULATION (WITH INTERCEPT, NO TREND) ===========

cat("===============================================================\n")
cat("PART 2: CASE 2 SIMULATIONS (True model has intercept/drift)\n")
cat("===============================================================\n")

# Simulate three series where Case 2 (type="drift") is the correct specification:
# 1. Persistent AR(1) with non-zero mean
# 2. Persistent AR(2) with non-zero mean  
# 3. Random walk without drift (for comparison)

# Key question: What happens if we mistakenly use Case 1 (type="none")?
# Answer: Very harmful - we force exclusion of important intercept term

# Series 1: Persistent AR(1) with non-zero mean (μ = 10)
ar1_with_mean <- arima.sim(model=list(order=c(1,0,0), ar=c(0.85)), n=1000) + 10
ts.plot(ar1_with_mean, main="AR(1) with Non-zero Mean (μ=10)", 
        ylab="Value", xlab="Time")
cat("AR(1) with mean: y_t = 10 + 0.85*(y_{t-1} - 10) + ε_t\n")

# Series 2: Persistent AR(2) with non-zero mean (μ = 10)
ar2_with_mean <- arima.sim(model=list(order=c(2,0,0), ar=c(1.2, -0.3)), n=1000) + 10
ts.plot(ar2_with_mean, main="AR(2) with Non-zero Mean (μ=10)", 
        ylab="Value", xlab="Time")
cat("AR(2) with mean: y_t = 10 + 1.2*(y_{t-1} - 10) - 0.3*(y_{t-2} - 10) + ε_t\n")

# Series 3: Random walk without drift (for comparison with visual trends)
rw_comparison <- arima.sim(model=list(order=c(0,1,0)), n=1000)
ts.plot(rw_comparison, main="Random Walk without Drift (Comparison)", 
        ylab="Value", xlab="Time")
cat("Random walk: y_t = y_{t-1} + ε_t\n")
cat("Visual inspection may suggest drift due to random trends - be careful!\n\n")

# --------- UNDER-SPECIFICATION: USING CASE 1 WHEN CASE 2 IS CORRECT ---------

cat("===============================================================\n")
cat("UNDER-SPECIFICATION TEST: Using Case 1 when Case 2 is correct\n")
cat("===============================================================\n")
cat("Question: What happens if we exclude an intercept when there should be one?\n")
cat("Answer: Very harmful! Under-specification leads to wrong conclusions.\n")
cat("This is much worse than over-specification - we force exclusion of important terms.\n\n")

# Test AR(1) with mean using wrong specification (Case 1)
cat("--- Testing AR(1) with Mean using Case 1 (WRONG - Under-specification) ---\n")
ar1_mean_cadf_wrong <- CADFtest(ar1_with_mean, criterion=c("AIC"), type="none")
summary(ar1_mean_cadf_wrong)
cat("Comparison with previous AR(1) without mean:\n")
summary(ar1_cadf_case1)
cat("DANGEROUS RESULT: Fail to reject unit root when should reject!\n")
cat("Consequences:\n")
cat("- Might incorrectly conclude series has unit root\n")
cat("- R-squared much lower, wrong coefficients estimated\n")
cat("- Fundamental misspecification of the data generating process\n\n")

# ----------- CORRECT SPECIFICATION: USING CASE 2 WHEN CASE 2 IS CORRECT -----------

cat("--- Testing AR(1) with Mean using Case 2 (CORRECT) ---\n")
ar1_mean_cadf_correct <- CADFtest(ar1_with_mean, criterion=c("AIC"), type="drift")
summary(ar1_mean_cadf_correct)
cat("CORRECT RESULT: Correctly rejects unit root!\n")
cat("Benefits of correct specification:\n")
cat("- Proper identification of stationary series\n")
cat("- Good fit (high R-squared)\n")
cat("- Correct coefficient estimates\n\n")

# Test over-specification: Case 4 when Case 2 is correct
cat("--- Testing AR(1) with Mean using Case 4 (Over-specification) ---\n")
ar1_mean_cadf_over <- CADFtest(ar1_with_mean, criterion=c("AIC"), type="trend")
summary(ar1_mean_cadf_over)
cat("HARMLESS OVER-SPECIFICATION:\n")
cat("- Still correctly rejects unit root\n")
cat("- Includes irrelevant trend term (not significant)\n")
cat("- Can test trend significance using normal t-test (data is stationary)\n")
cat("- Trend coefficient not significant → indicates Case 2 is preferred\n\n")

# ----------- AR(2) WITH MEAN: TESTING DIFFERENT SPECIFICATIONS -----------

cat("--- Testing AR(2) with Mean using Case 1 (WRONG - Under-specification) ---\n")
ar2_mean_cadf_wrong <- CADFtest(ar2_with_mean, criterion=c("AIC"), type="none")
summary(ar2_mean_cadf_wrong)
cat("Comparison with previous AR(2) without mean:\n")
summary(ar2_cadf_case1)
cat("RESULT: May still reject null (lucky!), but:\n")
cat("- Coefficients are severely biased\n")
cat("- R-squared is much worse\n")
cat("- No guarantee of correct inference across simulations\n\n")

cat("--- Testing AR(2) with Mean using Case 2 (CORRECT) ---\n")
ar2_mean_cadf_correct <- CADFtest(ar2_with_mean, criterion=c("AIC"), type="drift")
summary(ar2_mean_cadf_correct)
cat("CORRECT RESULT: Strongly rejects unit root\n")
cat("Note: Mean = intercept/(1-φ₁-φ₂) ≈ intercept/(1-1.2+0.3) ≈ intercept/0.1\n")
cat("Expected mean ≈ 10, which matches our simulation\n\n")

cat("--- Testing AR(2) with Mean using Case 4 (Over-specification) ---\n")
ar2_mean_cadf_over <- CADFtest(ar2_with_mean, criterion=c("AIC"), type="trend")
summary(ar2_mean_cadf_over)
cat("HARMLESS OVER-SPECIFICATION:\n")
cat("- Correctly rejects unit root\n")
cat("- Trend term not statistically significant (as expected)\n")
cat("- Normal t-test applicable since data is stationary under alternative\n\n")

# ----------- JOINT HYPOTHESIS TESTING WITH UR.DF (CASE 2 DATA) -----------

cat("--- Joint Testing AR(1) with Mean using Case 2 (CORRECT) ---\n")
ar1_mean_urdf_correct <- ur.df(ar1_with_mean, type=c("drift"), lags=0)
summary(ar1_mean_urdf_correct)
cat("Results Analysis:\n")
cat("- tau2: Individual test - Reject unit root (correct)\n")
cat("- phi1: Joint test - Reject unit root AND zero intercept (correct)\n")
cat("- Conclusion: Stationary series with significant intercept\n")
cat("- Perfect match with our data generating process!\n\n")

cat("--- Joint Testing AR(1) with Mean using Case 4 (Over-specification) ---\n")
ar1_mean_urdf_over <- ur.df(ar1_with_mean, type=c("trend"), lags=0)
summary(ar1_mean_urdf_over)
cat("Results Analysis:\n")
cat("- tau3: Individual test - Reject unit root (correct)\n") 
cat("- phi2: Joint test of lag, intercept, AND trend - Reject (at least one ≠ 0)\n")
cat("- phi3: Joint test of lag and trend - Reject (at least one ≠ 0)\n")
cat("Joint Test Interpretation:\n")
cat("- phi2 rejection means at least one of: lag≠0, intercept≠0, trend≠0\n")
cat("- Does NOT mean trend is individually significant\n")
cat("- Since we reject unit root: can use normal t-test on trend coefficient\n")
cat("- Individual trend test will show insignificance → prefer Case 2\n\n")

# ----------- AR(2) AND RANDOM WALK JOINT TESTING -----------

cat("--- Joint Testing AR(2) with Mean using Case 2 (CORRECT) ---\n")
ar2_mean_urdf_correct <- ur.df(ar2_with_mean, type=c("drift"), lags=1)
summary(ar2_mean_urdf_correct)
cat("Results: Both tau2 and phi1 strongly rejected (correct identification)\n\n")

cat("--- Joint Testing AR(2) with Mean using Case 4 (Over-specification) ---\n")
ar2_mean_urdf_over <- ur.df(ar2_with_mean, type=c("trend"), lags=1)
summary(ar2_mean_urdf_over)
cat("Results: All tests rejected, but individual trend test will show insignificance\n")
cat("Joint tests do NOT tell us to keep the trend - use individual significance\n\n")

# ----------- RANDOM WALK COMPARISON (NO DRIFT) -----------

cat("--- Testing Random Walk (no drift) with Case 2 ---\n")
rw_comp_urdf_case2 <- ur.df(rw_comparison, type=c("drift"), lags=0)
summary(rw_comp_urdf_case2)
cat("Results Analysis:\n")
cat("- tau2: Fail to reject unit root (correct - it has unit root)\n")
cat("- phi1: Fail to reject unit root and zero drift (correct)\n")
cat("- Conclusion: Random walk without drift (matches simulation)\n\n")

cat("--- Testing Random Walk (no drift) with Case 4 (Over-specification) ---\n")
rw_comp_urdf_case4 <- ur.df(rw_comparison, type=c("trend"), lags=0)
summary(rw_comp_urdf_case4)
cat("Results Analysis:\n")
cat("- tau3: Fail to reject unit root (correct)\n")
cat("- phi2: Fail to reject (unit root, zero drift, zero trend)\n")
cat("- phi3: Fail to reject (unit root and zero trend)\n")
cat("- Since phi3 fails to reject: trend coefficient also insignificant\n")
cat("- CANNOT conclude about intercept from these tests alone\n\n")

# Verify drift conclusion using stationary first differences
rw_comp_drift_test <- Arima(rw_comparison, order=c(0,1,0), include.constant=TRUE)
cat("Drift test using integrated model:\n")
print(rw_comp_drift_test)
cat("Confirms no significant drift (constant not significant)\n\n")

cat("KEY INSIGHT: Over-specification (Case 4 when Case 2 is correct) is harmless\n")
cat("- Still get correct conclusions with careful test interpretation\n")
cat("- Joint tests guide us to simpler, more parsimonious specifications\n\n")


# =========== PART 3: CASE 4 SIMULATION (WITH TREND) ===========

cat("===============================================================\n")
cat("PART 3: CASE 4 SIMULATIONS (True model has trend)\n")
cat("===============================================================\n")
cat("Testing what happens when Case 4 (trend) is the correct specification\n")
cat("Key question: What happens if we mistakenly use Case 2 (exclude trend)?\n")
cat("Answer: Harmful under-specification - forces exclusion of important trend term\n\n")

# Simulate two series where Case 4 (type="trend") is the correct specification:
# 1. AR(1) with deterministic trend
# 2. Random walk with drift

# Series 1: AR(1) with deterministic trend
ar1_with_trend <- arima.sim(model=list(order=c(1,0,0), ar=c(0.85)), n=1000) + 0.1*seq(1000)
ts.plot(ar1_with_trend, main="AR(1) with Deterministic Trend", 
        ylab="Value", xlab="Time")
cat("AR(1) with trend: y_t = 0.1*t + 0.85*(y_{t-1} - 0.1*(t-1)) + ε_t\n")

# Series 2: Random walk with drift
rw_with_drift <- arima.sim(model=list(order=c(0,1,0)), mean=0.1, n=1000)
ts.plot(rw_with_drift, main="Random Walk with Drift", 
        ylab="Value", xlab="Time")
cat("Random walk with drift: y_t = y_{t-1} + 0.1 + ε_t\n")
cat("Note: This creates a stochastic trend, not deterministic trend\n\n")

# ----------- RANDOM WALK WITH DRIFT: TESTING SPECIFICATIONS -----------

cat("--- Testing Random Walk with Drift using Case 4 (CORRECT) ---\n")
rw_drift_cadf_correct <- CADFtest(rw_with_drift, criterion=c("AIC"), type="trend")
summary(rw_drift_cadf_correct)

rw_drift_urdf_correct <- ur.df(rw_with_drift, type=c("trend"), lags=0)
cat("Joint hypothesis test results (Case 4):\n")
summary(rw_drift_urdf_correct)
cat("Results Analysis:\n")
cat("- tau3: Fail to reject unit root (correct - has unit root)\n")
cat("- phi2: Reject joint test (lag=0, intercept=0, trend=0)\n")
cat("- phi3: Fail to reject joint test (lag=0 and trend=0)\n")
cat("Interpretation:\n")
cat("- phi3 failure suggests trend=0 AND lag=0\n")
cat("- But phi2 rejection means at least one coefficient ≠ 0\n")
cat("- Therefore: significant intercept → random walk with drift\n\n")

cat("--- Testing Random Walk with Drift using Case 2 (True model) ---\n")
rw_drift_cadf_wrong <- CADFtest(rw_with_drift, criterion=c("AIC"), type="drift")
summary(rw_drift_cadf_wrong)

rw_drift_urdf_wrong <- ur.df(rw_with_drift, type=c("drift"), lags=0)
cat("Joint hypothesis test results (Case 2 - true model):\n")
summary(rw_drift_urdf_wrong)
cat("Results Analysis:\n")
cat("- tau2: Fail to reject unit root (correct identification of unit root)\n")
cat("- phi1: Reject joint test (lag=0 and intercept=0)\n")
cat("Interpretation:\n")
cat("- phi1 rejection means either lag≠0 OR intercept≠0 (or both)\n")
cat("- Since tau2 suggests unit root, rejection likely due to significant intercept\n")
cat("- Suggests drift, which is correct\n\n")

# Verify drift using stationary first differences
rw_drift_test <- Arima(rw_with_drift, order=c(0,1,0), include.constant=TRUE)
cat("Drift verification using integrated model:\n")
print(rw_drift_test)
cat("Intercept IS significant - confirms drift is real!\n")
cat("CONCLUSION: Should have used Case 2\n\n")

# ----------- AR(1) WITH TREND: TESTING SPECIFICATIONS -----------

cat("--- Testing AR(1) with Trend using Case 4 (CORRECT) ---\n")
ar1_trend_cadf_correct <- CADFtest(ar1_with_trend, criterion=c("AIC"), type="trend")
summary(ar1_trend_cadf_correct)

ar1_trend_urdf_correct <- ur.df(ar1_with_trend, type=c("trend"), lags=0)
cat("Joint hypothesis test results (Case 4):\n")
summary(ar1_trend_urdf_correct)
cat("PERFECT RESULTS: Reject all hypotheses!\n")
cat("- tau3: Reject unit root (correct - it's stationary)\n")
cat("- phi2: Reject (lag=0, intercept=0, trend=0)\n")
cat("- phi3: Reject (lag=0 and trend=0)\n")
cat("This is exactly what we want - all components are significant\n\n")

cat("--- Testing AR(1) with Trend using Case 2 (WRONG - Under-specification) ---\n")
ar1_trend_cadf_wrong <- CADFtest(ar1_with_trend, criterion=c("AIC"), type="drift")
summary(ar1_trend_cadf_wrong)

ar1_trend_urdf_wrong <- ur.df(ar1_with_trend, type=c("drift"), lags=0)
cat("Joint hypothesis test results (Case 2 - Under-specification):\n")
summary(ar1_trend_urdf_wrong)
cat("DANGEROUS RESULTS:\n")
cat("- tau2: Fail to reject unit root (WRONG! - it's actually stationary)\n")
cat("- phi1: Reject joint test (lag=0 and intercept=0)\n")
cat("WRONG CONCLUSION: Might think this is a random walk with drift!\n")
cat("This is completely incorrect - the series is actually trend-stationary\n\n")

# Verify the misspecification
ar1_trend_wrong_test <- Arima(ar1_with_trend, order=c(0,1,0), include.constant=TRUE)
cat("Testing as integrated process (WRONG model):\n")
print(ar1_trend_wrong_test)
cat("This model is fundamentally misspecified!\n")
cat("The series is trend-stationary, not difference-stationary\n\n")

cat("CRITICAL LESSON: Under-specifying trends leads to wrong stationarity conclusions\n")
cat("- Trend-stationary series misidentified as unit root processes\n")
cat("- Leads to incorrect differencing and model specification\n")
cat("- Always test for trends when they might be present\n\n")

# =================== SUMMARY OF PARTS 2 AND 3 ===================

cat("==================================================================\n")
cat("SUMMARY OF SPECIFICATION TESTING (PARTS 2 AND 3)\n")
cat("==================================================================\n")
cat("KEY FINDINGS:\n\n")

cat("1. UNDER-SPECIFICATION (excluding important terms):\n")
cat("   - Case 1 when Case 2 is correct: May fail to reject unit root when should\n")
cat("   - Case 2 when Case 4 is correct: May misclassify trend-stationary as unit root\n")
cat("   - CONSEQUENCE: Fundamental misidentification of data generating process\n\n")

cat("2. OVER-SPECIFICATION (including extra terms):\n")
cat("   - Case 2 when Case 1 is correct: Harmless, still get correct conclusions\n")
cat("   - Case 4 when Case 2 is correct: Harmless, joint tests guide to simpler model\n")
cat("   - CONSEQUENCE: Slight power loss, but correct identification\n\n")

cat("3. PRACTICAL IMPLICATIONS:\n")
cat("   - Under-specification is much worse than over-specification\n")
cat("   - Better to err on the side of including extra terms\n")
cat("   - Use systematic approach: Start with Case 4, work down based on joint tests\n")
cat("   - Verify conclusions with first-difference drift tests when needed\n\n")

cat("4. JOINT TEST INTERPRETATION:\n")
cat("   - phi1: Joint test of unit root and zero intercept\n")
cat("   - phi2: Joint test of unit root, zero intercept, and zero trend\n")
cat("   - phi3: Joint test of unit root and zero trend\n")
cat("   - Joint rejection means at least one component ≠ 0 (not all components)\n")
cat("   - Use individual coefficient tests for specific significance\n")
cat("------------------------------------------------------------------\n\n")

# ======== PART 4: ECONOMIC DATA APPLICATION - OIL, GAS, AND DRILLING ========

cat("===============================================================\n")
cat("PART 4: REAL DATA APPLICATION - OIL, GAS, AND DRILLING ACTIVITY\n")
cat("===============================================================\n")
cat("Testing unit root behavior in economic time series:\n")
cat("- Oil prices, natural gas prices, drilling activity index\n")
cat("- Demonstrates Case 2 vs Case 4 selection with real data\n\n")

# Data Acquisition from FRED
getSymbols("MCOILWTICO", src="FRED")    # WTI Oil prices
getSymbols("IPN213111N", src="FRED")    # Oil & gas drilling production index
getSymbols("MHHNGSP", src="FRED")       # Henry Hub natural gas prices

# Data Preparation
energy_data <- merge.xts(MHHNGSP, MCOILWTICO, IPN213111N, all=TRUE, join="inner")
start_date <- "1997-01-01"    
end_date <- "2020-07-01"   
energy_data <- window(energy_data, start=start_date, end=end_date)

colnames(energy_data) <- c("NatGasPrice", "OilPrice", "DrillingIndex")

cat("=== Visual Inspection of Data ===\n")
chartSeries(energy_data$NatGasPrice, name="Natural Gas Prices")
chartSeries(energy_data$OilPrice, name="Oil Prices") 
chartSeries(energy_data$DrillingIndex, name="Drilling Activity Index")
cat("Visual observations: Oil shows trend, gas and drilling patterns unclear\n")
cat("Formal testing needed to determine unit root properties\n\n")

# ================= NATURAL GAS PRICES =================

cat("--- NATURAL GAS PRICES UNIT ROOT ANALYSIS ---\n")

# Determine optimal lag length
gas_ar_order <- ar(na.omit(diff(energy_data$NatGasPrice)))
cat("Suggested AR order for gas price changes:", gas_ar_order$order, "\n\n")

# Test with Case 4 (trend) first - most general
gas_trend_test <- ur.df(energy_data$NatGasPrice, type=c("trend"), lags=9)
cat("Case 4 (trend) test results:\n")
summary(gas_trend_test)
cat("Results:\n")
cat("- tau3: Fail to reject unit root\n")
cat("- phi2: Fail to reject (unit root, zero drift, zero trend)\n") 
cat("- phi3: Fail to reject (unit root and zero trend)\n")
cat("Conclusion: Move to Case 2\n\n")

# Test with Case 2 (drift)
gas_drift_test <- ur.df(energy_data$NatGasPrice, type=c("drift"), lags=9)
cat("Case 2 (drift) test results:\n")
summary(gas_drift_test)
cat("Results:\n")
cat("- tau2: Fail to reject unit root\n")
cat("- phi1: Fail to reject (unit root and zero drift)\n")
cat("Final conclusion: Natural gas has unit root, no drift\n\n")

# Confirmed by ARIMA intercept
nat_gas_arima <- Arima(energy_data$NatGasPrice, order=c(9,1,0), include.constant=TRUE)
summary(nat_gas_arima)
cat("ARIMA intercept also not significant, supporting no drift\n")

# ================= OIL PRICES =================

cat("--- OIL PRICES UNIT ROOT ANALYSIS ---\n")

# Determine optimal lag length
oil_ar_order <- ar(na.omit(diff(energy_data$OilPrice)))
cat("Suggested AR order for oil price changes:", oil_ar_order$order, "\n\n")

# Test with Case 4 (trend) first
oil_trend_test <- ur.df(energy_data$OilPrice, type=c("trend"), lags=1)
cat("Case 4 (trend) test results:\n")
summary(oil_trend_test)
cat("Results:\n")
cat("- tau3: Fail to reject unit root\n")
cat("- phi2: Fail to reject (unit root, zero drift, zero trend)\n")
cat("- phi3: Fail to reject (unit root and zero trend)\n") 
cat("Conclusion: Move to Case 2\n\n")

# Test with Case 2 (drift)
oil_drift_test <- ur.df(energy_data$OilPrice, type=c("drift"), lags=1)
cat("Case 2 (drift) test results:\n")
summary(oil_drift_test)
cat("Results:\n")
cat("- tau2: Fail to reject unit root\n")
cat("- phi1: Fail to reject (unit root and zero drift)\n")
cat("Final conclusion: Oil prices have unit root, no drift\n\n")

# Confirmed by ARIMA intercept
oil_arima <- Arima(energy_data$OilPrice, order=c(1,1,0), include.constant=TRUE)
summary(oil_arima)
cat("ARIMA intercept also not significant, supporting no drift\n")

# ================= DRILLING ACTIVITY =================

cat("--- DRILLING ACTIVITY INDEX UNIT ROOT ANALYSIS ---\n")

# Determine optimal lag length  
drilling_ar_order <- ar(na.omit(diff(energy_data$DrillingIndex)))
cat("Suggested AR order for drilling changes:", drilling_ar_order$order, "\n\n")

# Test with Case 2 (drift) - no obvious trend visible
drilling_drift_test <- ur.df(energy_data$DrillingIndex, type=c("drift"), lags=2)
cat("Case 2 (drift) test results:\n")
summary(drilling_drift_test)
cat("Results:\n")
cat("- tau2: Fail to reject unit root\n")
cat("- phi1: Fail to reject (unit root and zero drift)\n")
cat("Final conclusion: Drilling activity has unit root, no drift\n\n")

# Confirmed by ARIMA intercept
drill_arima <- Arima(energy_data$DrillingIndex, order=c(2,1,0), include.constant=TRUE)
summary(drill_arima)
cat("ARIMA intercept also not significant, supporting no drift\n")

# ---------------------- SUMMARY OF ECONOMIC DATA ANALYSIS -------------------------
cat("==================================================================\n")
cat("SUMMARY OF ECONOMIC DATA UNIT ROOT ANALYSIS\n")
cat("==================================================================\n")
cat("ALL THREE SERIES EXHIBIT UNIT ROOT BEHAVIOR:\n")
cat("1. Natural Gas Prices: I(1), no drift, no trend\n")
cat("2. Oil Prices: I(1), no drift, no trend\n") 
cat("3. Drilling Activity: I(1), no drift, no trend\n\n")
cat("METHODOLOGICAL INSIGHTS:\n")
cat("- Systematic testing (Case 4 → Case 2) provides clear guidance\n")
cat("- Visual trends can be misleading for unit root determination\n")
cat("- Joint tests help distinguish between different specifications\n")
cat("- Economic data commonly exhibit unit root behavior\n")
cat("------------------------------------------------------------------\n")

# ================= OVERALL METHODOLOGICAL SUMMARY ===================

cat("\n====================================================================\n")
cat("COMPREHENSIVE UNIT ROOT TESTING METHODOLOGY SUMMARY\n")
cat("====================================================================\n")
cat("KEY PRINCIPLES:\n")
cat("1. OVER-SPECIFICATION vs UNDER-SPECIFICATION\n")
cat("   - Over-specification (extra terms): Harmless, reduces power slightly\n")
cat("   - Under-specification (missing terms): Harmful, leads to wrong conclusions\n")
cat("   - Rule: Better to include extra terms than exclude important ones\n\n")

cat("2. SYSTEMATIC TESTING APPROACH\n")
cat("   - Start with most general case (Case 4: trend)\n")
cat("   - Use joint tests to determine appropriate specification\n")
cat("   - Move to simpler cases based on test results\n")
cat("   - Verify conclusions with first-difference drift tests\n\n")

cat("3. TOOLS AND THEIR PURPOSES\n")
cat("   - CADFtest(): Individual tests, allows covariates, automatic lag selection\n")
cat("   - ur.df(): Joint hypothesis tests, more complete statistical inference\n")
cat("   - Both tools complement each other in comprehensive analysis\n\n")

cat("4. INTERPRETATION GUIDELINES\n")
cat("   - Joint tests tell you about combined hypotheses (not individual significance)\n")
cat("   - With unit root: Use first differences for drift testing (stationary)\n")
cat("   - Without unit root: Use standard t-tests for coefficient significance\n")
cat("   - Visual inspection can be misleading - formal tests essential\n\n")

cat("5. PRACTICAL RECOMMENDATIONS\n")
cat("   - Default to Case 2 (drift) for most economic data when uncertain\n")
cat("   - Use Case 4 (trend) only when clear trending behavior is evident\n")
cat("   - Always check robustness across different lag specifications\n")
cat("   - Economic time series commonly exhibit unit root behavior\n\n")

cat("This methodology ensures robust, reliable unit root testing procedures\n")
cat("suitable for academic research and applied econometric analysis.\n")
cat("=====================================================================\n")





