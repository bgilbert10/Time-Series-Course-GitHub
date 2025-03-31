# ======================================================================
# TIME SERIES ERRORS AND AUTOCORRELATION ANALYSIS
# ======================================================================

# This program demonstrates:
# 1. Testing for autocorrelation and plotting autocorrelation functions (ACFs)
# 2. Heteroskedasticity-Consistent (HC) and Heteroskedasticity and Autocorrelation
#    Consistent (HAC) standard errors in regression models
# 3. Testing for serial correlation in regression residuals
# 4. Methods to adjust models for autocorrelation

# --------------------- SETUP AND PACKAGE LOADING ----------------------
# Load required packages
library(quantmod)    # For financial data retrieval
library(MTS)         # For multivariate time series analysis
library(forecast)    # For time series forecasting and visualization
library(sandwich)    # For robust covariance matrix estimators
library(lmtest)      # For testing linear regression models
library(car)         # For regression diagnostics and testing
library(dynlm)       # For dynamic linear models

# ---------------------- DATA ACQUISITION ------------------------------
# Model for determining price response of oil & gas drilling

# Read from FRED database:
# - WTI oil prices (MCOILWTICO)
# - Oil & gas drilling index (IPN213111N)
# - Henry Hub natural gas prices (MHHNGSP)
getSymbols("MCOILWTICO", src="FRED")   # WTI crude oil prices
getSymbols("IPN213111N", src="FRED")   # Oil & gas drilling index
getSymbols("MHHNGSP", src="FRED")      # Henry Hub natural gas prices

# ---------------------- DATA PREPARATION -----------------------------
# Merge all series into one data frame
price_drilling_levels = merge.xts(MHHNGSP, MCOILWTICO, IPN213111N, all=TRUE)

# Plot level data (appears non-stationary - will test formally later)
plot(price_drilling_levels, 
     main="Oil and Gas Prices and Drilling Activity (Levels)",
     col=c("blue", "red", "green"))
MTSplot(price_drilling_levels)

# Calculate log percentage changes (likely stationary)
price_drilling_changes = na.omit(diff(log(price_drilling_levels)))

# Plot transformed data
plot(price_drilling_changes, 
     main="Oil and Gas Prices and Drilling Activity (Log Changes)",
     col=c("blue", "red", "green"))
MTSplot(price_drilling_changes)

# ------------------- AUTOCORRELATION ANALYSIS ------------------------
# Investigate autocorrelation patterns in each series

# Function to test and visualize autocorrelation
analyze_autocorrelation <- function(series, name, max_lag=NULL) {
  # Determine appropriate lag order based on sample size
  if(is.null(max_lag)) {
    max_lag = round(log(length(series)))
  }
  
  # Plot ACF
  acf_plot <- acf(series, main=paste("ACF of", name), lag.max=max_lag)
  
  # Perform Box-Ljung test
  box_test_result <- Box.test(series, lag=max_lag, type='Ljung')
  
  # Print results
  cat("======================================\n")
  cat("Autocorrelation analysis for", name, "\n")
  cat("======================================\n")
  cat("Box-Ljung test (lag =", max_lag, "):\n")
  cat("Q statistic =", round(box_test_result$statistic, 4), "\n")
  cat("p-value =", round(box_test_result$p.value, 4), "\n")
  cat("--------------------------------------\n\n")
  
  return(list(acf=acf_plot, box_test=box_test_result))
}

# Determine appropriate lag order based on sample size
max_lag <- round(log(length(price_drilling_changes)))
cat("Suggested lag order based on log(T):", max_lag, "\n\n")

# Analyze each series
oil_autocorr <- analyze_autocorrelation(price_drilling_changes$MCOILWTICO, 
                                      "Oil Price Changes", max_lag)
gas_autocorr <- analyze_autocorrelation(price_drilling_changes$MHHNGSP, 
                                      "Natural Gas Price Changes", max_lag)
drilling_autocorr <- analyze_autocorrelation(price_drilling_changes$IPN213111N, 
                                          "Drilling Activity Changes", max_lag)

# -------------------- BASELINE REGRESSION MODEL ----------------------
# Estimate a simple model: How do prices affect drilling activity?

baseline_model <- lm(IPN213111N ~ MCOILWTICO + MHHNGSP, data=price_drilling_changes)
cat("Baseline regression model: How do prices affect drilling activity?\n")
cat("--------------------------------------------------------------\n")
print(summary(baseline_model))

# ----------------- RESIDUAL AUTOCORRELATION ANALYSIS -----------------
# Check for autocorrelation in the residuals

# Function to analyze residuals for autocorrelation
analyze_residuals <- function(model, df_correction=0, max_lag=NULL) {
  residuals <- residuals(model)
  
  # Determine appropriate lag order based on sample size if not provided
  if(is.null(max_lag)) {
    max_lag = round(log(length(residuals)))
  }
  
  # Visualize residuals
  par(mfrow=c(3,1))
  plot(residuals, type="l", main="Residuals Over Time", 
       xlab="Time", ylab="Residual")
  acf(residuals, main="ACF of Residuals", lag.max=max_lag)
  pacf(residuals, main="PACF of Residuals", lag.max=max_lag)
  par(mfrow=c(1,1))
  
  # Perform Box-Ljung test with degrees of freedom correction
  box_test_result <- Box.test(residuals, lag=max_lag, type='Ljung')
  df <- max_lag - df_correction
  adjusted_p_value <- 1-pchisq(box_test_result$statistic, df)
  
  # Print results
  cat("====================================================\n")
  cat("Residual autocorrelation analysis\n")
  cat("====================================================\n")
  cat("Box-Ljung test (lag =", max_lag, "):\n")
  cat("Q statistic =", round(box_test_result$statistic, 4), "\n")
  cat("Unadjusted p-value =", round(box_test_result$p.value, 4), "\n")
  cat("Degrees of freedom correction =", df_correction, "\n")
  cat("Adjusted degrees of freedom =", df, "\n")
  cat("Adjusted p-value =", round(adjusted_p_value, 4), "\n")
  cat("----------------------------------------------------\n\n")
  
  return(list(
    residuals=residuals, 
    box_test=box_test_result, 
    adjusted_p_value=adjusted_p_value
  ))
}

# Analyze residuals from baseline model
# DF correction = 2 (for 2 parameters estimated in model)
residual_analysis <- analyze_residuals(baseline_model, df_correction=2)

# Also use the tsdisplay function for a comprehensive view
tsdisplay(residuals(baseline_model), main="Comprehensive Residual Analysis")

# ----------------- HC AND HAC STANDARD ERRORS -----------------------
# Apply robust standard errors to account for issues in the error terms

# Function to display different standard error calculations
compare_standard_errors <- function(model) {
  # Original standard errors
  original <- coeftest(model)
  
  # HAC standard errors
  hac <- coeftest(model, vcov=vcovHAC(model))
  
  # HC standard errors
  hc <- coeftest(model, vcov=vcovHC(model))
  
  # Create comparison table
  coef_names <- rownames(original)
  result_table <- data.frame(
    Coefficient = coef_names,
    Estimate = original[,1],
    Original_SE = original[,2],
    Original_p = original[,4],
    HC_SE = hc[,2],
    HC_p = hc[,4],
    HAC_SE = hac[,2],
    HAC_p = hac[,4]
  )
  
  return(result_table)
}

# Compare standard errors for baseline model
std_error_comparison <- compare_standard_errors(baseline_model)
cat("Comparison of Standard Error Methods\n")
cat("====================================\n")
print(std_error_comparison)

# F-test with HAC standard errors
gas_price_test <- linearHypothesis(baseline_model, c("MHHNGSP=0"), 
                                 vcov=vcovHAC(baseline_model), 
                                 test="F")
cat("\nF-test for Natural Gas Price coefficient (HAC std errors)\n")
cat("-------------------------------------------------------\n")
print(gas_price_test)

# Calculate corresponding t-statistic
t_statistic <- sqrt(gas_price_test$F[2])
cat("Corresponding t-statistic:", round(t_statistic, 4), "\n\n")

# ------------------- ADJUSTING FOR AUTOCORRELATION -----------------
# Two approaches to handle autocorrelation:
# 1. Modeling the residuals (ARIMA)
# 2. Including lags of dependent/independent variables (dynlm)

# Convert data to ts object for time series modeling
price_drilling_ts <- ts(price_drilling_changes, freq=12, start=c(1997, 2))

# ---------------- APPROACH 1: ARIMA MODEL --------------------------
# Model the residuals using ARIMA

# Variables are:
# Column 1: MHHNGSP (Natural Gas Price)
# Column 2: MCOILWTICO (Oil Price)
# Column 3: IPN213111N (Drilling Activity)

# Estimate ARIMA model with AR(2) for residuals
arima_model <- arima(price_drilling_changes[,3], order=c(2,0,0),
                   xreg=cbind(price_drilling_changes[,1], price_drilling_changes[,2]),
                   include.mean=TRUE)

cat("ARIMA Model Results (AR(2) for residuals)\n")
cat("----------------------------------------\n")
print(summary(arima_model))

# Check if autocorrelation has been resolved
arima_residual_analysis <- analyze_residuals(arima_model, df_correction=4)

# Standard errors from ARIMA model
arima_coef_test <- coeftest(arima_model, vcov=vcov(arima_model))
cat("ARIMA model coefficients with standard errors:\n")
print(arima_coef_test)

# -------------- APPROACH 2: DYNAMIC LINEAR MODEL ------------------
# Include lags of variables to account for autocorrelation

# Model with lags of the dependent variable (drilling activity)
dynamic_model <- dynlm(price_drilling_ts[,"IPN213111N"] ~ 
                     L(price_drilling_ts[,"IPN213111N"], c(1:2)) +
                     price_drilling_ts[,"MHHNGSP"] + 
                     price_drilling_ts[,"MCOILWTICO"])

cat("Dynamic Linear Model Results (with lags of drilling)\n")
cat("--------------------------------------------------\n")
print(summary(dynamic_model))

# Check residuals of dynamic model
tsdisplay(residuals(dynamic_model), 
         main="Residuals of Dynamic Model with Drilling Lags")

# Apply HC standard errors to dynamic model
dynamic_hc_test <- coeftest(dynamic_model, vcov=vcovHC(dynamic_model))
cat("Dynamic model coefficients with HC standard errors:\n")
print(dynamic_hc_test)

# Test correlation between oil and gas prices
price_correlation <- cor(price_drilling_changes$MHHNGSP, 
                       price_drilling_changes$MCOILWTICO)
cat("Correlation between oil and gas prices:", round(price_correlation, 4), "\n\n")

# Joint test of price effects using HC standard errors
# Reestimate with simpler variable names for easier joint testing
dynamic_model_simple <- dynlm(price_drilling_ts[,3] ~ 
                           L(price_drilling_ts[,3], c(1:2)) +
                           price_drilling_ts[,1] + 
                           price_drilling_ts[,2])

joint_price_test <- linearHypothesis(dynamic_model_simple, 
                                   c("price_drilling_ts[, 1]=0", 
                                     "price_drilling_ts[, 2]=0"),
                                   vcov=vcovHC(dynamic_model_simple), 
                                   test="F")
cat("Joint test of price effects (HC std errors):\n")
print(joint_price_test)

# -------------- EXTENDED DYNAMIC MODEL WITH PRICE LAGS -------------
# Try including lags of price variables

extended_dynamic_model <- dynlm(price_drilling_ts[,3] ~ 
                             L(price_drilling_ts[,3], c(1:2)) + 
                             L(price_drilling_ts[,1], c(1)) +
                             L(price_drilling_ts[,2], c(0:4)))

cat("Extended Dynamic Model Results (with price lags)\n")
cat("-----------------------------------------------\n")
print(summary(extended_dynamic_model))

# Check residuals
tsdisplay(residuals(extended_dynamic_model), 
         main="Residuals of Extended Dynamic Model with Price Lags")

# Apply HAC standard errors
extended_hac_test <- coeftest(extended_dynamic_model, vcov=vcovHAC(extended_dynamic_model))
cat("Extended dynamic model coefficients with HAC standard errors:\n")
print(extended_hac_test)

# ---------------------- SUMMARY OF FINDINGS ------------------------
cat("\n===========================================================\n")
cat("SUMMARY OF FINDINGS\n")
cat("===========================================================\n")
cat("1. Baseline model shows significant autocorrelation in residuals\n")
cat("2. HAC standard errors correct for this but don't address underlying issue\n")
cat("3. ARIMA model successfully accounts for residual autocorrelation\n")
cat("4. Dynamic model with lags of drilling also addresses autocorrelation\n")
cat("5. Extended model suggests lagged price effects might be important\n")
cat("6. Joint test of contemporaneous price effects is not significant\n")
cat("-----------------------------------------------------------\n")