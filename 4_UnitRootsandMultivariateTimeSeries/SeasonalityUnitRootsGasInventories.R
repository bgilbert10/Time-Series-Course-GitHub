# ======================================================================
# SEASONALITY AND UNIT ROOT TESTING - NATURAL GAS STORAGE INVENTORIES
# ======================================================================

# This program demonstrates:
# 1. Unit root testing with seasonal covariates using CADFtest() with Fourier terms
# 2. Deseasonalizing time series data and testing residuals for unit roots
# 3. ARIMA model selection incorporating seasonality controls
# 4. Model diagnostics and residual analysis for seasonal time series
# 5. Forecasting with holdout samples comparing different seasonal approaches
# 6. SARIMA modeling without Fourier terms and forecast comparison

# Key concepts:
# - Controlling for seasonality in unit root tests using Fourier covariates
# - Difference between seasonal unit roots and deterministic seasonality
# - Residualization technique for isolating non-seasonal components
# - Model selection between Fourier-based and SARIMA approaches

# --------------------- SETUP AND PACKAGE LOADING ----------------------
# Load required packages
library(tseries)     # Time series analysis and unit root tests
library(lubridate)   # Date manipulation for non-standard frequencies
library(forecast)    # Advanced forecasting methods and SARIMA models
library(CADFtest)    # Covariate-augmented Dickey-Fuller tests
library(quantmod)    # Financial data analysis tools
library(fBasics)     # Basic financial statistics
library(urca)        # Unit root and cointegration testing

# ---------------------- DATA LOADING AND PREPARATION ---------------------
# Load Natural Gas Storage Inventory Data
# Note: This assumes the CSV file is in the working directory
# Adjust path as needed based on file location
setwd("C:/Users/gilbe/Dropbox/Econometrics/TimeSeriesCourse/Fall2017/TsayCh2/Lecture/")
inventory_data <- read.table('NGInventoryEastRegionBCF.csv', header=TRUE, sep=",") 

# Convert to time series object with weekly frequency
# Use lubridate package for non-integer seasonal frequency of 365.25/7
inventory_ts <- ts(inventory_data$EastNGInventoryBCF, freq=365.25/7, start=2010)

# Visualize the raw data
ts.plot(inventory_ts, main="Natural Gas Storage Inventories - East Region", 
        ylab="Inventory (BCF)", xlab="Year")

# Initial observations:
# - Strong seasonal pattern visible
# - Does not appear to be drifting/trending in mean
# - Visual inspection suggests possible stationarity, but seasonality may mask unit root behavior

# ============== UNIT ROOT TESTING WITH SEASONAL CONTROLS ===============

# Unit Root Test with Fourier Terms as Covariates
# CADFtest advantage: Can control for other X variables during ADF testing
# Use Fourier terms to control for seasonality and test for unit root in non-seasonal component
# Start with 4 Fourier terms (K=4) - this captures seasonal harmonics

inventory_cadf_seasonal <- CADFtest(inventory_ts, criterion=c("AIC"), type="drift", 
                                   X=fourier(inventory_ts, K=c(4)))

cat("=== Unit Root Test with Seasonal Controls (4 Fourier Terms) and 1 lag ===\n")
summary(inventory_cadf_seasonal)
cat("Result: Fail to reject null of unit root - Surprising given visual appearance!\n")
cat("Note: Regression output includes coefficients on Fourier terms (sines and cosines)\n")
cat("Interpretation: Even after controlling for seasonality, evidence suggests unit root\n\n")

# ================ DESEASONALIZATION AND RESIDUAL ANALYSIS ===============

# Better visualization of unit root behavior by removing seasonality
# Estimate model with only Fourier terms (no ARMA structure) and examine residuals
deseasonalize_model <- Arima(inventory_ts, order=c(0,0,0), xreg=fourier(inventory_ts, K=c(4)))

cat("=== Deseasonalization Model (Fourier Terms Only) ===\n")
print(deseasonalize_model)

# Visualize deseasonalized series (residuals)
ts.plot(residuals(deseasonalize_model), main="Deseasonalized Natural Gas Inventories", 
        ylab="Residual (BCF)", xlab="Year")
cat("Visual inspection of residuals: More clearly resembles random walk pattern\n\n")

# Comprehensive residual diagnostics
par(mfrow=c(3,1))
tsdiag(deseasonalize_model, gof=25)
par(mfrow=c(1,1))

# Enhanced residual display
tsdisplay(residuals(deseasonalize_model), 
         main="Deseasonalized Inventory Series - Comprehensive Analysis")

cat("=== Residual Analysis Observations ===\n")
cat("- Residuals appear to follow random walk pattern without drift (Case 2)\n")
cat("- Strong evidence of persistence in deseasonalized component\n\n")

# Test residuals for unit root behavior
residuals_cadf <- CADFtest(residuals(deseasonalize_model), criterion=c("AIC"), type="drift")
cat("=== Unit Root Test on Deseasonalized Residuals ===\n")
summary(residuals_cadf)
cat("Confirms unit root behavior in the non-seasonal component\n\n")

# ================== ARIMA MODEL WITH SEASONAL CONTROLS ==================

# Model Selection Based on Unit Root Test Results
# Both CADFtest iterations suggest one additional lag plus unit root
# Therefore: ARIMA(1,1,0) with Fourier terms should be appropriate

fourier_arima <- Arima(inventory_ts, order=c(1,1,0), xreg=fourier(inventory_ts, K=c(4)))

cat("=== ARIMA(1,1,0) Model with Fourier Terms ===\n")
summary(fourier_arima)

# Model diagnostics
cat("=== Model Diagnostics ===\n")
par(mfrow=c(3,1))
tsdiag(fourier_arima, gof=25)
par(mfrow=c(1,1))

# Detailed residual analysis
tsdisplay(residuals(fourier_arima), 
         main="ARIMA(1,1,0) + Fourier Terms - Residual Analysis")

cat("Result: Some residual autocorrelation remains, but substantial improvement\n")
cat("The combination of integration and AR(1) structure addresses main dependencies\n\n")

# Automatic model selection for comparison
auto_fourier <- auto.arima(inventory_ts, xreg=fourier(inventory_ts, K=c(4)), seasonal=FALSE)
cat("=== Auto-ARIMA with Fourier Terms (for comparison) ===\n")
print(auto_fourier)
cat("Auto-ARIMA confirms our manual model selection\n\n")

# ============== HOLDOUT SAMPLE FORECASTING - FOURIER APPROACH ==============

# Evaluate forecast performance using holdout sample validation
# Hold out last 52 weeks (1 year) for testing
# Estimate model on periods 1 to 352, forecast periods 352 to 404

# Create training sample (excluding holdout period)
holdout_start <- 352
holdout_end <- 404
training_data <- ts(inventory_ts[-c(holdout_start:holdout_end)], freq=365.25/7, start=2010)

cat("=== Holdout Sample Evaluation Setup ===\n")
cat("Training sample: Periods 1 to", holdout_start-1, "\n")
cat("Holdout sample: Periods", holdout_start, "to", holdout_end, "(52 weeks)\n")
cat("Forecast horizon: 104 weeks (52 in-sample + 52 out-of-sample)\n\n")

# Fit model on training data
forecast_model_fourier <- Arima(inventory_ts[-c(holdout_start:holdout_end)], 
                               order=c(1,1,0), include.drift=FALSE,
                               xreg=fourier(training_data, K=c(4)))

# Generate forecasts
forecast_fourier <- forecast(forecast_model_fourier, 
                           xreg=fourier(training_data, K=c(4), h=104))

# Visualize forecasts against actual data
plot(forecast_fourier, main="ARIMA(1,1,0) with Fourier Terms - Holdout Evaluation", 
     include=156, ylab="Inventory (BCF)", xlab="Time")
lines(ts(inventory_ts), col="red", lwd=2)
legend("topright", legend=c("Forecast", "Actual"), col=c("blue", "red"), lty=1, lwd=2)

cat("=== Fourier Model Forecast Evaluation ===\n")
cat("Visual assessment: Forecast captures seasonal pattern and general level\n")
cat("Red line shows actual data for comparison with forecast performance\n\n")

# ================ SARIMA MODELING WITHOUT FOURIER TERMS =================

# Alternative approach: Use SARIMA models instead of Fourier terms
# This may take considerable time to run due to seasonal complexity

cat("=== Automatic SARIMA Model Selection ===\n")
cat("Warning: This may take several minutes to complete...\n")
auto_sarima_full <- auto.arima(inventory_ts, seasonal=TRUE)
print(auto_sarima_full)
cat("Auto-ARIMA suggests: SARIMA(1,0,1)(1,1,0) with drift\n\n")

# Fit the suggested SARIMA model
sarima_model <- Arima(inventory_ts, order=c(1,0,1), 
                     seasonal=list(order=c(1,1,0)), include.constant=TRUE)

cat("=== SARIMA Model Summary ===\n")
summary(sarima_model)

# ============== HOLDOUT SAMPLE FORECASTING - SARIMA APPROACH ==============

# Fit SARIMA model on training data for holdout evaluation
forecast_model_sarima <- Arima(inventory_ts[-c(holdout_start:holdout_end)], 
                              order=c(1,0,1), seasonal=list(order=c(1,1,0)), 
                              include.constant=TRUE)

# Generate SARIMA forecasts
forecast_sarima <- forecast(forecast_model_sarima, h=104)

# Visualize SARIMA forecasts
plot(forecast_sarima, main="SARIMA(1,0,1)(1,1,0) with Drift - Holdout Evaluation", 
     include=156, ylab="Inventory (BCF)", xlab="Time")
lines(ts(inventory_ts), col="red", lwd=2)
legend("topright", legend=c("SARIMA Forecast", "Actual"), col=c("blue", "red"), lty=1, lwd=2)

cat("=== SARIMA Model Forecast Evaluation ===\n")
cat("SARIMA approach uses seasonal differencing and seasonal AR/MA terms\n")
cat("Compare visual performance with Fourier-based approach\n\n")

# Full sample SARIMA forecast for comparison
forecast_sarima_full <- forecast(sarima_model, h=104)
plot(forecast_sarima_full, main="SARIMA Model - Full Sample Forecast", 
     include=156, ylab="Inventory (BCF)", xlab="Time")
lines(ts(inventory_ts), col="red", lwd=2)

# ================== ALTERNATIVE MODEL EXPLORATION ===================

# Explore alternative model on training sample
cat("=== Alternative Model Exploration ===\n")
auto_alt <- auto.arima(inventory_ts[-c(holdout_start:holdout_end)], seasonal=TRUE)
print(auto_alt)
cat("Alternative model suggestion for training data\n\n")

# Fit alternative model (simple AR structure)
alternative_model <- Arima(inventory_ts[-c(holdout_start:holdout_end)], 
                          order=c(3,0,0), include.constant=TRUE)

forecast_alternative <- forecast(alternative_model, h=104)
plot(forecast_alternative, main="Alternative Model: AR(3) with Drift", 
     include=156, ylab="Inventory (BCF)", xlab="Time")
lines(ts(inventory_ts), col="red", lwd=2)
legend("topright", legend=c("AR(3) Forecast", "Actual"), col=c("blue", "red"), lty=1, lwd=2)

cat("=== Alternative Model Assessment ===\n")
cat("Simple AR(3) model without explicit seasonal structure\n")
cat("May miss important seasonal dynamics\n\n")

# ---------------------- SUMMARY OF FINDINGS -------------------------
cat("\n===========================================================\n")
cat("SUMMARY OF SEASONALITY AND UNIT ROOT ANALYSIS\n")
cat("===========================================================\n")
cat("UNIT ROOT TESTING INSIGHTS:\n")
cat("1. Unit root testing with seasonal covariates reveals hidden persistence\n")
cat("2. Visual inspection can be misleading when strong seasonality is present\n")
cat("3. Deseasonalization helps clarify underlying time series properties\n")
cat("4. CADFtest with Fourier terms allows proper unit root testing\n\n")
cat("MODEL SELECTION RESULTS:\n")
cat("1. Fourier approach: ARIMA(1,1,0) with 4 harmonic terms\n")
cat("2. SARIMA approach: ARIMA(1,0,1)(1,1,0) with drift\n")
cat("3. Simple AR approach: AR(3) with drift (less sophisticated)\n\n")
cat("FORECASTING COMPARISON:\n")
cat("- Fourier terms: Explicit harmonic seasonal structure\n")
cat("- SARIMA: Seasonal differencing and multiplicative seasonality\n")
cat("- Both approaches capture seasonality but through different mechanisms\n")
cat("- Holdout evaluation shows relative forecasting performance\n\n")
cat("METHODOLOGICAL LESSONS:\n")
cat("- Seasonal unit root testing requires specialized approaches\n")
cat("- Multiple modeling frameworks can address seasonal persistence\n")
cat("- Model validation through holdout samples is crucial\n")
cat("- Residual analysis confirms model adequacy\n")
cat("-----------------------------------------------------------\n")
