# ARMA Modeling with Gold Prices and Macroeconomic Variables ------------------
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

# Load required packages
library(forecast)    # Time series forecasting and ARIMA models
library(quantmod)    # Financial data retrieval and manipulation
library(caschrono)   # Additional time series analysis functions
library(texreg)      # Regression output formatting
library(sandwich)    # Heteroskedasticity and autocorrelation consistent (HAC) standard errors
library(lmtest)      # Linear model diagnostic tests


# Step 1: Data Acquisition and Preparation ------------------------------------
# 
# Download time series data from multiple sources and merge into a single dataset.
# We use different data providers: Yahoo Finance for gold prices and FRED for
# macroeconomic variables.

# Download financial time series data
getSymbols("GC=F")                          # Gold futures from Yahoo Finance
getSymbols("DFII10", src="FRED")            # 10-year TIPS constant maturity
getSymbols("T10YIE", src="FRED")            # 10-year breakeven inflation rate
getSymbols("DTWEXBGS", src="FRED")          # Trade-weighted US dollar index

# Merge all time series into a single xts object
# Use adjusted closing prices for gold futures
gold_macro_data <- merge.xts(`GC=F`$`GC=F.Adjusted`, DFII10$DFII10, 
                             T10YIE$T10YIE, DTWEXBGS,
                             all=TRUE, fill=NA, retclass="xts")

# Assign meaningful column names
colnames(gold_macro_data) <- c("gold_price", "tips_yield", "breakeven_inflation", "trade_weighted_dollar")

# Define analysis period
# Note: This period includes the 2008 financial crisis and subsequent recovery
start_date <- "2008-01-03"    
end_date   <- "2020-06-30"    

# Extract data for specified time window and remove missing observations
gold_macro_data <- window(gold_macro_data, start = start_date, end = end_date)
gold_macro_data <- na.omit(gold_macro_data)

# Display basic information about the dataset
# Dataset Summary: Analysis period and variables overview
cat("Dataset Summary: Time period:", start_date, "to", end_date, 
    "| Observations:", nrow(gold_macro_data), "\n")


# Step 2: Exploratory Data Analysis - Stationarity Investigation --------------
#
# Examine the time series properties of each variable in levels.
# Non-stationary series typically show:
# - Persistent trends or random walks
# - ACF that decays very slowly
# - High autocorrelation at many lags
#
# Stationary series show:
# - Constant mean and variance over time
# - ACF that decays quickly to zero
# - Mean reversion properties

# Visual inspection of time series in levels

# Gold prices - typically non-stationary (random walk with drift)
chartSeries(gold_macro_data$gold_price, 
           main="Gold Futures Prices (Levels)",
           theme="white")
acf(gold_macro_data$gold_price, 
    main="ACF: Gold Prices (Levels)",
    lag.max=50)

# TIPS yields - interest rates can be persistent but may revert to mean
chartSeries(gold_macro_data$tips_yield,
           main="10-Year TIPS Yields (Levels)", 
           theme="white")
acf(gold_macro_data$tips_yield,
    main="ACF: TIPS Yields (Levels)",
    lag.max=50)

# Breakeven inflation - inflation expectations tend to be persistent
chartSeries(gold_macro_data$breakeven_inflation,
           main="10-Year Breakeven Inflation (Levels)",
           theme="white")
acf(gold_macro_data$breakeven_inflation,
    main="ACF: Breakeven Inflation (Levels)",
    lag.max=50)

# Trade-weighted dollar - exchange rate indices often non-stationary
chartSeries(gold_macro_data$trade_weighted_dollar,
           main="Trade-Weighted Dollar Index (Levels)",
           theme="white")
acf(gold_macro_data$trade_weighted_dollar,
    main="ACF: Trade-Weighted Dollar (Levels)",
    lag.max=50)

# Interpretation: If ACF declines slowly and remains significant at many lags,
# the series is likely non-stationary and requires differencing.


# Step 3: ARIMA Model Identification for Differenced Series ------------------
#
# Transform non-stationary series to achieve stationarity through differencing.
# For financial returns, we typically use log differences (continuously compounded returns).
# For variables already in percentage terms, simple differences are appropriate.

# Creating stationary transformations for modeling

# Calculate returns/changes for each series
# Gold: Use log differences to get continuously compounded returns
gold_returns <- ts(na.omit(diff(log(gold_macro_data$gold_price))))

# Other variables: Use simple differences (already in percentage/index form)
tips_changes <- ts(na.omit(diff(gold_macro_data$tips_yield)))
inflation_changes <- ts(na.omit(diff(gold_macro_data$breakeven_inflation)))  
dollar_changes <- ts(na.omit(diff(gold_macro_data$trade_weighted_dollar)))

# ARIMA Model Selection for Each Series ---------------------------------------

## Gold Returns Analysis - Testing different model selection approaches

# Method 1: armaselect() - uses BIC criterion (more parsimonious)
gold_armaselect <- armaselect(gold_returns)
print(head(gold_armaselect, 5))

# Method 2: auto.arima() - comprehensive automated selection
gold_auto <- auto.arima(gold_returns)
print(gold_auto)

# Method 3: ar() - pure autoregressive approach
gold_ar <- ar(gold_returns)
cat("Selected AR order:", gold_ar$order, "\n")

# Visual diagnostics
tsdisplay(gold_returns, main="Gold Returns: ACF, PACF, and Time Series")

# Interpretation: If all methods suggest AR(0) or MA(0), gold returns appear
# to be white noise - unpredictable, which supports market efficiency.

## ------------------- TIPS YIELD CHANGES ANALYSIS -------------------
cat("--- TIPS YIELD CHANGES ANALYSIS ---\n")

# Compare different model selection approaches
cat("armaselect() result:\n")
tips_armaselect <- armaselect(tips_changes)
print(head(tips_armaselect, 5))

cat("\nauto.arima() result:\n") 
tips_auto <- auto.arima(tips_changes)
print(tips_auto)

cat("\nar() result:\n")
tips_ar <- ar(tips_changes)
cat("Selected AR order:", tips_ar$order, "\n")

# Model diagnostics comparison
cat("\nComparing model diagnostics:\n")
cat("ARMA(1,2) model diagnostics:\n")
tips_arma12 <- arima(tips_changes, order=c(1,0,2), include.mean=FALSE)
tsdiag(tips_arma12, gof=15)

cat("AR(", tips_ar$order, ") model diagnostics:\n")
tips_ar_model <- arima(tips_changes, order=c(tips_ar$order,0,0), include.mean=FALSE)
tsdiag(tips_ar_model, gof=15)

cat("INTERPRETATION: Higher-order AR models may be needed to capture all\n")
cat("autocorrelation in interest rate changes.\n\n")

## ------------------- BREAKEVEN INFLATION CHANGES ANALYSIS -------------------
cat("--- BREAKEVEN INFLATION CHANGES ANALYSIS ---\n")

cat("armaselect() result:\n")
inflation_armaselect <- armaselect(inflation_changes)
print(head(inflation_armaselect, 5))

cat("\nauto.arima() result:\n")
inflation_auto <- auto.arima(inflation_changes)
print(inflation_auto)

cat("\nar() result:\n")
inflation_ar <- ar(inflation_changes)
cat("Selected AR order:", inflation_ar$order, "\n")

# Compare top candidate models
cat("\nComparing candidate models:\n")
inflation_ar1 <- arima(inflation_changes, order=c(1,0,0), include.mean=FALSE)
inflation_ma1 <- arima(inflation_changes, order=c(0,0,1), include.mean=FALSE)
inflation_ar2 <- arima(inflation_changes, order=c(2,0,0), include.mean=FALSE)

tsdiag(inflation_ar1, gof=15)
tsdiag(inflation_ma1, gof=15) 
tsdiag(inflation_ar2, gof=15)

cat("INTERPRETATION: AR(1), MA(1), or AR(2) models show similar performance\n")
cat("for inflation expectation changes.\n\n")

## ------------------- DOLLAR INDEX CHANGES ANALYSIS -------------------
cat("--- TRADE-WEIGHTED DOLLAR INDEX CHANGES ANALYSIS ---\n")

cat("armaselect() result:\n")
dollar_armaselect <- armaselect(dollar_changes)
print(head(dollar_armaselect, 5))

cat("\nauto.arima() result:\n")
dollar_auto <- auto.arima(dollar_changes)
print(dollar_auto)

cat("\nar() result:\n") 
dollar_ar <- ar(dollar_changes)
cat("Selected AR order:", dollar_ar$order, "\n")

# Test various candidate models
cat("\nTesting candidate models:\n")
dollar_ma1 <- arima(dollar_changes, order=c(0,0,1), include.mean=FALSE)
dollar_ar1 <- arima(dollar_changes, order=c(1,0,0), include.mean=FALSE)
dollar_arma32 <- arima(dollar_changes, order=c(3,0,2), include.mean=FALSE)

tsdiag(dollar_ma1, gof=15)
tsdiag(dollar_ar1, gof=15)
tsdiag(dollar_arma32, gof=15)

cat("INTERPRETATION: Exchange rate changes may require more complex ARMA\n")
cat("structures to fully capture the autocorrelation patterns.\n\n")


# Step 4: Regression Analysis - Gold Returns on Macro Variables ---------------
#
# Investigate the relationship between gold returns and changes in macroeconomic
# variables. This helps us understand what drives gold price movements.
#
# ECONOMIC HYPOTHESES:
# 1. TIPS yield changes: Higher real yields may reduce gold's appeal (negative coefficient)
# 2. Breakeven inflation: Higher inflation expectations may increase gold demand (positive coefficient)
# 3. Dollar strength: Stronger dollar typically reduces gold prices (negative coefficient)

# Regression Analysis: Gold Returns and Macro Variables

# Basic OLS regression
cat("--- BASIC OLS REGRESSION ---\n")
basic_regression <- lm(gold_returns ~ tips_changes + inflation_changes + dollar_changes)
summary(basic_regression)

# Interpretation of coefficients
cat("\nECONOMIC INTERPRETATION OF COEFFICIENTS:\n")
cat("- TIPS yield coefficient: Effect of real interest rate changes on gold returns\n")
cat("- Breakeven inflation coefficient: Effect of inflation expectations on gold returns\n") 
cat("- Dollar index coefficient: Effect of currency strength on gold returns\n")
cat("- Statistical significance indicates which variables reliably predict gold returns\n\n")

# Step 5: Residual Analysis and Serial Correlation Correction ----------------
#
# Check if regression residuals exhibit serial correlation, which violates
# the assumption of independent errors and can lead to incorrect inference.

# Residual Analysis and Serial Correlation Testing

# Method 1: Use ARIMA framework for easier residual diagnostics
cat("--- METHOD 1: ARIMA FRAMEWORK FOR REGRESSION ---\n")
arima_regression <- arima(gold_returns, order=c(0,0,0), 
                         xreg=cbind(tips_changes, inflation_changes, dollar_changes), 
                         include.mean = TRUE)

cat("Residual diagnostics for basic regression (ARIMA framework):\n")
tsdiag(arima_regression)

cat("Detailed residual analysis:\n")
tsdisplay(residuals(basic_regression), 
          main="Residuals from Basic OLS Regression")

cat("INTERPRETATION: Look for:\n")
cat("- Ljung-Box p-values < 0.05 indicate serial correlation\n") 
cat("- Significant ACF/PACF spikes suggest autocorrelated errors\n\n")

# Method 2: Automatic selection of error structure
cat("--- METHOD 2: AUTOMATIC ERROR STRUCTURE SELECTION ---\n")
cat("Using auto.arima() to find optimal error structure:\n")
auto_regression <- auto.arima(gold_returns, 
                             xreg=cbind(tips_changes, inflation_changes, dollar_changes))
print(auto_regression)

# Method 3: Manual specification based on auto.arima results
cat("\n--- METHOD 3: MANUAL ERROR STRUCTURE SPECIFICATION ---\n")
# Assuming auto.arima suggests MA(2) errors
regression_ma2 <- arima(gold_returns, order=c(0,0,2), 
                       xreg=cbind(tips_changes, inflation_changes, dollar_changes), 
                       include.mean = TRUE)
print(regression_ma2)

cat("\nDiagnostics for MA(2) error model:\n")
tsdiag(regression_ma2, gof=40)
tsdisplay(residuals(regression_ma2),
          main="Residuals from Regression with MA(2) Errors")

cat("INTERPRETATION: MA(2) errors should reduce autocorrelation.\n")
cat("Some remaining correlation may be acceptable if small.\n\n")

# Robust Standard Errors Approach ---------------------------------------------

# Alternative: Robust Standard Errors

cat("Instead of modeling error structure, use HAC (Heteroskedasticity and\n")
cat("Autocorrelation Consistent) standard errors:\n\n")

cat("--- COMPARISON OF STANDARD ERRORS ---\n")
cat("Standard OLS standard errors:\n")
ols_results <- coeftest(basic_regression)
print(ols_results)

cat("\nHAC (Robust) standard errors:\n")
hac_results <- coeftest(basic_regression, vcov=vcovHAC(basic_regression))
print(hac_results)

# Calculate the change in standard errors
se_comparison <- data.frame(
  Variable = rownames(ols_results),
  OLS_SE = ols_results[,2],
  HAC_SE = hac_results[,2],
  SE_Ratio = hac_results[,2] / ols_results[,2]
)

cat("\nStandard Error Comparison:\n")
print(se_comparison)

cat("\nINTERPRETATION:\n")
cat("- SE_Ratio > 1: HAC standard errors are larger (more conservative)\n")
cat("- Large ratios indicate significant autocorrelation in residuals\n")
cat("- Use HAC standard errors if residuals show serial correlation\n\n")

# Model Comparison and Recommendations ----------------------------------------

# Final Model Comparison

# Compare different approaches
models_summary <- data.frame(
  Approach = c("Basic OLS", "ARIMA with MA(2) errors", "OLS with HAC SE"),
  Pros = c("Simple, interpretable", "Accounts for error structure", "Robust to serial correlation"),
  Cons = c("May have biased SE", "More complex", "Doesn't model error structure"),
  When_to_Use = c("No serial correlation", "Clear ARMA error pattern", "Uncertain error structure")
)

print(models_summary)

cat("\nRECOMMENDATION FOR PRACTICE:\n")
cat("1. Start with basic OLS and test for serial correlation\n")
cat("2. If present, try ARIMA error specification or HAC standard errors\n") 
cat("3. Report multiple approaches if results differ substantially\n")
cat("4. Focus on economic significance, not just statistical significance\n")
cat("5. Consider that financial markets may have complex error structures\n\n")

# Analysis Summary


  