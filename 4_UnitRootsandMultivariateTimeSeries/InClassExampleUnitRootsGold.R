# UNIT ROOT TESTING EXAMPLES - COMMODITIES AND MACROECONOMIC DATA ==================

# This program demonstrates unit root testing procedures on:
# 1. Macro and oil price data (lumber PPI, metals PPI, oil, inflation)
# 2. Gold price data with monetary policy variables (gold, TIPS, inflation, USD index)
# 
# Key concepts:
# - Individual unit root tests using cadftest() with covariate inclusion capability
# - Joint hypothesis testing using ur.df() for combined tests
# - Testing with different specifications (Case 2: drift vs Case 4: trend)
# - Interpreting unit root test results for economic time series

# --------------------- SETUP AND PACKAGE LOADING ----------------------
# Load required packages
library(quantmod)    # For financial and economic data retrieval from FRED
library(fBasics)     # Basic financial statistics and tests
library(tseries)     # Time series analysis and additional tests
library(CADFtest)    # Covariate-augmented Dickey-Fuller tests
library(urca)        # Unit root and cointegration tests with joint testing
library(forecast)    # Time series forecasting and model evaluation
library(lubridate)   # Date manipulation and formatting

# ================ PART 1: MACRO AND OIL PRICE DATA ================

# Data Acquisition - Retrieve economic data from FRED
getSymbols("T10YIE", src="FRED")     # 10-year break-even inflation rate (daily)
getSymbols("WPU081", src="FRED")     # Lumber Producer Price Index (monthly)
getSymbols("WPU10", src="FRED")      # Metals Producer Price Index (monthly)
getSymbols("MCOILWTICO", src="FRED") # WTI Oil spot price (monthly)

# Data Preparation
# Convert daily inflation data to monthly averages for compatibility
inflation_monthly <- apply.monthly(na.omit(T10YIE), FUN=mean)
inflation_monthly <- xts(inflation_monthly, order.by = as.yearmon(index(inflation_monthly)))

# Define common time window for analysis
start_date <- "2003-01-01"    
end_date <- "2021-08-01"   

# Extract data for the specified time window and remove missing values
lumber_ppi <- na.omit(window(WPU081, start = start_date, end = end_date))
metals_ppi <- na.omit(window(WPU10, start = start_date, end = end_date))
oil_prices <- na.omit(window(MCOILWTICO, start = start_date, end = end_date))
inflation_data <- na.omit(window(inflation_monthly, start = start_date, end = end_date))

# Combine all series into one dataset with descriptive column names
macro_oil_data <- merge.xts(inflation_data, lumber_ppi, metals_ppi, log(oil_prices), join="inner")
colnames(macro_oil_data) <- c("Inflation", "LumberPPI", "MetalsPPI", "LogOilPrice")

# --------------------- LUMBER PPI UNIT ROOT ANALYSIS ---------------------
# Test Lumber Producer Price Index for unit roots

# Convert to time series object for analysis
lumber_ts <- ts(macro_oil_data$LumberPPI, freq=12, start=2003)
ts.plot(lumber_ts, main="Lumber Producer Price Index", 
        ylab="PPI", xlab="Year")

# Lumber appears to have a trend - use Case 4 (trend specification)
lumber_cadf_trend <- CADFtest(lumber_ts, max.lag.y=48, criterion=c("BIC"), type="trend")
cat("=== Lumber PPI Unit Root Test (Individual) ===\n")
summary(lumber_cadf_trend)
cat("Result: Fail to reject null of unit root with 8 lags\n\n")

# Joint hypothesis testing using ur.df()
lumber_urdf_trend <- ur.df(lumber_ts, lags=8, type="trend")
cat("=== Lumber PPI Unit Root Test (Joint - Case 4) ===\n")
summary(lumber_urdf_trend)
cat("Result: Fail to reject phi2 joint test, but trend is individually significant\n")
cat("Conclusion: Move to Case 2 (drift specification)\n\n")

# Test with Case 2 (drift) specification
lumber_urdf_drift <- ur.df(lumber_ts, lags=8, type="drift")
cat("=== Lumber PPI Unit Root Test (Joint - Case 2) ===\n")
summary(lumber_urdf_drift)
cat("Result: Fail to reject phi1 joint test\n")
cat("Conclusion: Random walk without drift (or with deterministic trend)\n\n")

# Verify by testing first differences for drift
lumber_diff_test <- lm(na.omit(diff(lumber_ts)) ~ 1)
cat("=== Testing First Differences for Drift ===\n")
summary(lumber_diff_test)
cat("Intercept not significant - confirms no drift in random walk\n\n")

# --------------------- METALS PPI UNIT ROOT ANALYSIS ---------------------
# Test Metals Producer Price Index for unit roots

# Convert to time series object for analysis
metals_ts <- ts(macro_oil_data$MetalsPPI, freq=12, start=2003)
ts.plot(metals_ts, main="Metals Producer Price Index", 
        ylab="PPI", xlab="Year")

# Metals PPI appears to have a trend - use Case 4 (trend specification)
metals_cadf_trend <- CADFtest(metals_ts, max.lag.y=48, criterion=c("BIC"), type="trend")
cat("=== Metals PPI Unit Root Test (Individual) ===\n")
summary(metals_cadf_trend)
cat("Result: Fail to reject null of unit root with 1 lag\n\n")

# Joint hypothesis testing using ur.df()
metals_urdf_trend <- ur.df(metals_ts, lags=1, type="trend")
cat("=== Metals PPI Unit Root Test (Joint - Case 4) ===\n")
summary(metals_urdf_trend)
cat("Result: Fail to reject phi2 joint test\n")
cat("Result: Trend and Intercept terms are significant\n")
cat("Conclusion: Try Case 2 (drift specification) with caution\n\n")

# Test with Case 2 (drift) specification
metals_urdf_drift <- ur.df(metals_ts, lags=1, type="drift")
cat("=== Metals PPI Unit Root Test (Joint - Case 2) ===\n")
summary(metals_urdf_drift)
cat("Result: Fail to reject phi1 joint test\n")
cat("Conclusion: Random walk without drift\n\n")

# Confirmed by ARIMA intercept
metal_arima <- Arima(metals_ts, order=c(1,1,0), include.constant=TRUE)
summary(metal_arima)
cat("ARIMA intercept also not significant, supporting no drift\n")

# Test first differences for significant drift
metals_diff_test <- lm(na.omit(diff(metals_ts)) ~ 1)
cat("=== Testing First Differences for Drift ===\n")
summary(metals_diff_test)
cat("Result: Intercept is significant without lags\n")

metals_t_test <- t.test(na.omit(diff(metals_ts)))
cat("=== t-test for Mean of First Differences ===\n")
print(metals_t_test)
cat("Intercept IS significant - suggests drift may be present\n")
cat("Decision point: Evidence conflicts between unit root test and drift test\n\n")

# --------------------- OIL PRICE UNIT ROOT ANALYSIS ---------------------
# Test WTI Oil Prices (log transformed) for unit roots

# Convert to time series object for analysis
oil_ts <- ts(macro_oil_data$LogOilPrice, freq=12, start=2003)
ts.plot(oil_ts, main="Log WTI Oil Prices", 
        ylab="Log Price", xlab="Year")

# WTI appears NOT to have a trend - use Case 2 (drift specification)
oil_cadf_drift <- CADFtest(oil_ts, max.lag.y=48, criterion=c("BIC"), type="drift")
cat("=== Log WTI Oil Price Unit Root Test (Individual) ===\n")
summary(oil_cadf_drift)
cat("Result: REJECT the null of unit root with 1 lag!\n")
cat("Note: Examine the lagged coefficient magnitude\n\n")

# Joint hypothesis testing using ur.df()
oil_urdf_drift <- ur.df(oil_ts, lags=1, type="drift")
cat("=== Log WTI Oil Price Unit Root Test (Joint - Case 2) ===\n")
summary(oil_urdf_drift)
cat("Result: Reject both individual (tau2) and joint (phi1) null hypotheses\n")
cat("Conclusion: Log oil prices are stationary (no unit root)\n\n")

# ================ PART 2: GOLD AND MONETARY POLICY DATA ================

# Data Acquisition - Retrieve gold and monetary policy data
getSymbols("GC=F")                    # Gold futures price (daily)
getSymbols("DFII10", src="FRED")          # 10-year TIPS constant maturity (daily)
getSymbols("DTWEXBGS", src="FRED")        # Trade-weighted US Dollar Index (daily)

# Data Preparation for Gold Analysis
# Combine gold, TIPS, breakeven inflation, and dollar index data
gold_monetary_data <- merge.xts(`GC=F`$`GC=F.Adjusted`, DFII10$DFII10, 
                                T10YIE$T10YIE, DTWEXBGS, join="inner")

colnames(gold_monetary_data) <- c("GoldPrice", "TIPS", "BreakevenInflation", "DollarIndex")

# Define time window for gold analysis
gold_start_date <- "2006-01-03"    
gold_end_date <- "2021-08-30"    

# Extract gold analysis dataset
gold_data <- window(gold_monetary_data, start=gold_start_date, end=gold_end_date)
gold_data <- na.omit(gold_data)


# --------------------- GOLD PRICE UNIT ROOT ANALYSIS ---------------------
# Test Gold prices for unit roots

# Visualize gold price data
ts.plot(gold_data$GoldPrice, main="Gold Prices", 
        ylab="Price ($)", xlab="Year")

# Gold appears to have a trend - use Case 4 (trend specification)
gold_cadf_trend <- CADFtest(gold_data$GoldPrice, criterion=c("AIC"), type="trend")
cat("=== Gold Price Unit Root Test (Individual) ===\n")
summary(gold_cadf_trend)
cat("Result: Fail to reject null of unit root with 1 additional lag\n\n")

# Joint hypothesis testing using ur.df()
gold_urdf_trend <- ur.df(gold_data$GoldPrice, type=c("trend"), lags=1)
cat("=== Gold Price Unit Root Test (Joint - Case 4) ===\n")
summary(gold_urdf_trend)
cat("Result: Fail to reject phi2 joint test\n")
cat("Result: trend term not individually significant\n")
cat("Conclusion: Move to Case 2 (drift specification)\n\n")

# Test with Case 2 (drift) specification
gold_urdf_drift <- ur.df(gold_data$GoldPrice, type=c("drift"), lags=1)
cat("=== Gold Price Unit Root Test (Joint - Case 2) ===\n")
summary(gold_urdf_drift)
cat("Result: Fail to reject phi1 joint test\n")
cat("Conclusion: Random walk without drift\n")
cat("Note: Somewhat surprising based on visual trend in data\n\n")

# Test gold returns for drift
gold_returns_test <- lm(na.omit(diff(log(gold_data$GoldPrice))) ~ 1)
cat("=== Testing Gold Log Returns for Drift ===\n")
summary(gold_returns_test)
cat("Intercept not significant - confirms no drift in gold price random walk\n\n")

# Confirmed by ARIMA intercept
gold_arima <- Arima(gold_data$GoldPrice, order=c(1,1,0), include.constant=TRUE)
summary(gold_arima)
cat("ARIMA intercept also not significant, supporting no drift\n")

# --------------------- TIPS UNIT ROOT ANALYSIS ---------------------
# Test 10-year TIPS yields for unit roots

# Visualize TIPS yield data
ts.plot(gold_data$TIPS, main="10-Year TIPS Yields", 
        ylab="Yield (%)", xlab="Year")

# TIPS appears to have a trend - use Case 4 (trend specification)
tips_cadf_trend <- CADFtest(gold_data$TIPS, criterion=c("AIC"), type="trend")
cat("=== TIPS Yield Unit Root Test (Individual) ===\n")
summary(tips_cadf_trend)
cat("Result: Fail to reject null of unit root with 1 additional lag\n\n")

# Joint hypothesis testing using ur.df()
tips_urdf_trend <- ur.df(gold_data$TIPS, type=c("trend"), lags=1)
cat("=== TIPS Yield Unit Root Test (Joint - Case 4) ===\n")
summary(tips_urdf_trend)
cat("Result: Fail to reject phi2 joint test\n")
cat("Conclusion: Move to Case 2 (drift specification)\n\n")

# Test with Case 2 (drift) specification
tips_urdf_drift <- ur.df(gold_data$TIPS, type=c("drift"), lags=1)
cat("=== TIPS Yield Unit Root Test (Joint - Case 2) ===\n")
summary(tips_urdf_drift)
cat("Result: Fail to reject phi1 joint test\n")
cat("Conclusion: Random walk without drift\n")
cat("Note: Somewhat surprising based on visual trend in data\n\n")

# Test TIPS yield changes for drift
tips_changes_test <- lm(na.omit(diff(gold_data$TIPS)) ~ 1)
cat("=== Testing TIPS Yield Changes for Drift ===\n")
summary(tips_changes_test)
cat("Intercept not significant - confirms no drift in TIPS yield random walk\n\n")

# Confirmed by ARIMA intercept
tips_arima <- Arima(gold_data$TIPS, order=c(1,1,0), include.constant=TRUE)
summary(tips_arima)
cat("ARIMA intercept also not significant, supporting no drift\n")

# --------------------- DOLLAR INDEX UNIT ROOT ANALYSIS ---------------------
# Test Trade-weighted US Dollar Index for unit roots

# Visualize Dollar Index data
ts.plot(gold_data$DollarIndex, main="Trade-weighted US Dollar Index", 
        ylab="Index", xlab="Year")

# Dollar Index appears to have a trend - use Case 4 (trend specification)
dollar_cadf_trend <- CADFtest(gold_data$DollarIndex, criterion=c("AIC"), type="trend")
cat("=== Dollar Index Unit Root Test (Individual) ===\n")
summary(dollar_cadf_trend)
cat("Result: Fail to reject null of unit root with 1 additional lag\n\n")

# Joint hypothesis testing using ur.df()
dollar_urdf_trend <- ur.df(gold_data$DollarIndex, type=c("trend"), lags=1)
cat("=== Dollar Index Unit Root Test (Joint - Case 4) ===\n")
summary(dollar_urdf_trend)
cat("Result: Fail to reject phi2 joint test\n")
cat("Result: But trend is individually significant\n")
cat("Conclusion: Explore Case 2 (drift specification) with caution\n\n")

# Test with Case 2 (drift) specification
dollar_urdf_drift <- ur.df(gold_data$DollarIndex, type=c("drift"), lags=1)
cat("=== Dollar Index Unit Root Test (Joint - Case 2) ===\n")
summary(dollar_urdf_drift)
cat("Result: Fail to reject phi1 joint test\n")
cat("Conclusion: Random walk without drift\n\n")

# Test Dollar Index changes for drift
dollar_changes_test <- lm(na.omit(diff(gold_data$DollarIndex)) ~ 1)
cat("=== Testing Dollar Index Changes for Drift ===\n")
summary(dollar_changes_test)
cat("Intercept not significant - confirms no drift in Dollar Index random walk\n\n")

# Confirmed by ARIMA intercept
doll_arima <- Arima(gold_data$DollarIndex, order=c(1,1,0), include.constant=TRUE)
summary(doll_arima)
cat("ARIMA intercept also not significant, supporting no drift\n")

# ---------------------- SUMMARY OF FINDINGS -------------------------
cat("\n===========================================================\n")
cat("SUMMARY OF UNIT ROOT TEST FINDINGS\n")
cat("===========================================================\n")
cat("MACRO AND OIL PRICE DATA:\n")
cat("1. Lumber PPI: Unit root (I(1)), no drift\n")
cat("2. Metals PPI: Unit root (I(1)), conflicting evidence on drift\n")
cat("3. Log Oil Prices: Stationary (I(0)), no unit root\n\n")
cat("GOLD AND MONETARY POLICY DATA:\n")
cat("1. Gold Prices: Unit root (I(1)), no drift\n")
cat("2. TIPS Yields: Unit root (I(1)), no drift\n")
cat("3. Dollar Index: Unit root (I(1)), no drift\n\n")
cat("METHODOLOGICAL INSIGHTS:\n")
cat("- Joint tests (ur.df) provide more information than individual tests\n")
cat("- Testing with both Case 2 and Case 4 helps identify correct specification\n")
cat("- Visual inspection can be misleading for unit root determination\n")
cat("- First difference tests help confirm drift conclusions\n")
cat("-----------------------------------------------------------\n") 
