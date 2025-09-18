# ======================================================================
# RETURNS DISTRIBUTIONS ANALYSIS
# ======================================================================

# --------------------- SETUP AND PACKAGE LOADING ----------------------
# Load required packages for statistical analysis and data visualization
# install.packages("fBasics") # Uncomment to install if needed
library(fBasics)              # For basic statistical analysis
# install.packages("quantmod") # Uncomment to install if needed
library(quantmod)             # For financial data and returns calculation

# Alternative package loading approach using pacman
# if(!require(pacman))install.packages("pacman")
# pacman::p_load(fBasics,quantmod)

# Set working directory
setwd("C:/Users/gilbe/Dropbox/Econometrics/TimeSeriesCourse") 

# -------------------- IBM RETURNS ANALYSIS (SIMPLE) ------------------
# Load IBM returns data
data_ibm = read.table('Fall2017/TsayCh1/Lecture/d-ibm3dx7008.txt', header=TRUE) 

# Examine data structure
head(data_ibm)
tail(data_ibm)
dim(data_ibm)

# Extract IBM returns
ibm_returns = data_ibm[,2]

# Time series visualization
ts.plot(ibm_returns, main="Daily simple returns of IBM",
        xlab="Time", ylab="Returns") 

# Calculate key statistics for simple returns
simple_ibm_returns = ibm_returns
basicStats(simple_ibm_returns)
mean_returns = mean(simple_ibm_returns)
variance_returns = var(simple_ibm_returns)
sd_returns = sqrt(variance_returns)
skew_returns = skewness(simple_ibm_returns)
kurt_returns = kurtosis(simple_ibm_returns)

# Test mean returns (null hypothesis: mean=0)
t.test(simple_ibm_returns) 
t.test(simple_ibm_returns, alternative=c("greater")) # one-sided test

# Visualize return distribution vs normal
calculate_and_plot_density = function(returns, title="Returns Distribution") {
  # Create histogram
  hist(returns, nclass=40, main=title, xlab="Returns", ylab="Frequency")
  
  # Calculate and plot density
  density_estimate = density(returns)
  plot(density_estimate$x, density_estimate$y, type="l", 
       main=paste(title, "Density"), xlab="Returns", ylab="Density")
  
  # Add normal distribution overlay
  mean_val = mean(returns)
  sd_val = sd(returns)
  x_vals = seq(min(density_estimate$x), max(density_estimate$x), length.out=100)
  y_vals = dnorm(x_vals, mean=mean_val, sd=sd_val)
  lines(x_vals, y_vals, lty=2, col="red") 
  legend("topright", c("Empirical", "Normal"), lty=c(1,2), col=c("black", "red"))
}

calculate_and_plot_density(simple_ibm_returns, "IBM Simple Returns")

# Test normality
test_normality = function(returns, name="Returns") {
  cat("Normality Tests for", name, "\n")
  cat("------------------------------------------------\n")
  
  # Jarque-Bera test
  jb_test = normalTest(returns, method='jb')
  print(jb_test)
  
  # Test skewness
  s1 = skewness(returns)
  sample_size = length(returns)
  skew_test = s1/sqrt(6/sample_size)
  p_value_skew = 2*(pnorm(abs(skew_test), lower.tail=FALSE))
  cat("Skewness:", s1, "| Test statistic:", skew_test, "| p-value:", p_value_skew, "\n")
  
  # Test kurtosis
  k4 = kurtosis(returns)
  kurt_test = k4/sqrt(24/sample_size)
  p_value_kurt = 2*(pnorm(abs(kurt_test), lower.tail=FALSE))
  cat("Kurtosis:", k4, "| Test statistic:", kurt_test, "| p-value:", p_value_kurt, "\n")
  cat("------------------------------------------------\n\n")
}

test_normality(simple_ibm_returns, "IBM Simple Returns")

# -------------------- IBM RETURNS ANALYSIS (LOG) ---------------------
# Log returns calculation and analysis
log_ibm_returns = 100*log(ibm_returns+1) # Computes log return in percentage

# Calculate statistics
basicStats(log_ibm_returns)

# T-tests
t.test(log_ibm_returns) 
t.test(log_ibm_returns, alternative=c("greater"))

# Distribution comparison
calculate_and_plot_density(log_ibm_returns, "IBM Log Returns")

# Normality testing
test_normality(log_ibm_returns, "IBM Log Returns")

# -------------------- ENERGY SECTOR ANALYSIS ------------------------
# Fetch market index data
getSymbols("XLE", from="2000-01-03")  # SPDR Energy ETF
getSymbols("SPY", from="2000-01-03")  # S&P 500 Index

# Calculate returns for XLE
calculate_returns = function(symbol_data) {
  returns = list(
    daily_simple = 100*dailyReturn(symbol_data, leading=FALSE, type='arithmetic'),
    daily_log = 100*dailyReturn(symbol_data, leading=FALSE, type='log'),
    monthly_simple = 100*monthlyReturn(symbol_data, leading=FALSE, type='arithmetic'),
    monthly_log = 100*monthlyReturn(symbol_data, leading=FALSE, type='log')
  )
  return(returns)
}

# Calculate returns for all securities
xle_returns = calculate_returns(XLE)
spy_returns = calculate_returns(SPY)

# Visualize returns
chartSeries(XLE, type=c("line"), theme="white", TA=NULL, 
           main="SPDR Energy ETF Price")
chartSeries(xle_returns$daily_log, type=c("line"), theme="white", TA=NULL,
           main="XLE Daily Log Returns")
chartSeries(xle_returns$monthly_log, type=c("line"), theme="white", TA=NULL,
           main="XLE Monthly Log Returns")

# -------------------- INDIVIDUAL ENERGY STOCKS -----------------------
# Fetch and process individual stock data
getSymbols("SRE", from="2000-01-03")  # Sempra Energy
getSymbols("BRK-B", from="2000-01-03")  # Berkshire Hathaway
getSymbols("DUK", from="2000-01-03")  # Duke Energy

# Calculate returns
sre_returns = calculate_returns(SRE)
brk_returns = calculate_returns(`BRK-B`)
duk_returns = calculate_returns(DUK)

# Visualize
chartSeries(SRE, type=c("line"), theme="white", TA=NULL,
           main="Sempra Energy Price")
chartSeries(sre_returns$daily_log, type=c("line"), theme="white", TA=NULL,
           main="SRE Daily Log Returns")

# ------------------- COMBINED RETURNS ANALYSIS -----------------------
# Create data frames for each return type
create_combined_returns = function() {
  # Daily simple returns
  daily_simple_returns = cbind(
    spy_returns$daily_simple,
    xle_returns$daily_simple,
    sre_returns$daily_simple,
    brk_returns$daily_simple,
    duk_returns$daily_simple
  )
  colnames(daily_simple_returns) = c("SPY", "XLE", "SRE", "BRK", "DUK")
  
  # Daily log returns
  daily_log_returns = cbind(
    spy_returns$daily_log,
    xle_returns$daily_log,
    sre_returns$daily_log,
    brk_returns$daily_log,
    duk_returns$daily_log
  )
  colnames(daily_log_returns) = c("SPY", "XLE", "SRE", "BRK", "DUK")
  
  # Monthly simple returns
  monthly_simple_returns = cbind(
    spy_returns$monthly_simple,
    xle_returns$monthly_simple,
    sre_returns$monthly_simple,
    brk_returns$monthly_simple,
    duk_returns$monthly_simple
  )
  colnames(monthly_simple_returns) = c("SPY", "XLE", "SRE", "BRK", "DUK")
  
  # Monthly log returns
  monthly_log_returns = cbind(
    spy_returns$monthly_log,
    xle_returns$monthly_log,
    sre_returns$monthly_log,
    brk_returns$monthly_log,
    duk_returns$monthly_log
  )
  colnames(monthly_log_returns) = c("SPY", "XLE", "SRE", "BRK", "DUK")
  
  return(list(
    daily_simple = daily_simple_returns,
    daily_log = daily_log_returns,
    monthly_simple = monthly_simple_returns,
    monthly_log = monthly_log_returns
  ))
}

combined_returns = create_combined_returns()

# Calculate summary statistics
basicStats(combined_returns$daily_simple)
basicStats(combined_returns$daily_log)
basicStats(combined_returns$monthly_simple)
basicStats(combined_returns$monthly_log)

# ------------------ BERKSHIRE HATHAWAY DETAILED ANALYSIS -------------
# Analysis of Berkshire Hathaway daily log returns
brk_daily_log = na.omit(as.data.frame(brk_returns$daily_log))

# Summary statistics
basicStats(brk_daily_log)

# Test mean returns
t.test(brk_daily_log[,1])
t.test(brk_daily_log[,1], alternative=c("greater"))

# Distribution visualization
calculate_and_plot_density(brk_daily_log[,1], "Berkshire Hathaway Daily Log Returns")

# Normality testing
test_normality(brk_daily_log[,1], "Berkshire Hathaway Daily Log Returns")

# ---------------------- ADDITIONAL EXAMPLES --------------------------
# Below are examples from Tsay's book for reference and practice

# Example: Apple stock returns
# x = read.table("Tsay2017/Lecture1/d-aapl0413.txt", header=TRUE)
# y = ts(x[,3], frequency=252, start=c(2004,1))
# plot(y, type='l', xlab='year', ylab='rtn',
#      main='Daily returns of Apple stock: 2004 to 2013')

# Example: T-bill spreads
# x = read.table("Tsay2017/Lecture1/m-tb3ms.txt", header=TRUE)
# y = read.table("Tsay2017/Lecture1/m-tb6ms.txt", header=TRUE)
# int = cbind(x[300:914,4], y[,4])
# tdx = (c(1:615)+11)/12+1959
# plot(tdx, int[,1], xlab='year', ylab='rate', type='l', ylim=c(0,16.5),
#      main="3-Month and 6-Month Treasury Bill Rates")
# lines(tdx, int[,2], lty=2, col="blue")
# legend("topright", c("3-Month", "6-Month"), lty=c(1,2), col=c("black", "blue"))