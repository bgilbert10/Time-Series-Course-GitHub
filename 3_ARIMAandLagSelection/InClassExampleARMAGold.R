# In-Class Activity: Gold Prices and Macroeconomic Variables -----------------
# Date: September 25 Class Session
#
# ACTIVITY OBJECTIVES:
# 1. Practice downloading and merging multiple financial time series
# 2. Conduct preliminary stationarity analysis using visual inspection
# 3. Apply ARIMA modeling to individual economic time series
# 4. Estimate regression models with time series data
# 5. Diagnose and correct serial correlation in regression residuals
#
# STUDENT LEARNING OUTCOMES:
# By completing this activity, students will be able to:
# - Retrieve data from multiple sources (Yahoo Finance, FRED)
# - Merge time series with different frequencies and missing observations
# - Distinguish between stationary and non-stationary series visually
# - Select appropriate transformations for economic variables
# - Interpret ARIMA model selection criteria
# - Test for and correct serial correlation in regression models
#
# ECONOMIC CONTEXT:
# Gold is traditionally viewed as a hedge against inflation and currency debasement.
# This activity explores the empirical relationships between gold returns and:
# - Real interest rates (TIPS yields)
# - Inflation expectations (breakeven inflation)
# - Currency strength (trade-weighted dollar index)
#
# TIME PERIOD UPDATED: January 3, 2008 to June 30, 2020
# (Note: Original activity used 2006-2020, updated for data availability)

# Load required packages with descriptions
library(forecast)    # ARIMA modeling and forecasting functions
library(quantmod)    # Quantitative financial modeling and data retrieval
library(caschrono)   # Additional chronological and time series tools
library(texreg)      # LaTeX and HTML regression table output
library(sandwich)    # Robust covariance matrix estimators (HAC)
library(lmtest)      # Linear model specification and diagnostic tests


# Step 1: Data Collection and Preparation -------------------------------------
#
# INSTRUCTION FOR STUDENTS:
# Follow along as we download data from different sources. Note the different
# data providers and how we handle missing values and date alignment.

# Download time series data from different sources
cat("Downloading data from Yahoo Finance and FRED...\n")

# Yahoo Finance: Gold futures contract
getSymbols("GC=F")                          # Gold futures continuous contract

# FRED (Federal Reserve Economic Data): Macroeconomic variables  
getSymbols("DFII10", src="FRED")             # 10-year TIPS constant maturity yield
getSymbols("T10YIE", src="FRED")             # 10-year breakeven inflation rate
getSymbols("DTWEXBGS", src="FRED")           # Trade-weighted US dollar index

cat("Data successfully downloaded!\n\n")

# STUDENT EXERCISE: Examine each series individually before merging
# Uncomment the following lines to explore:
# head(`GC=F`)
# head(DFII10) 
# head(T10YIE)
# head(DTWEXBGS)

cat("Merging datasets with different frequencies and observation dates...\n")

# Merge all time series into a single xts object
# Note: Different series may have different observation frequencies
economics_data <- merge.xts(`GC=F`$`GC=F.Adjusted`, DFII10$DFII10, 
                           T10YIE$T10YIE, DTWEXBGS,
                           all=TRUE, fill=NA, retclass="xts")

# Assign descriptive variable names
colnames(economics_data) <- c("gold_price", "tips_yield", "breakeven_inflation", "trade_weighted_dollar")

cat("Variable definitions:\n")
cat("- gold_price: Gold futures adjusted closing price ($/oz)\n")
cat("- tips_yield: 10-year Treasury Inflation-Protected Securities yield (%)\n") 
cat("- breakeven_inflation: 10-year breakeven inflation rate (%)\n")
cat("- trade_weighted_dollar: Trade-weighted US dollar index\n\n")

# Define study period
cat("Setting analysis period...\n")
study_start_date <- "2008-01-03"    
study_end_date   <- "2020-06-30"    

cat("Analysis period:", study_start_date, "to", study_end_date, "\n")
cat("This period includes the 2008 Financial Crisis and subsequent recovery.\n\n")

# Extract data for specified time window and remove missing observations
economics_data <- window(economics_data, start = study_start_date, end = study_end_date)
economics_data <- na.omit(economics_data)

cat("Final dataset summary:\n")
cat("Observations:", nrow(economics_data), "\n")
cat("Variables:", ncol(economics_data), "\n")
cat("Date range:", index(economics_data)[1], "to", index(economics_data)[nrow(economics_data)], "\n\n")


# ===============================================================================
# STEP 2: STATIONARITY INVESTIGATION - VISUAL ANALYSIS
# ===============================================================================
#
# INSTRUCTION FOR STUDENTS:
# Before formal statistical tests, we examine each series visually.
# Look for trends, structural breaks, and persistent patterns that indicate
# non-stationarity.

# Step 2: Visual Stationarity Analysis

# QUESTION FOR STUDENTS: What visual patterns suggest non-stationarity?
# ANSWER: Look for trends, permanent shifts, slow-decaying ACF

# ------------------- GOLD PRICES ANALYSIS -------------------
cat("Analyzing Gold Prices (Levels)...\n")
cat("STUDENT TASK: Describe the trend and volatility patterns you observe\n")

chartSeries(economics_data$gold_price,
           main="Gold Futures Prices: Level Series",
           theme="white")
           
acf(economics_data$gold_price, 
    main="ACF: Gold Prices (Levels)", 
    lag.max=50)
    
cat("INSTRUCTOR NOTE: Slowly decaying ACF suggests unit root/non-stationarity\n\n")

# ------------------- TIPS YIELDS ANALYSIS -------------------
cat("Analyzing TIPS Yields (Levels)...\n") 
cat("STUDENT QUESTION: Do interest rates appear mean-reverting or trending?\n")

chartSeries(economics_data$tips_yield,
           main="10-Year TIPS Yields: Level Series",
           theme="white")
           
acf(economics_data$tips_yield,
    main="ACF: TIPS Yields (Levels)",
    lag.max=50)
    
cat("DISCUSSION POINT: Interest rates show some mean reversion but can be persistent\n\n")

# ------------------- BREAKEVEN INFLATION ANALYSIS -------------------
cat("Analyzing Breakeven Inflation (Levels)...\n")
cat("STUDENT OBSERVATION: How do inflation expectations behave over time?\n")

chartSeries(economics_data$breakeven_inflation,
           main="10-Year Breakeven Inflation: Level Series", 
           theme="white")
           
acf(economics_data$breakeven_inflation,
    main="ACF: Breakeven Inflation (Levels)",
    lag.max=50)
    
cat("TEACHING POINT: Inflation expectations can have unit roots or be near-unit root\n\n")

# ------------------- TRADE-WEIGHTED DOLLAR ANALYSIS -------------------
cat("Analyzing Trade-Weighted Dollar Index (Levels)...\n")
cat("STUDENT EXERCISE: Compare this to other exchange rate series you've seen\n")

chartSeries(economics_data$trade_weighted_dollar,
           main="Trade-Weighted Dollar Index: Level Series",
           theme="white")
           
acf(economics_data$trade_weighted_dollar,
    main="ACF: Trade-Weighted Dollar (Levels)", 
    lag.max=50)

cat("CLASS DISCUSSION: Exchange rates often exhibit random walk behavior\n\n")

cat("PRELIMINARY CONCLUSION FROM VISUAL ANALYSIS:\n")
cat("- All series appear non-stationary in levels\n") 
cat("- Slowly decaying ACF patterns support this conclusion\n")
cat("- First differencing will likely be needed for modeling\n")
cat("- Next step: Transform to stationary series\n\n")


# ===============================================================================
# STEP 3: CREATING STATIONARY TRANSFORMATIONS
# ===============================================================================
#
# INSTRUCTION FOR STUDENTS:
# Now we create stationary versions of each series using appropriate transformations.
# The choice of transformation depends on the economic interpretation we want.

# Step 3: Creating Stationary Transformations ---------------------------------

# TRANSFORMATION STRATEGY:
# - Gold prices → Log differences (continuously compounded returns)
# - Interest rates → Simple differences (changes in percentage points)
# - Exchange rates → Simple differences (changes in index points)

# STUDENT QUESTION: Why log differences for gold but not for interest rates?
# ANSWER: Gold prices are in levels (multiplicative), rates are already in percentages

# Transform each series appropriately
gold_returns <- ts(na.omit(diff(log(economics_data$gold_price))))
tips_changes <- ts(na.omit(diff(economics_data$tips_yield)))
inflation_changes <- ts(na.omit(diff(economics_data$breakeven_inflation)))
dollar_changes <- ts(na.omit(diff(economics_data$trade_weighted_dollar)))

cat("Transformed variables created:\n")
cat("- gold_returns: Continuously compounded gold returns\n")
cat("- tips_changes: Changes in TIPS yields (percentage points)\n") 
cat("- inflation_changes: Changes in breakeven inflation (percentage points)\n")
cat("- dollar_changes: Changes in trade-weighted dollar index\n\n")

# STUDENT EXERCISE: Examine the transformed series
cat("STUDENT EXERCISE: Plot the transformed series and comment on stationarity\n")
par(mfrow=c(2,2))
plot(gold_returns, main="Gold Returns", ylab="Log Difference")
plot(tips_changes, main="TIPS Yield Changes", ylab="Percentage Points") 
plot(inflation_changes, main="Inflation Expectation Changes", ylab="Percentage Points")
plot(dollar_changes, main="Dollar Index Changes", ylab="Index Points")
par(mfrow=c(1,1))

# all approaches agree on white noise for gold returns
armaselect(gold_returns)
auto.arima(gold_returns)
ar(gold_returns)
tsdisplay(gold_returns)

# armaselect says ARMA(1,2), but uses BIC which penalizes parameters more heavily. 
armaselect(tips_changes)
# auto.arima says ARMA(1,2)
auto.arima(tips_changes)
# ar() says AR(8)
ar(tips_changes)
# Only the AR(8) cleans up the autocorrelation
tsdiag(arima(tips_changes,order=c(1,0,2),include.mean=F),gof=15)
tsdiag(arima(tips_changes,order=c(8,0,0),include.mean=F),gof=15)

# armaselect says ARMA(1,0)
armaselect(inflation_changes)
# auto.arima says ARMA(0,1)
auto.arima(inflation_changes)
# ar() says AR(4)
ar(inflation_changes)
# ARMA(0,1) and AR(2) both do pretty well
tsdiag(arima(inflation_changes,order=c(1,0,0),include.mean=F),gof=15)
tsdiag(arima(inflation_changes,order=c(0,0,1),include.mean=F),gof=15)
tsdiag(arima(inflation_changes,order=c(2,0,0),include.mean=F),gof=15)

# armaselect says ARMA(0,0)
armaselect(dollar_changes)
# auto.arima says ARMA(0,1)
auto.arima(dollar_changes)
# ar() says AR(1)
ar(dollar_changes)
# neither of these looks okay
tsdiag(arima(dollar_changes,order=c(0,0,1),include.mean=F),gof=15)
tsdiag(arima(dollar_changes,order=c(1,0,0),include.mean=F),gof=15)
# after playing around a bit:
tsdiag(arima(dollar_changes,order=c(3,0,2),include.mean=F),gof=15)


# 4) currency-related macro variables are significant explanatory factors in
# gold returns. 
mod <- lm(gold_returns ~ tips_changes + inflation_changes + dollar_changes)
summary(mod)


# 5)
# same results, but easier to evaluate residuals:
mod1 <- arima(gold_returns,order=c(0,0,0),xreg=cbind(tips_changes,inflation_changes,dollar_changes),include.mean = T)
# clear autocorrelation in residuals from Ljung-Box tests: 
tsdiag(mod1)
tsdisplay(residuals(mod))

# auto.arima says MA(2) errors are optimal
auto.arima(gold_returns,xreg=cbind(tips_changes,inflation_changes,dollar_changes))

mod2 <- arima(gold_returns,order=c(0,0,2),xreg=cbind(tips_changes,inflation_changes,dollar_changes),include.mean = T)
# standard errors are very, very similar
mod2

# Ljung-Box tests look better, but still some significant individual ACFs.
# is it enough to worry about? Maybe not. 
tsdiag(mod2, gof=40)
tsdisplay(residuals(mod2))

# I did some trial and error. It takes a large model to get white noise errors. 

# alternatively, compare results from OLS standard errors to HAC standard errors
# HAC standard errors increase a bit. 
coeftest(mod)
coeftest(mod,vcov=vcovHAC(mod))

# which one to report? MA(1) errors, large ARMA errors, or basic model with HAC? 
# my view: just report them all. 


  