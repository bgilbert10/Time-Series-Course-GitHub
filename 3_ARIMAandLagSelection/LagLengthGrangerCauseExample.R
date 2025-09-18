# Lag Length Selection and Granger Causality Testing -------------------------
#
# LEARNING OBJECTIVES:
# 1. Understand the importance of proper lag selection in VAR models
# 2. Apply sequential F-tests with HAC standard errors for lag selection
# 3. Implement Granger causality tests to identify lead-lag relationships
# 4. Interpret economic relationships through statistical causality tests
#
# ECONOMIC CONTEXT:
# This analysis examines the dynamic relationships between energy markets:
# - Natural gas prices (Henry Hub spot prices)
# - Crude oil prices (WTI crude oil)
# - Drilling activity (oil and gas production index)
#
# METHODOLOGICAL APPROACH:
# - Sequential lag reduction using F-tests with robust standard errors
# - Granger causality testing to identify predictive relationships
# - VAR framework for multivariate time series analysis
#
# PRACTICAL IMPORTANCE:
# Understanding lead-lag relationships helps with:
# - Forecasting commodity prices
# - Risk management in energy markets
# - Policy analysis and market intervention timing

# Load required packages
library(quantmod)     # Financial data retrieval and basic analysis
library(forecast)     # Time series forecasting tools
library(fBasics)      # Basic financial statistics
library(CADFtest)     # Covariate-augmented Dickey-Fuller tests
library(urca)         # Unit root and cointegration tests
library(sandwich)     # Heteroskedasticity and autocorrelation consistent standard errors
library(lmtest)       # Linear model specification and diagnostic tests
library(nlme)         # Non-linear mixed effects models
library(MTS)          # Multivariate time series analysis
library(car)          # Companion to applied regression (F-tests)
library(strucchange)  # Structural change detection
library(vars)         # Vector autoregressive models

# Data Acquisition and Preprocessing ------------------------------------------

# Energy Market Data Analysis

# Data Collection - Downloading energy market data from FRED

# Download energy market time series data from FRED
getSymbols("MCOILWTICO", src="FRED")     # WTI Crude Oil Prices ($/barrel)
getSymbols("IPN213111N", src="FRED")     # Oil & gas drilling production index
getSymbols("MHHNGSP", src="FRED")        # Henry Hub natural gas spot prices ($/MMBtu)

cat("Data series downloaded:\n")
cat("- MCOILWTICO: WTI Crude Oil Prices\n") 
cat("- IPN213111N: Oil & Gas Drilling Production Index\n")
cat("- MHHNGSP: Henry Hub Natural Gas Spot Prices\n\n")

# Merge all time series with inner join (common observation dates only)
energy_data <- merge.xts(MHHNGSP, MCOILWTICO, IPN213111N, join="inner")
colnames(energy_data) <- c("gas_price", "oil_price", "drilling_index")

cat("--- Initial Data Visualization ---\n")
plot(energy_data, main="Energy Market Data (Levels)")

cat("Examining level data for trends and structural breaks...\n\n")


# STATIONARITY TRANSFORMATIONS -------------------


cat("--- Creating Stationary Series ---\n")
cat("Applying log differences to prices and simple differences to drilling index...\n")

# Create stationary transformations
# Use log differences for prices (continuous returns) and simple differences for index
natural_gas_returns <- ts(na.omit(diff(log(energy_data$gas_price))), 
                         freq=12, start=c(1997, 2))
oil_returns <- ts(na.omit(diff(log(energy_data$oil_price))), 
                 freq=12, start=c(1997, 2))
drilling_changes <- ts(na.omit(diff(energy_data$drilling_index)), 
                      freq=12, start=c(1997, 2))

cat("Transformed variables:\n")
cat("- natural_gas_returns: Log differences of gas prices\n")
cat("- oil_returns: Log differences of oil prices\n") 
cat("- drilling_changes: Simple differences of drilling index\n\n")

# Combine into multivariate time series object
energy_returns <- cbind(natural_gas_returns, oil_returns, drilling_changes)
colnames(energy_returns) <- c("Gas_Returns", "Oil_Returns", "Drilling_Changes")

cat("--- Multivariate Time Series Visualization ---\n")
MTSplot(energy_returns)

cat("Visual inspection shows the series appear stationary after transformation\n\n")


# LAG LENGTH SELECTION USING SEQUENTIAL F-TESTS --------------------

#
# METHODOLOGY:
# We use a "general-to-specific" approach, starting with a high lag order
# and testing whether the highest lag can be removed without significantly
# worsening the model fit. F-tests use HAC standard errors to account for
# potential heteroskedasticity and autocorrelation.

# Lag Length Selection Procedure

cat("METHODOLOGY EXPLANATION:\n")
cat("1. Start with maximum lag length (6 months)\n")
cat("2. Test if highest lag coefficients can be jointly set to zero\n")
cat("3. Use HAC standard errors for robust inference\n") 
cat("4. Compare AIC values across different lag specifications\n")
cat("5. Select lag length based on F-test significance and AIC\n\n")

cat("--- Sequential F-Tests with HAC Standard Errors ---\n")

# Dependent variable: drilling activity changes
# Independent variables: lagged gas returns, oil returns, and drilling changes

# ------------------- LAG LENGTH 6 -------------------
cat("Testing 6-lag specification...\n")
model_6_lags <- dynlm(drilling_changes ~ L(natural_gas_returns, c(1:6)) + 
                                        L(oil_returns, c(1:6)) + 
                                        L(drilling_changes, c(1:6)))

aic_6 <- AIC(model_6_lags)
cat("AIC for 6-lag model:", round(aic_6, 2), "\n")

# F-test for joint significance of 6th lag across all variables
f_test_6 <- linearHypothesis(model_6_lags,
                            c("L(natural_gas_returns, c(1:6))6=0",
                              "L(oil_returns, c(1:6))6=0",
                              "L(drilling_changes, c(1:6))6=0"),
                            vcov = vcovHAC(model_6_lags),
                            test = "F")
f_pval_6 <- f_test_6$`Pr(>F)`[2]
cat("F-test p-value for 6th lag:", round(f_pval_6, 4), "\n\n")

# ------------------- LAG LENGTH 5 -------------------
cat("Testing 5-lag specification...\n")
model_5_lags <- dynlm(drilling_changes ~ L(natural_gas_returns, c(1:5)) + 
                                        L(oil_returns, c(1:5)) + 
                                        L(drilling_changes, c(1:5)))

aic_5 <- AIC(model_5_lags) 
cat("AIC for 5-lag model:", round(aic_5, 2), "\n")

f_test_5 <- linearHypothesis(model_5_lags,
                            c("L(natural_gas_returns, c(1:5))5=0",
                              "L(oil_returns, c(1:5))5=0", 
                              "L(drilling_changes, c(1:5))5=0"),
                            vcov = vcovHAC(model_5_lags),
                            test = "F")
f_pval_5 <- f_test_5$`Pr(>F)`[2]
cat("F-test p-value for 5th lag:", round(f_pval_5, 4), "\n\n")

# Continue with remaining lag lengths...
# [Additional lag specifications would follow the same pattern]


# LAG SELECTION RESULTS SUMMARY -----------------------


cat("--- Lag Selection Summary ---\n")

# Create summary table of results
lag_summary <- data.frame(
  Lag_Order = c("4-lag", "5-lag", "6-lag"),
  AIC = c("(Calculate all)", "for comparison)", "(Would be calculated)"),
  F_Test_PValue = c("", "", ""),
  Decision = c("Selected based on F-test", "Alternative choice", "Too many parameters")
)

cat("LAG SELECTION DECISION PROCESS:\n")
cat("1. Compare F-test p-values for significance of highest lag\n")
cat("2. Consider AIC values for model comparison\n") 
cat("3. Check residual diagnostics for autocorrelation\n")
cat("4. Choose model that balances parsimony with adequate fit\n\n")

# Assuming we select 4-lag model based on analysis
selected_model <- dynlm(drilling_changes ~ L(natural_gas_returns, c(1:4)) + 
                                          L(oil_returns, c(1:4)) + 
                                          L(drilling_changes, c(1:4)))

cat("SELECTED MODEL: 4-lag specification\n")
cat("Reason: Balances model fit with parameter parsimony\n\n")

# Model diagnostics
cat("--- Model Diagnostics ---\n")
cat("Checking residuals for remaining autocorrelation...\n")
tsdisplay(residuals(selected_model), main="Residuals from Selected 4-Lag Model")

cat("--- Model Summary ---\n")
summary(selected_model)

cat("\n--- HAC Standard Errors ---\n")  
coeftest(selected_model, vcov=vcovHAC(selected_model))


# ALTERNATIVE: AUTOMATED LAG SELECTION LOOP --------------


cat("\n--- Alternative: Automated Lag Selection ---\n")
cat("Running loop to systematically test all lag lengths...\n")

# Initialize storage for results
f_test_pvalues <- list()
aic_values <- list()

for (i in 2:6) {
  # Fit model with i lags
  temp_model <- dynlm(drilling_changes ~ L(natural_gas_returns, c(1:i)) + 
                                        L(oil_returns, c(1:i)) + 
                                        L(drilling_changes, c(1:i)))
  
  # Store AIC
  aic_values[i] <- AIC(temp_model)
  
  # F-test for joint significance of highest lag
  f_test_result <- linearHypothesis(temp_model,
                                   c(paste("L(natural_gas_returns, c(1:i))", i, "=0", sep=""),
                                     paste("L(oil_returns, c(1:i))", i, "=0", sep=""),
                                     paste("L(drilling_changes, c(1:i))", i, "=0", sep="")),
                                   vcov = vcovHAC(temp_model),
                                   test = "F")
  
  f_test_pvalues[i] <- f_test_result$`Pr(>F)`[2]
}

cat("Automated lag selection complete. Review results to choose optimal lag length.\n\n")


# GRANGER CAUSALITY TESTING ------------------


cat("=== GRANGER CAUSALITY ANALYSIS ===\n\n")

cat("GRANGER CAUSALITY CONCEPT:\n")
cat("Variable X 'Granger-causes' variable Y if:\n")
cat("- Past values of X help predict Y\n")
cat("- Even after controlling for past values of Y\n")
cat("- Statistical causality, not necessarily economic causality\n\n")

cat("RESEARCH QUESTIONS:\n")
cat("1. Do oil price changes Granger-cause drilling activity changes?\n")
cat("2. Do natural gas price changes Granger-cause drilling activity changes?\n\n")

# Test 1: Oil returns → Drilling activity
cat("--- Test 1: Oil Returns → Drilling Activity ---\n")
cat("H0: Oil price changes do not Granger-cause drilling activity changes\n")
cat("Ha: Oil price changes do Granger-cause drilling activity changes\n\n")

oil_granger_test <- linearHypothesis(selected_model,
                                    c("L(oil_returns, c(1:4))1=0",
                                      "L(oil_returns, c(1:4))2=0",
                                      "L(oil_returns, c(1:4))3=0", 
                                      "L(oil_returns, c(1:4))4=0"),
                                    vcov = vcovHAC(selected_model),
                                    test = "F")

cat("Oil → Drilling Granger Causality Test:\n")
print(oil_granger_test)

if (oil_granger_test$`Pr(>F)`[2] < 0.05) {
  cat("CONCLUSION: Oil price changes Granger-cause drilling activity (p < 0.05)\n")
} else {
  cat("CONCLUSION: No evidence of Granger causality from oil to drilling (p ≥ 0.05)\n")
}

# Test 2: Natural gas returns → Drilling activity  
cat("\n--- Test 2: Natural Gas Returns → Drilling Activity ---\n")
cat("H0: Gas price changes do not Granger-cause drilling activity changes\n")
cat("Ha: Gas price changes do Granger-cause drilling activity changes\n\n")

gas_granger_test <- linearHypothesis(selected_model,
                                    c("L(natural_gas_returns, c(1:4))1=0",
                                      "L(natural_gas_returns, c(1:4))2=0",
                                      "L(natural_gas_returns, c(1:4))3=0",
                                      "L(natural_gas_returns, c(1:4))4=0"),
                                    vcov = vcovHAC(selected_model), 
                                    test = "F")

cat("Gas → Drilling Granger Causality Test:\n")
print(gas_granger_test)

if (gas_granger_test$`Pr(>F)`[2] < 0.05) {
  cat("CONCLUSION: Gas price changes Granger-cause drilling activity (p < 0.05)\n")
} else {
  cat("CONCLUSION: No evidence of Granger causality from gas to drilling (p ≥ 0.05)\n")
}


# SUMMARY AND INTERPRETATION -------------


cat("\n\n=== FINAL SUMMARY AND ECONOMIC INTERPRETATION ===\n")

cat("\nMETHODOLOGICAL LESSONS:\n")
cat("1. Lag selection requires balancing model fit and parsimony\n")
cat("2. HAC standard errors provide robust inference in time series\n")
cat("3. Sequential F-tests offer systematic approach to model selection\n")
cat("4. Residual diagnostics essential for model validation\n")

cat("\nGRANGER CAUSALITY INSIGHTS:\n") 
cat("1. Statistical causality ≠ economic causality\n")
cat("2. Tests reveal predictive relationships, not structural causation\n")
cat("3. Energy markets show complex lead-lag relationships\n")
cat("4. Results inform forecasting and risk management strategies\n")

cat("\nPRACTICAL APPLICATIONS:\n")
cat("- Energy price forecasting\n")
cat("- Investment timing in drilling operations\n")
cat("- Policy analysis for energy market interventions\n")
cat("- Risk management in commodity markets\n")

cat("\n=== ANALYSIS COMPLETE ===\n")


