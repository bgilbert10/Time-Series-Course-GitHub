# Comprehensive ARIMA Modeling and Lag Selection Tutorial --------------------
#
# LEARNING OBJECTIVES:
# 1. Understand the process of ARIMA model identification and estimation
# 2. Learn multiple approaches to lag selection in time series models
# 3. Master residual diagnostics for assessing model adequacy
# 4. Generate and evaluate forecasts using holdout samples
# 5. Interpret cyclical patterns through characteristic polynomial roots
# 6. Compare model performance using information criteria
#
# METHODOLOGICAL APPROACH:
# This tutorial follows the Box-Jenkins methodology:
# 1. Model Identification (ACF/PACF analysis, automatic selection)
# 2. Model Estimation (parameter fitting)
# 3. Model Diagnostic Checking (residual analysis)
# 4. Forecasting (prediction and evaluation)
#
# DATA: We analyze two contrasting time series:
# - Industrial drilling activity (highly cyclical economic series)
# - Unemployment rates (macroeconomic indicator with persistence)

# Load required packages
library(forecast)    # Advanced time series forecasting and model selection
library(quantmod)    # Financial and economic data retrieval from FRED/Yahoo
library(caschrono)   # Additional time series analysis functions

# Case Study 1: Drilling Activity ---------------------------------------------

# Data Acquisition - Retrieve drilling activity data from FRED

# Get oil and gas drilling production index
getSymbols("IPN213111S", src="FRED")   # Industrial Production: Drilling Oil and Gas Wells

# FRED does not allow selecting specific dates, so let's pick a fixed window
# so that results don't change every time the code is run
start_date <- "1972-04-01"    
end_date <- "2022-08-03"   

# Extract data within the specified date range
drilling_data <- na.omit(window(IPN213111S, start=start_date, end=end_date))

# Visualize the level data
chartSeries(drilling_data, theme="white", 
           main="Oil and Gas Drilling Production Index")
acf(drilling_data, main="ACF of Drilling Production Index")

# Calculate first differences to achieve stationarity
drilling_changes <- data.frame(na.omit(diff(drilling_data)))
ts.plot(drilling_changes, 
       main="Changes in Drilling Production Index",
       ylab="First Difference")

# --------------------- MODEL IDENTIFICATION --------------------------
# Identify potential ARMA model structures

# Visualize ACF and PACF
acf(drilling_changes, main="ACF of Drilling Changes")
pacf(drilling_changes, main="PACF of Drilling Changes")

# Automatically select optimal AR order using AIC
ar_order <- ar(drilling_changes, order.max=20)
cat("Optimal AR order selected by ar():", ar_order$order, "\n\n")

# --------------------- MODEL ESTIMATION AND DIAGNOSTICS --------------
# Fit various ARMA models and evaluate their performance

# Function to evaluate ARIMA model with diagnostics
evaluate_arima_model <- function(model, model_name, data, gof_lag=36) {
  cat("============================================================\n")
  cat("Model Evaluation:", model_name, "\n")
  cat("============================================================\n")
  
  # Print model summary
  print(model)
  
  # Calculate AIC and BIC
  aic_value <- AIC(model)
  bic_value <- BIC(model)
  
  cat("\nInformation Criteria:\n")
  cat("AIC:", round(aic_value, 2), "\n")
  cat("BIC:", round(bic_value, 2), "\n\n")
  
  # Diagnostic plots
  par(mfrow=c(3,1))
  tsdiag(model, gof=gof_lag)
  par(mfrow=c(1,1))
  
  # More detailed residual analysis
  tsdisplay(residuals(model), 
           main=paste("Residuals of", model_name))
  
  # Box-Ljung test for residual autocorrelation
  box_test <- Box.test(model$residuals, lag=gof_lag)
  cat("Box-Ljung Test (lag =", gof_lag, "):\n")
  cat("Q statistic:", round(box_test$statistic, 4), "\n")
  cat("p-value:", round(box_test$p.value, 4), "\n")
  
  # Adjusted p-value with degrees of freedom correction
  p_df <- gof_lag - length(model$coef)
  adj_p_value <- 1 - pchisq(box_test$statistic, p_df)
  cat("Adjusted p-value (df =", p_df, "):", round(adj_p_value, 4), "\n")
  cat("------------------------------------------------------------\n\n")
  
  # Return model evaluation metrics
  return(list(
    model = model,
    model_name = model_name,
    aic = aic_value,
    bic = bic_value,
    box_test = box_test,
    adj_p_value = adj_p_value
  ))
}

# Model 1: Full AR(15) model (all coefficients estimated)
model_ar15 <- arima(drilling_changes, order=c(15, 0, 0))
eval_ar15 <- evaluate_arima_model(model_ar15, "AR(15) - All Coefficients", drilling_changes)

# Model 2: AR(15) with manually constrained coefficients (version 1)
# Keep only coefficients that were significant in the full model
constrained_coefs1 <- c(NA, 0, 0, 0, 0, 0, 0, NA, NA, 0, 0, 0, 0, NA, NA, NA)  # Last entry is for intercept
model_ar15_constrained1 <- arima(drilling_changes, order=c(15, 0, 0), fixed=constrained_coefs1)
eval_ar15_constrained1 <- evaluate_arima_model(
  model_ar15_constrained1, 
  "AR(15) - Constrained (Version 1)", 
  drilling_changes
)

# Model 3: AR(15) with updated constrained coefficients (version 2)
# Keep coefficients that were close to significant in the full model
constrained_coefs2 <- c(NA, NA, NA, NA, NA, 0, 0, NA, NA, NA, 0, NA, NA, NA, NA, NA)  # Last entry is for intercept
model_ar15_constrained2 <- arima(drilling_changes, order=c(15, 0, 0), fixed=constrained_coefs2)
eval_ar15_constrained2 <- evaluate_arima_model(
  model_ar15_constrained2, 
  "AR(15) - Constrained (Version 2)", 
  drilling_changes
)

# Model 4: Automatically selected ARIMA model for the level data
auto_model_level <- auto.arima(drilling_data)
cat("Auto-ARIMA model for level data:", 
    paste0("ARIMA(", paste(auto_model_level$arma[c(1,6,2)], collapse=","), ")"), 
    "\n\n")

# Model 5: Automatically selected ARIMA model for the differenced data
auto_model_diff <- auto.arima(drilling_changes)
eval_auto_arima <- evaluate_arima_model(
  auto_model_diff, 
  paste0("Auto-ARIMA: ARIMA(", paste(auto_model_diff$arma[c(1,6,2)], collapse=","), ")"), 
  drilling_changes
)

# Model 6: Full search with auto.arima (with parameters to consider more models)
cat("Running comprehensive auto.arima search (may take a few minutes)...\n")
auto_model_comprehensive <- auto.arima(
  drilling_changes, max.p=15, max.order=100, 
  seasonal=FALSE, stepwise=FALSE, trace=TRUE, approximation=FALSE
)

# Extract best model from comprehensive search
best_order <- auto_model_comprehensive$arma[c(1,6,2)]
cat("\nBest model from comprehensive search: ARIMA(", 
    paste(best_order, collapse=","), ")\n\n")

# Estimate the best model with clean parameters
model_best <- arima(drilling_changes, order=best_order, include.mean=FALSE)
eval_best <- evaluate_arima_model(
  model_best, 
  paste0("Best Model: ARIMA(", paste(best_order, collapse=","), ")"), 
  drilling_changes
)

# --------------------- MODEL COMPARISON ----------------------------
# Compare all models on information criteria

# Create a comparison table
model_comparison <- data.frame(
  Model = c(
    "AR(15) - All Coefficients",
    "AR(15) - Constrained (Version 1)",
    "AR(15) - Constrained (Version 2)",
    paste0("Auto-ARIMA: ARIMA(", paste(auto_model_diff$arma[c(1,6,2)], collapse=","), ")"),
    paste0("Best Model: ARIMA(", paste(best_order, collapse=","), ")")
  ),
  AIC = c(
    eval_ar15$aic,
    eval_ar15_constrained1$aic,
    eval_ar15_constrained2$aic,
    eval_auto_arima$aic,
    eval_best$aic
  ),
  BIC = c(
    BIC(model_ar15),
    BIC(model_ar15_constrained1),
    BIC(model_ar15_constrained2),
    BIC(auto_model_diff),
    BIC(model_best)
  ),
  ResidualAutocorrelation = c(
    ifelse(eval_ar15$adj_p_value > 0.05, "No", "Yes"),
    ifelse(eval_ar15_constrained1$adj_p_value > 0.05, "No", "Yes"),
    ifelse(eval_ar15_constrained2$adj_p_value > 0.05, "No", "Yes"),
    ifelse(eval_auto_arima$adj_p_value > 0.05, "No", "Yes"),
    ifelse(eval_best$adj_p_value > 0.05, "No", "Yes")
  )
)

# Display comparison
cat("\nModel Comparison:\n")
cat("================================================================\n")
print(model_comparison)
cat("================================================================\n\n")

# --------------------- FORECASTING ---------------------------------
# Generate and evaluate forecasts from the best model

# Function to generate forecasts and prediction intervals
generate_forecast <- function(model, h=50, include_plot=TRUE, plot_title=NULL) {
  # Generate predictions
  predictions <- predict(model, h)
  
  # Calculate 95% prediction intervals
  lower_ci <- predictions$pred - 1.96 * predictions$se
  upper_ci <- predictions$pred + 1.96 * predictions$se
  
  # Combine into a table
  forecast_table <- cbind(
    Lower_95 = lower_ci,
    Forecast = predictions$pred,
    Upper_95 = upper_ci
  )
  
  # Generate forecast object and plot if requested
  forecast_obj <- forecast(model, h=h)
  
  if (include_plot) {
    plot(forecast_obj, include=50, main=plot_title)
  }
  
  return(list(
    forecast_table = forecast_table,
    forecast_obj = forecast_obj,
    predictions = predictions
  ))
}

# Generate forecasts from the best model
cat("Generating forecasts from the best model...\n\n")
best_forecasts <- generate_forecast(
  model_best, h=50, 
  plot_title="Forecasts from Best Model"
)

# Display forecast values
cat("First 10 forecast periods:\n")
print(head(best_forecasts$forecast_table, 10))

# --------------------- HOLDOUT SAMPLE EVALUATION -------------------
# Evaluate forecast accuracy using a holdout sample

# Function to evaluate forecast accuracy with holdout sample
evaluate_forecast_accuracy <- function(data, model_spec, h=50, holdout_indices=NULL) {
  # Convert data to time series
  data_ts <- as.ts(data)
  
  # Determine holdout indices if not provided
  if (is.null(holdout_indices)) {
    n <- length(data_ts)
    holdout_indices <- (n-h+1):n
  }
  
  # Split data into training and holdout
  training_data <- data_ts[-holdout_indices]
  holdout_data <- data_ts[holdout_indices]
  
  # Fit model on training data
  if ("fixed" %in% names(model_spec)) {
    model <- do.call(arima, c(list(x=training_data), model_spec))
  } else {
    model <- arima(training_data, order=model_spec$order, include.mean=model_spec$include.mean)
  }
  
  # Generate forecasts
  forecasts <- forecast(model, h=length(holdout_indices))
  
  # Plot forecasts and holdout data
  plot(forecasts, main="Forecast Evaluation with Holdout Sample", include=50)
  lines(data_ts, col="black")
  
  # Calculate forecast accuracy measures
  errors <- holdout_data - forecasts$mean
  accuracy_metrics <- c(
    ME = mean(errors),
    RMSE = sqrt(mean(errors^2)),
    MAE = mean(abs(errors)),
    MPE = mean(100 * errors / holdout_data),
    MAPE = mean(100 * abs(errors / holdout_data))
  )
  
  return(list(
    model = model,
    forecasts = forecasts,
    holdout_data = holdout_data,
    errors = errors,
    accuracy_metrics = accuracy_metrics
  ))
}

# Define holdout period
holdout_indices <- 555:604  # Last 50 observations

# Evaluate AR(15) constrained model
cat("Evaluating AR(15) constrained model with holdout sample...\n")
ar15_holdout <- evaluate_forecast_accuracy(
  drilling_changes, 
  list(order=c(15,0,0), fixed=constrained_coefs2),
  holdout_indices=holdout_indices
)

# Evaluate best model
cat("Evaluating best model with holdout sample...\n")
best_holdout <- evaluate_forecast_accuracy(
  drilling_changes, 
  list(order=best_order, include.mean=FALSE),
  holdout_indices=holdout_indices
)

# Compare forecast accuracy
cat("\nForecast Accuracy Comparison:\n")
cat("======================================================\n")
accuracy_comparison <- data.frame(
  AR15_Constrained = ar15_holdout$accuracy_metrics,
  Best_Model = best_holdout$accuracy_metrics
)
print(accuracy_comparison)
cat("======================================================\n\n")

# --------------------- CYCLE ANALYSIS -----------------------------
# Analyze cyclical components via roots of characteristic polynomials

# Function to analyze cycles in ARMA models
analyze_cycles <- function(model, max_coef=NULL) {
  # Extract AR coefficients
  if (is.null(max_coef)) {
    ar_coefs <- model$coef[grep("^ar", names(model$coef))]
  } else {
    ar_coefs <- model$coef[1:max_coef]
  }
  
  # Create characteristic polynomial
  char_poly <- c(1, -ar_coefs)
  
  # Find roots
  roots <- polyroot(char_poly)
  
  # Calculate moduli
  moduli <- Mod(roots)
  
  # Identify complex roots and calculate cycle lengths
  is_complex <- abs(Im(roots)) > 1e-10
  complex_roots <- roots[is_complex]
  
  # Count complex conjugate pairs
  n_pairs <- length(complex_roots) / 2
  
  # Calculate cycle lengths for each complex conjugate pair
  cycle_lengths <- numeric(n_pairs)
  
  cat("\nCycle Analysis Results:\n")
  cat("==============================================\n")
  cat("Number of roots:", length(roots), "\n")
  cat("Number of complex conjugate pairs:", n_pairs, "\n\n")
  
  if (n_pairs > 0) {
    # Process only unique pairs (first half of complex roots)
    for (i in 1:n_pairs) {
      root <- complex_roots[i]
      modulus <- Mod(root)
      real_part <- Re(root)
      
      # Calculate cycle length
      cycle_length <- 2 * pi / acos(real_part / modulus)
      cycle_lengths[i] <- cycle_length
      
      cat("Pair", i, ":\n")
      cat("  Root:", format(root), "\n")
      cat("  Modulus:", round(modulus, 4), "\n")
      cat("  Cycle length:", round(cycle_length, 2), "periods\n\n")
    }
  } else {
    cat("No complex roots found, no cyclical components present.\n")
  }
  
  return(list(
    roots = roots,
    moduli = moduli,
    complex_roots = complex_roots,
    cycle_lengths = cycle_lengths
  ))
}

# Analyze cycles in the best model
cat("Analyzing cycles in the best model (ARIMA(", paste(best_order, collapse=","), ")):\n")
best_cycles <- analyze_cycles(model_best, max_coef=best_order[1])

# Analyze cycles in the AR(15) constrained model
cat("Analyzing cycles in the AR(15) constrained model:\n")
ar15_cycles <- analyze_cycles(model_ar15_constrained2, max_coef=15)

# Case Study 2: Unemployment Rates --------------------------------------------

# Data Acquisition - Retrieve unemployment rate data from FRED

# Get unemployment rate data
getSymbols("UNRATE", src="FRED")

# Select date range
start_date <- "1948-01-01"    
end_date <- "2020-08-01"   

# Extract and process data
unemployment_data <- na.omit(window(UNRATE, start=start_date, end=end_date))
unemployment_rates <- as.numeric(unemployment_data[,1])

# Visualize level data
ts.plot(unemployment_rates, main="US Unemployment Rate", 
       ylab="Unemployment Rate (%)")

# Check stationarity with ACF and PACF
acf(unemployment_rates, main="ACF of Unemployment Rate")
pacf(unemployment_rates, main="PACF of Unemployment Rate")

# Calculate differences for stationarity
unemployment_changes <- na.omit(diff(unemployment_rates))
ts.plot(unemployment_changes, main="Monthly Changes in Unemployment Rate",
       ylab="First Difference")

# --------------------- MODEL IDENTIFICATION --------------------------
# Identify potential ARMA model structures for unemployment data

# Visualize ACF and PACF of differenced data
acf(unemployment_changes, main="ACF of Unemployment Rate Changes")
pacf(unemployment_changes, main="PACF of Unemployment Rate Changes")

# Automatic model selection
ar_order_unemp <- ar(unemployment_changes)
cat("Optimal AR order selected by ar() for unemployment:", 
    ar_order_unemp$order, "\n\n")

arma_selection <- armaselect(unemployment_changes)
cat("ARMA orders suggested by armaselect():\n")
print(head(arma_selection))

auto_arima_unemp <- auto.arima(unemployment_changes)
cat("\nAuto-ARIMA model for unemployment changes:", 
    paste0("ARIMA(", paste(auto_arima_unemp$arma[c(1,6,2)], collapse=","), ")"), 
    "\n\n")

# --------------------- MODEL ESTIMATION AND EVALUATION ---------------
# Fit and evaluate various models for unemployment data

# Model 1: AR(4) with constraints
constrained_coefs_ar4 <- c(0, NA, 0, NA)
model_ar4 <- arima(unemployment_changes, order=c(4, 0, 0), 
                  include.mean=FALSE, fixed=constrained_coefs_ar4)
eval_ar4 <- evaluate_arima_model(model_ar4, "AR(4) - Constrained", unemployment_changes)

# Model 2: MA(1)
model_ma1 <- arima(unemployment_changes, order=c(0, 0, 1))
eval_ma1 <- evaluate_arima_model(model_ma1, "MA(1)", unemployment_changes)

# Model 3: ARMA(1,1)
model_arma11 <- arima(unemployment_changes, order=c(1, 0, 1))
eval_arma11 <- evaluate_arima_model(model_arma11, "ARMA(1,1)", unemployment_changes)

# Compare unemployment models
unemp_comparison <- data.frame(
  Model = c(
    "AR(4) - Constrained",
    "MA(1)",
    "ARMA(1,1)"
  ),
  AIC = c(
    eval_ar4$aic,
    eval_ma1$aic,
    eval_arma11$aic
  ),
  BIC = c(
    BIC(model_ar4),
    BIC(model_ma1),
    BIC(model_arma11)
  ),
  ResidualAutocorrelation = c(
    ifelse(eval_ar4$adj_p_value > 0.05, "No", "Yes"),
    ifelse(eval_ma1$adj_p_value > 0.05, "No", "Yes"),
    ifelse(eval_arma11$adj_p_value > 0.05, "No", "Yes")
  )
)

# Display comparison
cat("\nUnemployment Model Comparison:\n")
cat("================================================================\n")
print(unemp_comparison)
cat("================================================================\n\n")

# --------------------- CYCLE ANALYSIS FOR UNEMPLOYMENT --------------
# Analyze cycles in the AR(4) model

cat("Analyzing cycles in the AR(4) model for unemployment:\n")
ar4_cycles <- analyze_cycles(model_ar4, max_coef=4)

# --------------------- INTEGRATED MODEL AND FORECASTING -------------
# Fit ARIMA model with original (non-differenced) data

# ARIMA(4,1,0) model
model_arima410 <- arima(unemployment_rates, order=c(4,1,0), fixed=constrained_coefs_ar4)
cat("ARIMA(4,1,0) model for unemployment rates:\n")
print(model_arima410)

# Generate forecasts
unemployment_forecasts <- generate_forecast(
  model_arima410, h=48, 
  plot_title="Unemployment Rate Forecasts (48 months)"
)

# --------------------- HOLDOUT EVALUATION FOR UNEMPLOYMENT ----------
# Evaluate forecast accuracy for unemployment

# Define holdout period for unemployment
unemp_holdout_indices <- 861:872  # Last 12 months

# Evaluate ARIMA model
cat("Evaluating ARIMA(4,1,0) model for unemployment with holdout sample...\n")
unemp_holdout <- evaluate_forecast_accuracy(
  unemployment_rates, 
  list(order=c(4,1,0), fixed=constrained_coefs_ar4),
  h=12,
  holdout_indices=unemp_holdout_indices
)

# Display forecast accuracy
cat("\nUnemployment Forecast Accuracy:\n")
cat("======================================================\n")
print(unemp_holdout$accuracy_metrics)
cat("======================================================\n\n")

# --------------------- SUMMARY OF FINDINGS -------------------------
cat("\n===========================================================\n")
cat("SUMMARY OF FINDINGS\n")
cat("===========================================================\n")
cat("1. For drilling activity:\n")
cat("   - Best model: ARIMA(", paste(best_order, collapse=","), ")\n")
cat("   - Cycles of approx.", round(mean(best_cycles$cycle_lengths), 1), "months\n")
cat("   - Forecast accuracy (RMSE):", round(best_holdout$accuracy_metrics["RMSE"], 4), "\n\n")
cat("2. For unemployment:\n")
cat("   - Best model: AR(4) / ARIMA(4,1,0)\n")
cat("   - Cycle of approx.", round(ar4_cycles$cycle_lengths[1], 1), "months\n")
cat("   - Forecast accuracy (RMSE):", round(unemp_holdout$accuracy_metrics["RMSE"], 4), "\n")
cat("-----------------------------------------------------------\n")