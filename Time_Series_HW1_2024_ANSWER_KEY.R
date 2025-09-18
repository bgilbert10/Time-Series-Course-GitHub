# ======================================================================
# TIME SERIES ECONOMETRICS EBGN 594 - HOMEWORK 1 FALL 2024 ANSWER KEY
# ======================================================================

# Load required packages
library(quantmod)     # For financial data
library(fBasics)      # For basic statistics
library(forecast)     # For time series analysis

# Set working directory (adjust as needed)
# setwd("your/working/directory")

# Set seed for reproducibility
set.seed(123)

# ======================================================================
# PART I: LLN AND CLT WITH GAUSS-MARKOV ASSUMPTION VIOLATIONS
# ======================================================================

# This section demonstrates whether coefficient estimates converge to normal
# distributions around their true values when Gauss-Markov assumptions fail.
# We combine Case 5b (omitted variable) from BiasEfficiencyGaussMarkov.R with
# LLN/CLT simulation framework from LLN_CLT_SampleStats.R

cat("PART I: LLN and CLT with Omitted Variable Bias\n")
#cat("=" %x% 50, "\n")

# --- Simulation Parameters ---
n_samples_list <- c(50, 100, 500, 1000, 5000)  # Different sample sizes
n_simulations <- 1000                           # Number of simulations per sample size
true_beta1 <- 0.5                              # True coefficient on x1
true_beta2 <- 0.3                              # True coefficient on x2 (omitted variable)
sigma_error <- 1                               # Error term standard deviation

# Function to simulate omitted variable bias case
simulate_omitted_variable <- function(n, true_beta1, true_beta2, sigma_error) {
  # Generate correlated regressors (x1 and omitted x2)
  x1 <- rnorm(n, mean=0, sd=1)
  x2 <- 0.6*x1 + rnorm(n, mean=0, sd=0.8)  # x2 correlated with x1
  
  # Generate dependent variable (true model includes both x1 and x2)
  epsilon <- rnorm(n, mean=0, sd=sigma_error)
  y <- true_beta1*x1 + true_beta2*x2 + epsilon
  
  # Estimate regression omitting x2 (biased regression)
  model <- lm(y ~ x1)
  beta1_hat <- coef(model)[2]  # Estimated coefficient on x1
  
  return(beta1_hat)
}

# Storage for simulation results
results_omitted <- list()

cat("Running simulations for different sample sizes...\n")

# Run simulations for each sample size
for(i in 1:length(n_samples_list)) {
  n <- n_samples_list[i]
  cat("Sample size:", n, "- Running", n_simulations, "simulations...\n")
  
  # Run simulations
  beta1_estimates <- replicate(n_simulations, 
                              simulate_omitted_variable(n, true_beta1, true_beta2, sigma_error))
  
  # Store results
  results_omitted[[i]] <- list(
    n = n,
    estimates = beta1_estimates,
    mean_estimate = mean(beta1_estimates),
    bias = mean(beta1_estimates) - true_beta1,
    variance = var(beta1_estimates),
    se = sd(beta1_estimates)
  )
}

# --- Analysis and Visualization ---
cat("\nResults Summary:\n")
cat("----------------\n")
cat("True beta1 =", true_beta1, "\n")
cat("Expected bias due to omitted variable = true_beta2 * Cov(x1,x2)/Var(x1)\n")

# Calculate theoretical bias
theoretical_bias <- true_beta2 * 0.6  # Since Cov(x1,x2)/Var(x1) ≈ 0.6
cat("Theoretical bias ≈", theoretical_bias, "\n\n")

# Print results for each sample size
for(i in 1:length(results_omitted)) {
  result <- results_omitted[[i]]
  cat("Sample Size:", result$n, "\n")
  cat("  Mean Estimate:", round(result$mean_estimate, 4), "\n")
  cat("  Bias:", round(result$bias, 4), "\n")
  cat("  Standard Error:", round(result$se, 4), "\n")
  cat("  95% Coverage of True Value:", 
      round(mean(result$estimates >= true_beta1 - 1.96*result$se & 
                result$estimates <= true_beta1 + 1.96*result$se) * 100, 1), "%\n\n")
}

# Plot 1: Distribution of estimates for different sample sizes
par(mfrow=c(2,3))
for(i in 1:length(results_omitted)) {
  result <- results_omitted[[i]]
  hist(result$estimates, breaks=30, main=paste("n =", result$n),
       xlab="Estimated beta1", freq=FALSE, 
       xlim=c(min(unlist(lapply(results_omitted, function(x) x$estimates))),
              max(unlist(lapply(results_omitted, function(x) x$estimates)))))
  
  # Add theoretical normal distribution (if assumptions held)
  x_vals <- seq(par("usr")[1], par("usr")[2], length.out=100)
  lines(x_vals, dnorm(x_vals, true_beta1, result$se), col="blue", lwd=2)
  
  # Add actual distribution mean
  abline(v=result$mean_estimate, col="red", lwd=2)
  abline(v=true_beta1, col="green", lwd=2, lty=2)
  
  if(i == 1) {
    legend("topright", c("True β₁", "Estimated mean", "Normal if unbiased"),
           col=c("green", "red", "blue"), lty=c(2,1,1), lwd=2, cex=0.8)
  }
}

# Plot 2: Bias and variance as function of sample size
par(mfrow=c(1,2))

# Bias plot
plot(n_samples_list, sapply(results_omitted, function(x) x$bias),
     type="b", pch=16, col="red", lwd=2,
     xlab="Sample Size", ylab="Bias", main="Bias vs Sample Size",
     ylim=c(min(sapply(results_omitted, function(x) x$bias))*0.99,
            max(sapply(results_omitted, function(x) x$bias))*1.01))
abline(h=0, col="gray", lty=2)
abline(h=theoretical_bias, col="blue", lty=2, lwd=2)
legend("topright", c("Observed Bias", "Zero Bias", "Theoretical Bias"),
       col=c("red", "gray", "blue"), lty=c(1,2,2), lwd=c(2,1,2))

# Variance plot (should decrease with sample size)
plot(n_samples_list, sapply(results_omitted, function(x) x$variance),
     type="b", pch=16, col="blue", lwd=2,
     xlab="Sample Size", ylab="Variance", main="Variance vs Sample Size")

par(mfrow=c(1,1))

cat("Conclusions for Part I:\n")
cat("1. Bias does NOT disappear with larger sample size (LLN fails for coefficient)\n")
cat("2. Variance decreases with sample size, but distribution centers on biased value\n")
cat("3. CLT still applies, but around the biased value, not true value\n")
cat("4. This demonstrates violation of Gauss-Markov assumptions has serious consequences\n\n")

# ======================================================================
# BONUS: SERIALLY CORRELATED RESIDUALS (Case 4)
# ======================================================================

cat("BONUS EXTENSION: Serially Correlated Residuals\n")
#cat("=" %x% 45, "\n")

# Function to simulate with serially correlated residuals
simulate_serial_correlation <- function(n, true_beta1, sigma_error, rho=0.7) {
  x1 <- rnorm(n, mean=0, sd=1)
  
  # Generate AR(1) errors
  epsilon <- numeric(n)
  epsilon[1] <- rnorm(1, 0, sigma_error)
  for(t in 2:n) {
    epsilon[t] <- rho * epsilon[t-1] + rnorm(1, 0, sigma_error * sqrt(1-rho^2))
  }
  
  y <- true_beta1 * x1 + epsilon
  
  model <- lm(y ~ x1)
  return(coef(model)[2])
}

# Run simulation for serial correlation case
cat("Running simulations with AR(1) errors (rho = 0.7)...\n")

results_serial <- list()
for(i in 1:length(n_samples_list)) {
  n <- n_samples_list[i]
  beta1_estimates <- replicate(n_simulations, 
                              simulate_serial_correlation(n, true_beta1, sigma_error))
  
  results_serial[[i]] <- list(
    n = n,
    estimates = beta1_estimates,
    mean_estimate = mean(beta1_estimates),
    bias = mean(beta1_estimates) - true_beta1,
    variance = var(beta1_estimates)
  )
}

cat("\nSerial Correlation Results:\n")
cat("True beta1 =", true_beta1, "\n")
for(i in 1:length(results_serial)) {
  result <- results_serial[[i]]
  cat("n =", result$n, ": Mean =", round(result$mean_estimate, 4), 
      ", Bias =", round(result$bias, 4), "\n")
}

cat("\nConclusion: Serial correlation preserves unbiasedness but affects efficiency\n")
cat("(estimates are still centered on true value but have higher variance)\n\n")

# ======================================================================
# PART II: FINANCIAL RETURNS ANALYSIS
# ======================================================================

cat("PART II: Financial Returns Analysis\n")
#cat("=" %x% 35, "\n")

# --- Step 1: Download Financial Data ---
cat("Step 1: Downloading financial data...\n")

# Download equity and commodity data
# Example: Oil company (XOM) vs Oil prices (crude oil futures) and S&P 500
getSymbols("XOM", from="2015-01-01")    # ExxonMobil
getSymbols("SPY", from="2015-01-01")    # S&P 500 ETF  
getSymbols("USO", from="2015-01-01")    # Oil ETF (proxy for oil prices)

# Calculate log returns
xom_returns <- 100 * diff(log(Cl(XOM)))
spy_returns <- 100 * diff(log(Cl(SPY)))
uso_returns <- 100 * diff(log(Cl(USO)))

# Remove NAs and align data
returns_data <- na.omit(merge(xom_returns, spy_returns, uso_returns))
colnames(returns_data) <- c("XOM_Returns", "SPY_Returns", "USO_Returns")

cat("Data period:", as.character(start(returns_data)), "to", as.character(end(returns_data)), "\n")
cat("Number of observations:", nrow(returns_data), "\n\n")

# --- Step 2: Summary Statistics ---
cat("Step 2: Summary Statistics\n")
cat("--------------------------\n")

# Calculate basic statistics for each return series
xom_stats <- basicStats(returns_data$XOM_Returns)
spy_stats <- basicStats(returns_data$SPY_Returns)  
uso_stats <- basicStats(returns_data$USO_Returns)

# Create comprehensive summary table
summary_stats <- data.frame(
  Statistic = rownames(xom_stats),
  XOM = round(xom_stats[,1], 4),
  SPY = round(spy_stats[,1], 4),
  USO = round(uso_stats[,1], 4)
)

print(summary_stats)

cat("\nInterpretation:\n")
cat("- Mean returns are close to zero (expected for daily returns)\n")
cat("- Standard deviations show relative volatility (oil stocks typically more volatile)\n")
cat("- Skewness indicates asymmetry in return distributions\n")
cat("- Kurtosis > 3 indicates fat tails (excess kurtosis)\n\n")

# --- Step 3: Normality Testing ---
cat("Step 3: Normality Testing\n")
cat("-------------------------\n")

# Function to perform comprehensive normality testing
test_normality_comprehensive <- function(returns, name) {
  cat("Testing normality for", name, ":\n")
  
  # Jarque-Bera test
  # H0: Returns are normally distributed (skewness=0 and excess kurtosis=0)
  # H1: Returns are not normally distributed
  jb_test <- jarqueberaTest(returns)
  cat("Jarque-Bera Test:\n")
  cat("  H0: Skewness = 0 and Excess Kurtosis = 0 (normal distribution)\n")
  cat("  H1: Not normal distribution\n")
  cat("  Test statistic:", round(jb_test@test$statistic, 4), "\n")
  cat("  p-value:", format(jb_test@test$p.value, scientific=TRUE), "\n")
  cat("  Conclusion:", ifelse(jb_test@test$p.value < 0.05, 
                             "Reject H0 - Not normally distributed", 
                             "Fail to reject H0 - Appears normally distributed"), "\n\n")
  
  # Test skewness separately
  n <- length(returns)
  skew_stat <- skewness(returns)
  skew_se <- sqrt(6/n)
  skew_t <- skew_stat / skew_se
  skew_pval <- 2 * (1 - pnorm(abs(skew_t)))
  
  cat("Skewness Test:\n")
  cat("  H0: Skewness = 0 (symmetric distribution)\n")
  cat("  H1: Skewness ≠ 0 (asymmetric distribution)\n")
  cat("  Skewness:", round(skew_stat, 4), "\n")
  cat("  t-statistic:", round(skew_t, 4), "\n")
  cat("  p-value:", round(skew_pval, 4), "\n")
  cat("  Conclusion:", ifelse(skew_pval < 0.05,
                             "Reject H0 - Significantly skewed",
                             "Fail to reject H0 - Not significantly skewed"), "\n\n")
  
  # Test excess kurtosis separately  
  kurt_stat <- kurtosis(returns)  # This gives excess kurtosis
  kurt_se <- sqrt(24/n)
  kurt_t <- kurt_stat / kurt_se
  kurt_pval <- 2 * (1 - pnorm(abs(kurt_t)))
  
  cat("Excess Kurtosis Test:\n")
  cat("  H0: Excess Kurtosis = 0 (normal tail thickness)\n")
  cat("  H1: Excess Kurtosis ≠ 0 (fat or thin tails)\n")
  cat("  Excess Kurtosis:", round(kurt_stat, 4), "\n")
  cat("  t-statistic:", round(kurt_t, 4), "\n")
  cat("  p-value:", round(kurt_pval, 4), "\n")
  cat("  Conclusion:", ifelse(kurt_pval < 0.05,
                             "Reject H0 - Significant excess kurtosis (fat/thin tails)",
                             "Fail to reject H0 - Normal tail thickness"), "\n\n")
  
  #cat("=" %x% 50, "\n")
}

# Test normality for each series
test_normality_comprehensive(returns_data$XOM_Returns, "XOM Returns")
test_normality_comprehensive(returns_data$SPY_Returns, "SPY Returns")
test_normality_comprehensive(returns_data$USO_Returns, "USO Returns")

# --- Step 4: Autocorrelation Analysis ---
cat("Step 4: Autocorrelation Analysis of Raw Returns\n")
cat("-----------------------------------------------\n")

# Focus on XOM returns (dependent variable for regression)
cat("Analyzing autocorrelation in XOM returns (dependent variable):\n\n")

# Plot ACF and PACF
par(mfrow=c(2,1))
acf(returns_data$XOM_Returns, main="ACF of XOM Returns", lag.max=20)
pacf(returns_data$XOM_Returns, main="PACF of XOM Returns", lag.max=20)
par(mfrow=c(1,1))

# Ljung-Box test for autocorrelation
# H0: No autocorrelation (all autocorrelations up to lag k are zero)
# H1: At least one autocorrelation is non-zero
lb_test <- Box.test(returns_data$XOM_Returns, lag=10, type="Ljung-Box")

cat("Ljung-Box Test for Serial Correlation in XOM Returns:\n")
cat("H0: No autocorrelation (ρ₁ = ρ₂ = ... = ρ₁₀ = 0)\n")
cat("H1: At least one autocorrelation ≠ 0\n")
cat("Test statistic (Q):", round(lb_test$statistic, 4), "\n")
cat("Degrees of freedom:", lb_test$parameter, "\n")
cat("p-value:", round(lb_test$p.value, 4), "\n")
cat("Conclusion:", ifelse(lb_test$p.value < 0.05,
                         "Reject H0 - Significant autocorrelation detected",
                         "Fail to reject H0 - No significant autocorrelation"), "\n")

cat("\nKey Observations:\n")
cat("• ACF pattern shows:", ifelse(lb_test$p.value < 0.05, 
                                  "evidence of autocorrelation", 
                                  "little autocorrelation"), "\n")
cat("• PACF pattern suggests potential AR structure in returns\n")
cat("• Most financial returns show little autocorrelation but may have volatility clustering\n\n")

# --- Step 5: Regression Analysis ---
cat("Step 5: Regression Analysis\n")
cat("---------------------------\n")

# Estimate regression: XOM returns on SPY and USO returns
# This represents a multi-factor model for XOM returns
regression_model <- lm(XOM_Returns ~ SPY_Returns + USO_Returns, data=returns_data)

cat("Regression Model: XOM_Returns = β₀ + β₁*SPY_Returns + β₂*USO_Returns + ε\n\n")
print(summary(regression_model))

cat("\nCoefficient Interpretation:\n")
cat("β₁ (SPY coefficient):", round(coef(regression_model)[2], 4), 
    "- Market beta (sensitivity to market movements)\n")
cat("β₂ (USO coefficient):", round(coef(regression_model)[3], 4), 
    "- Oil exposure (sensitivity to oil price movements)\n")
cat("Both coefficients are", ifelse(summary(regression_model)$coefficients[2,4] < 0.05 & 
                                   summary(regression_model)$coefficients[3,4] < 0.05,
                                   "statistically significant", "not statistically significant"), "\n\n")

# --- Step 6: Residual Autocorrelation Analysis ---
cat("Step 6: Residual Autocorrelation Analysis\n")
cat("-----------------------------------------\n")

# Extract residuals
model_residuals <- residuals(regression_model)

cat("Testing for autocorrelation in regression residuals:\n")
cat("(This is different from Step 4 which tested raw returns)\n\n")

# Plot ACF and PACF of residuals
par(mfrow=c(2,1))
acf(model_residuals, main="ACF of Regression Residuals", lag.max=20)
pacf(model_residuals, main="PACF of Regression Residuals", lag.max=20)
par(mfrow=c(1,1))

# Ljung-Box test on residuals
lb_test_residuals <- Box.test(model_residuals, lag=10, type="Ljung-Box")

cat("Ljung-Box Test for Serial Correlation in Regression Residuals:\n")
cat("H0: No autocorrelation in residuals\n")
cat("H1: Residuals are autocorrelated\n")
cat("Test statistic (Q):", round(lb_test_residuals$statistic, 4), "\n")
cat("p-value:", round(lb_test_residuals$p.value, 4), "\n")
cat("Conclusion:", ifelse(lb_test_residuals$p.value < 0.05,
                         "Reject H0 - Residuals show significant autocorrelation",
                         "Fail to reject H0 - No significant autocorrelation in residuals"), "\n\n")

# Plot residuals over time
plot(as.numeric(model_residuals), type="l", 
     main="Regression Residuals Over Time", 
     xlab="Observation", ylab="Residual")

cat("Residual Analysis Summary:\n")
cat("• Residual autocorrelation would suggest model misspecification\n")
cat("• If present, could indicate need for:\n")
cat("  - Lagged dependent variables\n")
cat("  - Additional explanatory variables\n") 
cat("  - Different model specification\n")
cat("• HAC standard errors could correct for autocorrelation without model changes\n\n")

# --- Additional Analysis: Model Diagnostics ---
cat("Additional Model Diagnostics:\n")
cat("-----------------------------\n")

cat("R-squared:", round(summary(regression_model)$r.squared, 4), 
    "- Proportion of XOM return variance explained\n")
cat("Adjusted R-squared:", round(summary(regression_model)$adj.r.squared, 4), "\n")
cat("Residual standard error:", round(summary(regression_model)$sigma, 4), "\n")
cat("F-statistic:", round(summary(regression_model)$fstatistic[1], 4), 
    "(p-value:", format(pf(summary(regression_model)$fstatistic[1], 
                         summary(regression_model)$fstatistic[2],
                         summary(regression_model)$fstatistic[3], 
                         lower.tail=FALSE), scientific=TRUE), ")\n\n")

# --- Summary and Conclusions ---
cat("OVERALL CONCLUSIONS:\n")
cat("===================\n")
cat("Part I demonstrated that:\n")
cat("• Omitted variable bias persists regardless of sample size\n")
cat("• LLN fails for biased estimators (bias doesn't disappear)\n")
cat("• CLT still applies but around biased values\n\n")

cat("Part II financial analysis showed:\n")
cat("• Financial returns typically exhibit fat tails and are not normally distributed\n")
cat("• Raw returns may show little autocorrelation\n")
cat("• Market factor models can explain significant variation in individual stock returns\n")
cat("• Residual diagnostics are crucial for validating model specifications\n")
cat("• Serial correlation in residuals suggests model improvements needed\n\n")

cat("Key Learning Points:\n")
cat("• Always test model assumptions\n")
cat("• Understand difference between raw data autocorrelation and residual autocorrelation\n")
cat("• Financial data often violates normality assumptions\n")
cat("• Proper hypothesis testing requires stating H0 and H1 clearly\n")

# ======================================================================
# END OF ANSWER KEY
# ======================================================================