# ======================================================================
# LAW OF LARGE NUMBERS (LLN) AND CENTRAL LIMIT THEOREM (CLT)
# ======================================================================

# This program demonstrates the LLN and CLT applied to various sample statistics:
# - Mean, variance, skewness, kurtosis
# - Regression coefficients
# - Tests for normality on simulated and financial data

# --------------------- SETUP AND PACKAGE LOADING ----------------------
# Load required packages
if(!require(pacman)) install.packages("pacman")
pacman::p_load(fBasics, quantmod, MASS)

# ======================= POPULATION DEFINITION =======================

# Set seed for reproducibility
set.seed(12345)

# Define population parameters
population_mean <- 2.5
population_sd <- 0.5
population_skew <- 0.0
population_kurt <- 0.0  # Excess kurtosis (subtract 3 to get sample kurtosis)

# Generate a large population (million observations)
population_data <- rnorm(1000000, mean = population_mean, sd = population_sd)

# Visualize population distribution
hist(population_data, breaks = 100, main = "Population Distribution", 
     xlab = "Value", ylab = "Frequency")
abline(v = population_mean, lwd = 3, lty = 2, col = "red")

# ====================== LAW OF LARGE NUMBERS (LLN) ===================

# Demonstrate convergence of sample moments to population moments
# as sample size increases

# Sample sizes to demonstrate
sample_sizes <- c(10, 100, 1000, 10000)

# Calculate moments for sample size 10
sample_10 <- sample(population_data, 10, replace = TRUE)
mean_10 <- mean(sample_10)
sd_10 <- sd(sample_10)
skew_10 <- skewness(sample_10)
kurt_10 <- kurtosis(sample_10, method = "moment")

# Calculate moments for sample size 100
sample_100 <- sample(population_data, 100, replace = TRUE)
mean_100 <- mean(sample_100)
sd_100 <- sd(sample_100)
skew_100 <- skewness(sample_100)
kurt_100 <- kurtosis(sample_100, method = "moment")

# Calculate moments for sample size 1000
sample_1000 <- sample(population_data, 1000, replace = TRUE)
mean_1000 <- mean(sample_1000)
sd_1000 <- sd(sample_1000)
skew_1000 <- skewness(sample_1000)
kurt_1000 <- kurtosis(sample_1000, method = "moment")

# Calculate moments for sample size 10000
sample_10000 <- sample(population_data, 10000, replace = TRUE)
mean_10000 <- mean(sample_10000)
sd_10000 <- sd(sample_10000)
skew_10000 <- skewness(sample_10000)
kurt_10000 <- kurtosis(sample_10000, method = "moment")

# Display results in a data frame
lln_results <- data.frame(
  SampleSize = sample_sizes,
  Mean = c(mean_10, mean_100, mean_1000, mean_10000),
  StdDev = c(sd_10, sd_100, sd_1000, sd_10000),
  Skewness = c(skew_10, skew_100, skew_1000, skew_10000),
  Kurtosis = c(kurt_10, kurt_100, kurt_1000, kurt_10000)
)

# Add population values for comparison
population_row <- data.frame(
  SampleSize = "Population",
  Mean = population_mean,
  StdDev = population_sd,
  Skewness = population_skew,
  Kurtosis = population_kurt
)

# Combine results
lln_results <- rbind(lln_results, population_row)
print(lln_results)

# ====================== CENTRAL LIMIT THEOREM =======================

# Demonstrates convergence of sampling distributions to normal
# regardless of original distribution shape

# Define simulation parameters
realizations <- 10000  # Number of samples or bootstrap realizations
sample_size <- 1000    # Set to 10, 100, 1000, or 10000 to see effect of sample size

# Initialize lists to store results
sample_means <- list()
sample_sds <- list()
sample_skews <- list()
sample_kurts <- list()

# Generate samples and calculate statistics
for(i in 1:realizations) {
  current_sample <- sample(population_data, sample_size, replace = TRUE)
  sample_means[[i]] <- mean(current_sample)
  sample_sds[[i]] <- sd(current_sample)
  sample_skews[[i]] <- skewness(current_sample)
  sample_kurts[[i]] <- kurtosis(current_sample)
}

# Convert lists to vectors
sample_means <- unlist(sample_means)
sample_sds <- unlist(sample_sds)
sample_skews <- unlist(sample_skews)
sample_kurts <- unlist(sample_kurts)

# ================ THEORETICAL DISTRIBUTIONS =================

# Compute theoretical distributions for sample statistics

# 1. Mean: Normal with mean=pop_mean and SD=pop_sd/sqrt(N)
mean_x_range <- seq(0, 5, 0.001)
mean_density <- dnorm(x = mean_x_range, 
                      mean = population_mean, 
                      sd = population_sd/sqrt(sample_size))

# 2. Standard Deviation: Derived from chi-squared distribution
sd_range <- seq(0, 1, 0.001)
sd_cdf <- sqrt(((sample_size-1) * population_sd^2) / qchisq(p = sd_range, df = sample_size-1))

# Calculate SD density by differentiating CDF
sd_density <- numeric(length(sd_range)-1)
for (i in 2:length(sd_range)) {
  sd_density[i] <- (sd_range[i] - sd_range[i-1]) / (sd_cdf[i-1] - sd_cdf[i])
}

# 3. Skewness: Normal with mean=0 and SD=sqrt(6/N)
skew_x_range <- seq(-3, 3, 0.001)
skew_density <- dnorm(x = skew_x_range, 
                      mean = population_skew, 
                      sd = sqrt(6/sample_size))

# 4. Kurtosis: Normal with mean=0 and SD=sqrt(24/N)
kurt_x_range <- seq(-3, 3, 0.001)
kurt_density <- dnorm(x = kurt_x_range, 
                      mean = population_kurt, 
                      sd = sqrt(24/sample_size))

# ================ PLOT SAMPLING DISTRIBUTIONS =================

# Plot distribution of sample means
plot_sampling_distribution <- function(sample_stat, theoretical_x, theoretical_y, 
                                     title, xlab, true_value) {
  # Plot histogram
  hist(sample_stat, breaks = 100, 
       main = title,
       xlab = xlab, 
       freq = FALSE)
  
  # Add vertical line at true parameter value
  abline(v = true_value, lwd = 3, col = "red", lty = 2)
  
  # Add theoretical density curve
  lines(theoretical_x, theoretical_y, col = "blue", lwd = 2)
  
  # Add legend
  legend("topright", 
         legend = c("Empirical distribution", "Theoretical distribution", "Population value"),
         col = c("black", "blue", "red"), 
         lty = c(1, 1, 2),
         lwd = c(1, 2, 3))
}

# Plot all distributions
par(mfrow = c(2, 2))

# Mean
plot_sampling_distribution(
  sample_means, 
  mean_x_range, 
  mean_density,
  paste("Distribution of Sample Means (n =", sample_size, ")"),
  "Sample Mean",
  population_mean
)

# Standard Deviation
plot_sampling_distribution(
  sample_sds, 
  sd_cdf, 
  sd_density,
  paste("Distribution of Sample Std. Dev. (n =", sample_size, ")"),
  "Sample Standard Deviation",
  population_sd
)

# Skewness
plot_sampling_distribution(
  sample_skews, 
  skew_x_range, 
  skew_density,
  paste("Distribution of Sample Skewness (n =", sample_size, ")"),
  "Sample Skewness",
  population_skew
)

# Kurtosis
plot_sampling_distribution(
  sample_kurts, 
  kurt_x_range, 
  kurt_density,
  paste("Distribution of Sample Kurtosis (n =", sample_size, ")"),
  "Sample Kurtosis",
  population_kurt
)

par(mfrow = c(1, 1))

# ============== CLT FOR NON-NORMAL DISTRIBUTION ===============

# Demonstrate CLT for a non-normal (Bernoulli) distribution
# Create a binary population (coin flips)
bernoulli_population <- sample(c(0, 1), 1000000, replace = TRUE)

# Plot the population distribution
hist(bernoulli_population, breaks = 3, main = "Non-normal Population (Bernoulli)",
     xlab = "Value", ylab = "Frequency")
abline(v = mean(bernoulli_population), lwd = 3, lty = 3, col = "red")

# Collect sample means from the Bernoulli population
bernoulli_means <- list()
for(i in 1:realizations) {
  bernoulli_means[[i]] <- mean(sample(bernoulli_population, sample_size, replace = TRUE))
}
bernoulli_means <- unlist(bernoulli_means)

# Plot the sampling distribution of means
hist(bernoulli_means, breaks = 50, 
     main = "CLT: Distribution of Bernoulli Sample Means", 
     xlab = paste("Average of samples with n =", sample_size),
     freq = FALSE)
abline(v = 0.5, lwd = 3, col = "red")

# Add theoretical normal curve
theoretical_x <- seq(min(bernoulli_means), max(bernoulli_means), length.out = 100)
theoretical_y <- dnorm(theoretical_x, mean = 0.5, sd = sqrt(0.25/sample_size))
lines(theoretical_x, theoretical_y, col = "blue", lwd = 2)

legend("topright", 
       legend = c("Empirical distribution", "Theoretical distribution", "Population mean"),
       col = c("black", "blue", "red"), 
       lty = c(1, 1, 1),
       lwd = c(1, 2, 3))

# ================ LLN AND CLT FOR REGRESSION ==================

# Create a population for regression
# Use the population_data as X
x_variable <- population_data

# Define true regression parameters
true_intercept <- 0.5
true_slope <- 0.8

# Create error term
error_term <- rnorm(1000000, mean = 0, sd = 2)

# Generate dependent variable
y_variable <- true_intercept + true_slope * x_variable + error_term

# Store X and Y together for sampling
regression_data <- cbind(y_variable, x_variable)

# Run regression on full population
population_model <- lm(y_variable ~ x_variable)
summary(population_model)

# =================== LLN FOR REGRESSION ======================

# Demonstrate LLN for regression coefficients
sample_sizes <- c(10, 100, 1000, 10000)

# Function to sample and run regression
sample_regression <- function(data, size) {
  sample_indices <- sample(nrow(data), size, replace = TRUE)
  sample_data <- data[sample_indices, ]
  model <- lm(sample_data[, 1] ~ sample_data[, 2])
  return(coef(model))
}

# Sample size 10
sample_10_coefs <- sample_regression(regression_data, 10)

# Sample size 100
sample_100_coefs <- sample_regression(regression_data, 100)

# Sample size 1000
sample_1000_coefs <- sample_regression(regression_data, 1000)

# Sample size 10000
sample_10000_coefs <- sample_regression(regression_data, 10000)

# Create results table
regression_lln <- data.frame(
  SampleSize = sample_sizes,
  Intercept = c(sample_10_coefs[1], sample_100_coefs[1], 
               sample_1000_coefs[1], sample_10000_coefs[1]),
  Slope = c(sample_10_coefs[2], sample_100_coefs[2], 
           sample_1000_coefs[2], sample_10000_coefs[2])
)

# Add true values
regression_lln <- rbind(regression_lln, 
                      data.frame(SampleSize = "True", 
                                Intercept = true_intercept, 
                                Slope = true_slope))
print(regression_lln)

# =================== CLT FOR REGRESSION ======================

# Demonstrate CLT for regression coefficients
intercept_samples <- list()
slope_samples <- list()

for(i in 1:realizations) {
  sample_indices <- sample(nrow(regression_data), sample_size, replace = TRUE)
  sample_data <- regression_data[sample_indices, ]
  model <- lm(sample_data[, 1] ~ sample_data[, 2])
  intercept_samples[[i]] <- coef(model)[1]
  slope_samples[[i]] <- coef(model)[2]
}

intercept_samples <- unlist(intercept_samples)
slope_samples <- unlist(slope_samples)

# Plot sampling distributions of regression coefficients
par(mfrow = c(1, 2))

# Intercept
hist(intercept_samples, breaks = 50, 
     main = "Distribution of Regression Intercepts", 
     xlab = paste("Intercepts with n =", sample_size))
abline(v = true_intercept, lwd = 3, col = "red")

# Add theoretical normal curve
theoretical_x <- seq(min(intercept_samples), max(intercept_samples), length.out = 100)
theoretical_y <- dnorm(theoretical_x, mean = true_intercept, 
                     sd = sd(intercept_samples))
lines(theoretical_x, theoretical_y, col = "blue", lwd = 2)

# Slope
hist(slope_samples, breaks = 50, 
     main = "Distribution of Regression Slopes", 
     xlab = paste("Slopes with n =", sample_size))
abline(v = true_slope, lwd = 3, col = "red")

# Add theoretical normal curve
theoretical_x <- seq(min(slope_samples), max(slope_samples), length.out = 100)
theoretical_y <- dnorm(theoretical_x, mean = true_slope, 
                     sd = sd(slope_samples))
lines(theoretical_x, theoretical_y, col = "blue", lwd = 2)

par(mfrow = c(1, 1))