# ======================================================================
# BIAS AND EFFICIENCY ILLUSTRATION - GAUSS MARKOV THEOREM
# ======================================================================

# This program illustrates bias and efficiency issues when Gauss-Markov assumptions fail
# Key concepts:
# - Bias: deviation of expected value of sample parameter estimate from "true" population parameter
# - Efficiency: the variance of the sample parameter estimate should be as small as possible
# - Consistency: distribution of sample parameter estimate should converge to population as n increases

# --------------------- SETUP AND PACKAGE LOADING ----------------------
# Load required packages
# install.packages("MASS")  # Uncomment to install if needed
library(MASS)               # For multivariate normal distribution
# install.packages("car")   # Uncomment to install if needed
library(car)                # For scatterplot matrices

# ---------------------- GAUSS-MARKOV THEOREM -------------------------
# OLS is Best (lowest variance) Linear Unbiased Estimator (BLUE) IF:
# 1. True model linear in parameters and residuals: y = b0 + b1*x1 + b2*x2 + e
# 2. X variables (right hand side) are not constants or perfectly correlated
# 3. Residuals "e" have constant variance (homoskedasticity vs. heteroskedastic)
# 4. Residuals "e" are uncorrelated with each other (no peer effects, no serial correlation)
# 5. All X variables are uncorrelated with the residual e (no endogeneity)

# In each case we will run linear regression y = b0 + b1*x1 + b2*x2 + e
# but the "true" model or "data generating process" will violate an assumption

# ================ PART 1: TRUE MODEL IS NOT LINEAR ================

# Set seed for reproducibility
set.seed(826)

# Create covariance matrix for x1, x2, and e 
covariance_matrix <- matrix(c(4,1,0,1,2,0,0,0,1), 3, 3)
covariance_matrix  # Notice e does not covary with x1 or x2 (assumption 5) 
                   # Also x1 and x2 can covary, but not perfectly (assumption 2)

# Define mean vector for x1, x2, and e
mean_vector <- c(10, 3, 0)

# Generate multivariate normal data
simulated_data <- mvrnorm(n=1000, mu=mean_vector, Sigma=covariance_matrix)

# Assign column names and convert to data frame
colnames(simulated_data) <- c("x1", "x2", "e")
simulated_data <- as.data.frame(simulated_data)
head(simulated_data)

# Visualize the distributions
hist(simulated_data$x1, breaks = 100, main="Distribution of x1", 
     xlab="x1", ylab="Frequency")
hist(simulated_data$x2, breaks = 100, main="Distribution of x2",
     xlab="x2", ylab="Frequency") 
hist(simulated_data$e, breaks = 100, main="Distribution of Error Term",
     xlab="e", ylab="Frequency")

# Check sample covariance and correlation matrices
cov(simulated_data)
cor(simulated_data)

# Create scatterplot matrix
scatterplotMatrix(simulated_data)

# Generate non-linear outcome variable
outcome_y <- exp(10 - 6*simulated_data$x1 + 5*simulated_data$x2 + simulated_data$e)
# Note: log(y) is linear in parameters and residual, but y is not

# Plot relationships
par(mfrow=c(2,2))
plot(simulated_data$x1, outcome_y, main="y vs x1", xlab="x1", ylab="y")
plot(simulated_data$x1, log(outcome_y), main="log(y) vs x1", xlab="x1", ylab="log(y)")
plot(simulated_data$x2, outcome_y, main="y vs x2", xlab="x2", ylab="y")
plot(simulated_data$x2, log(outcome_y), main="log(y) vs x2", xlab="x2", ylab="log(y)")
par(mfrow=c(1,1))

# Run linear regression (will be biased due to non-linearity)
model_misspecified <- lm(outcome_y ~ x1 + x2, data=simulated_data)
summary(model_misspecified)

# Run correct model with log transformation
model_correct <- lm(log(outcome_y) ~ x1 + x2, data=simulated_data)
summary(model_correct)
# Note how coefficients in log model are close to true values: 10, -6, 5

# ================ PART 2: PERFECT MULTICOLLINEARITY ================

# Set seed for reproducibility
set.seed(826)

# Create covariance matrix for x1 and e
covariance_matrix <- matrix(c(4,0,0,1), 2, 2)
covariance_matrix  # Notice e does not covary with x1 (assumption 5)

# Define mean vector for x1 and e
mean_vector <- c(10, 0)

# Generate multivariate normal data
simulated_data <- mvrnorm(n=1000, mu=mean_vector, Sigma=covariance_matrix)

# Create x2 as perfect multiple of x1 (violates assumption 2)
x2_values <- 7 * simulated_data[,1]
simulated_data <- as.data.frame(cbind(simulated_data[,1], x2_values, simulated_data[,2]))

# Assign column names
colnames(simulated_data) <- c("x1", "x2", "e")
head(simulated_data)

# Check sample covariance and correlation
cov(simulated_data)
cor(simulated_data)

# Create scatterplot matrix
scatterplotMatrix(simulated_data)

# Generate linear outcome variable
outcome_y <- 10 - 6*simulated_data$x1 + 5*simulated_data$x2 + simulated_data$e
# Note: y is linear in parameters (satisfies assumption 1)

# Plot relationships
par(mfrow=c(1,2))
plot(simulated_data$x1, outcome_y, main="y vs x1", xlab="x1", ylab="y")
plot(simulated_data$x2, outcome_y, main="y vs x2", xlab="x2", ylab="y")
par(mfrow=c(1,1))

# Run linear regression with perfect multicollinearity
# Note: Some stats packages will not produce output due to perfect correlation
model_multicollinear <- lm(outcome_y ~ x1 + x2, data=simulated_data)
summary(model_multicollinear)
# Why an intercept of 10 and coefficient of 29? Because -6 + 5*7 = 29

# Demonstrate that intercept-only model works (intercept can be constant)
mean(outcome_y)
summary(lm(outcome_y ~ 1))

# CASE: CONSTANT VARIABLE
# Set seed for reproducibility
set.seed(826)

# Covariance matrix with x2 having zero variance (constant)
covariance_matrix <- matrix(c(4,0,0,0,0,0,0,0,1), 3, 3)
covariance_matrix  # Notice e does not covary with x1 or x2 (assumption 5)
                   # But x2 has no variance (violates assumption 2)

# Define mean vector
mean_vector <- c(10, 3, 0)

# Generate data
simulated_data <- mvrnorm(n=1000, mu=mean_vector, Sigma=covariance_matrix)

# Assign column names and convert to data frame
colnames(simulated_data) <- c("x1", "x2", "e")
simulated_data <- as.data.frame(simulated_data)
head(simulated_data)

# Check sample statistics
cov(simulated_data)
cor(simulated_data)
#scatterplotMatrix(simulated_data)

# Generate linear outcome
outcome_y <- 10 - 6*simulated_data$x1 + 5*simulated_data$x2 + simulated_data$e
par(mfrow=c(1,2))
plot(simulated_data$x1, outcome_y, main="y vs x1", xlab="x1", ylab="y")
plot(simulated_data$x2, outcome_y, main="y vs x2", xlab="x2", ylab="y")
par(mfrow=c(1,1))

# Run linear regression
model_constant <- lm(outcome_y ~ x1 + x2, data=simulated_data)
summary(model_constant)
# Why an intercept of 25 and coefficient of -6? Because 10 + 5*3 = 25

# ================ PART 3: HETEROSKEDASTICITY ================

# Set seed for reproducibility
set.seed(826)

# Covariance matrix for x1 and x2
covariance_matrix <- matrix(c(4,1,1,2), 2, 2)
covariance_matrix       

# Define mean vector
mean_vector <- c(10, 3)

# Generate data for predictors
predictors <- mvrnorm(n=1000, mu=mean_vector, Sigma=covariance_matrix)

# Create two different error structures
# 1. Homoskedastic errors (constant variance)
homoskedastic_errors <- rnorm(n=1000, mean=0, sd=1)

# 2. Heteroskedastic errors (variance depends on x1 and x2)
error_variance <- (homoskedastic_errors^2) * (predictors[,1]^2 + predictors[,2]^2)
heteroskedastic_errors <- rnorm(n=1000, mean=0, sd=sqrt(error_variance))

# Create data frames for both cases
homoskedastic_data <- as.data.frame(cbind(predictors, homoskedastic_errors))
colnames(homoskedastic_data) <- c("x1", "x2", "e1")

heteroskedastic_data <- as.data.frame(cbind(predictors, heteroskedastic_errors))
colnames(heteroskedastic_data) <- c("x1", "x2", "e2")

# Visualize the data
scatterplotMatrix(homoskedastic_data)
scatterplotMatrix(heteroskedastic_data)

# Generate outcome variables
outcome_y1 <- 10 - 6*homoskedastic_data$x1 + 5*homoskedastic_data$x2 + homoskedastic_data$e1
outcome_y2 <- 10 - 6*heteroskedastic_data$x1 + 5*heteroskedastic_data$x2 + heteroskedastic_data$e2

# Plot outcomes
par(mfrow=c(1,2))
plot(homoskedastic_data$x1, outcome_y1, main="y1 (Homoskedastic) vs x1", 
     xlab="x1", ylab="y1")
plot(heteroskedastic_data$x1, outcome_y2, main="y2 (Heteroskedastic) vs x1", 
     xlab="x1", ylab="y2")
par(mfrow=c(1,1))

# Run linear regressions
model_homoskedastic <- lm(outcome_y1 ~ x1 + x2, data=homoskedastic_data)
summary(model_homoskedastic)

model_heteroskedastic <- lm(outcome_y2 ~ x1 + x2, data=heteroskedastic_data)
summary(model_heteroskedastic)
# Note differences in coefficient standard errors, residual std error, and R-squared

# ================ PART 4: SERIAL CORRELATION ================

# Set seed for reproducibility
set.seed(826)

# Covariance matrix for x1 and x2
covariance_matrix <- matrix(c(4,1,1,2), 2, 2)
covariance_matrix       

# Define mean vector
mean_vector <- c(10, 3)

# Generate data for predictors
predictors <- mvrnorm(n=1000, mu=mean_vector, Sigma=covariance_matrix)

# Create two different error structures
# 1. Independent errors (no serial correlation)
independent_errors <- rnorm(n=1000, mean=0, sd=1)

# 2. Serially correlated errors (AR(1) process)
correlated_errors <- arima.sim(model=list(ar=c(0.8)), n=1000, sd=1)

# Create data frames for both cases
independent_data <- as.data.frame(cbind(predictors, independent_errors))
colnames(independent_data) <- c("x1", "x2", "e1")

correlated_data <- as.data.frame(cbind(predictors, correlated_errors))
colnames(correlated_data) <- c("x1", "x2", "e2")

# Visualize the data
scatterplotMatrix(independent_data)
scatterplotMatrix(correlated_data)

# Generate outcome variables
outcome_y1 <- 10 - 6*independent_data$x1 + 5*independent_data$x2 + independent_data$e1
outcome_y2 <- 10 - 6*correlated_data$x1 + 5*correlated_data$x2 + correlated_data$e2

# Plot outcomes
par(mfrow=c(1,2))
plot(independent_data$x1, outcome_y1, main="y1 (Independent Errors) vs x1", 
     xlab="x1", ylab="y1")
plot(correlated_data$x1, outcome_y2, main="y2 (Correlated Errors) vs x1", 
     xlab="x1", ylab="y2")
par(mfrow=c(1,1))

# Run linear regressions
model_independent <- lm(outcome_y1 ~ x1 + x2, data=independent_data)
summary(model_independent)

model_correlated <- lm(outcome_y2 ~ x1 + x2, data=correlated_data)
summary(model_correlated)
# Note differences in standard errors and other statistics

# ================ PART 5: ENDOGENEITY ================

# CASE 5a: DIRECT CORRELATION BETWEEN X AND ERROR
# Set seed for reproducibility
set.seed(826)

# Create a random positive definite covariance matrix
n <- 3
random_matrix <- matrix(runif(n^2)*2-1, ncol=n) 
covariance_matrix <- t(random_matrix) %*% random_matrix
covariance_matrix  # Notice e covaries with x1 and x2 (violates assumption 5)

# Define mean vector
mean_vector <- c(10, 3, 0)

# Generate data
endogenous_data <- mvrnorm(n=1000, mu=mean_vector, Sigma=covariance_matrix)

# Assign column names and convert to data frame
colnames(endogenous_data) <- c("x1", "x2", "e1")
endogenous_data <- as.data.frame(endogenous_data)
head(endogenous_data)

# Visualize the data
hist(endogenous_data$x1, breaks = 100, main="Distribution of x1", 
     xlab="x1", ylab="Frequency")
hist(endogenous_data$x2, breaks = 100, main="Distribution of x2",
     xlab="x2", ylab="Frequency")
hist(endogenous_data$e1, breaks = 100, main="Distribution of Error Term",
     xlab="e1", ylab="Frequency")

# Sample statistics
cov(endogenous_data)
cor(endogenous_data)
scatterplotMatrix(endogenous_data)

# Generate outcome variable
outcome_y1 <- 10 - 6*endogenous_data$x1 + 5*endogenous_data$x2 + endogenous_data$e1

# Plot relationships
par(mfrow=c(1,2))
plot(endogenous_data$x1, outcome_y1, main="y vs x1", xlab="x1", ylab="y")
plot(endogenous_data$x2, outcome_y1, main="y vs x2", xlab="x2", ylab="y")
par(mfrow=c(1,1))

# Run linear regression with endogenous variables
model_endogenous <- lm(outcome_y1 ~ x1 + x2, data=endogenous_data)
summary(model_endogenous)
# Note: Coefficient estimates are biased (differ from true values -6 and 5)

# CASE 5b: OMITTED VARIABLE BIAS
# Set seed for reproducibility
set.seed(826)

# Covariance matrix for x1, x2, and e
covariance_matrix <- matrix(c(4,1,0,1,2,0,0,0,1), 3, 3)
covariance_matrix  # e does not covary with x1 or x2 (satisfies assumption 5)

# Define mean vector
mean_vector <- c(10, 3, 0)

# Generate data
complete_data <- mvrnorm(n=1000, mu=mean_vector, Sigma=covariance_matrix)

# Assign column names and convert to data frame
colnames(complete_data) <- c("x1", "x2", "e")
complete_data <- as.data.frame(complete_data)

# Generate outcome variable
outcome_y2 <- 10 - 6*complete_data$x1 + 5*complete_data$x2 + complete_data$e

# Run correct regression with both variables
model_complete <- lm(outcome_y2 ~ x1 + x2, data=complete_data)
summary(model_complete)

# Run regression with omitted variable x1
model_omit_x1 <- lm(outcome_y2 ~ x2, data=complete_data)
summary(model_omit_x1)

# Run regression with omitted variable x2
model_omit_x2 <- lm(outcome_y2 ~ x1, data=complete_data)
summary(model_omit_x2)
# Note: Coefficients are biased due to omitted variable bias when x1 and x2 covary

# CASE 5c: AUTOCORRELATED DEPENDENT VARIABLE
# Set seed for reproducibility
set.seed(826)

# Generate an autocorrelated error term (AR(1) process)
ar_errors <- arima.sim(model=list(ar=c(0.8)), n=999, sd=1)

# Generate an autocorrelated dependent variable
ar_outcome <- numeric(1000)
ar_outcome[1] <- rnorm(n=1, mean=10, sd=1)  # Initial value

# Generate AR(1) process: y_t = 10 + 0.4*y_{t-1} + e_t
for(i in 2:1000) {
  ar_outcome[i] <- 10 + 0.4*ar_outcome[i-1] + ar_errors[i-1]
}

# Estimate AR(1) model
arima(ar_outcome, order=c(1,0,0))