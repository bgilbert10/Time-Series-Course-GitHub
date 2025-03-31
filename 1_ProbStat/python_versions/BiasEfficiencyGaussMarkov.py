#!/usr/bin/env python
# -*- coding: utf-8 -*-

# =============================================================================
# BIAS AND EFFICIENCY ILLUSTRATION WHEN GAUSS-MARKOV ASSUMPTIONS FAIL
# =============================================================================

# Import required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import multivariate_normal
from statsmodels.graphics.tsaplots import plot_acf

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_mvnorm(n, mu, sigma):
    """
    Generate multivariate normal random variables
    
    Parameters:
    -----------
    n : int
        Number of observations
    mu : array-like
        Mean vector
    sigma : array-like
        Covariance matrix
        
    Returns:
    --------
    numpy.ndarray
        Matrix of random variables
    """
    return multivariate_normal.rvs(mean=mu, cov=sigma, size=n)

def scatterplot_matrix(df, title=None):
    """
    Create a scatterplot matrix with histograms on the diagonal
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing variables
    title : str, optional
        Title for the plot
    """
    sns.set(style="ticks")
    plot = sns.pairplot(df, diag_kind="kde")
    if title:
        plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()

def run_ols_analysis(X, y, feature_names=None, title=None):
    """
    Run OLS regression and print summary statistics
    
    Parameters:
    -----------
    X : array-like
        Explanatory variables
    y : array-like
        Dependent variable
    feature_names : list, optional
        Names of explanatory variables
    title : str, optional
        Title for the regression output
        
    Returns:
    --------
    statsmodels.regression.linear_model.RegressionResults
        Regression results
    """
    # Add constant
    X_with_const = sm.add_constant(X)
    
    # Create model and fit
    model = sm.OLS(y, X_with_const)
    results = model.fit()
    
    # Print results
    if title:
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)
    
    print(results.summary())
    return results

# =============================================================================
# PART 1: MODEL NOT LINEAR IN PARAMETERS
# =============================================================================

def part1_nonlinear_model():
    """
    Illustrate bias when true model is not linear in parameters
    """
    print("\n" + "=" * 70)
    print("PART 1: MODEL NOT LINEAR IN PARAMETERS")
    print("=" * 70)
    
    # Set seed for reproducibility
    np.random.seed(826)
    
    # Define covariance matrix and mean vector
    sigma = np.array([
        [4, 1, 0],
        [1, 2, 0],
        [0, 0, 1]
    ])
    
    mu = np.array([10, 3, 0])
    
    # Generate multivariate normal data
    data = generate_mvnorm(1000, mu, sigma)
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=['x1', 'x2', 'e'])
    
    # Display sample statistics
    print("\nCovariance Matrix:")
    print(df.cov())
    print("\nCorrelation Matrix:")
    print(df.corr())
    
    # Plot distributions
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(df['x1'], bins=50)
    plt.title('Distribution of x1')
    
    plt.subplot(1, 3, 2)
    plt.hist(df['x2'], bins=50)
    plt.title('Distribution of x2')
    
    plt.subplot(1, 3, 3)
    plt.hist(df['e'], bins=50)
    plt.title('Distribution of e')
    
    plt.tight_layout()
    plt.show()
    
    # Create scatterplot matrix
    scatterplot_matrix(df, title="Scatterplot Matrix of Variables")
    
    # Generate true y (outcome)
    # True model: y = exp(10 - 6*x1 + 5*x2 + e)
    df['y'] = np.exp(10 - 6*df['x1'] + 5*df['x2'] + df['e'])
    
    # Plot relationships
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.scatter(df['x1'], df['y'], alpha=0.5)
    plt.title('y vs x1')
    plt.xlabel('x1')
    plt.ylabel('y')
    
    plt.subplot(2, 2, 2)
    plt.scatter(df['x1'], np.log(df['y']), alpha=0.5)
    plt.title('log(y) vs x1')
    plt.xlabel('x1')
    plt.ylabel('log(y)')
    
    plt.subplot(2, 2, 3)
    plt.scatter(df['x2'], df['y'], alpha=0.5)
    plt.title('y vs x2')
    plt.xlabel('x2')
    plt.ylabel('y')
    
    plt.subplot(2, 2, 4)
    plt.scatter(df['x2'], np.log(df['y']), alpha=0.5)
    plt.title('log(y) vs x2')
    plt.xlabel('x2')
    plt.ylabel('log(y)')
    
    plt.tight_layout()
    plt.show()
    
    # Run linear regression on misspecified model
    X = df[['x1', 'x2']]
    y = df['y']
    run_ols_analysis(X, y, title="Linear Regression on Non-Linear Model")
    
    # Run correct regression (log transformation)
    log_y = np.log(df['y'])
    run_ols_analysis(X, log_y, title="Linear Regression on Log-Transformed Model")

# =============================================================================
# PART 2: PERFECT MULTICOLLINEARITY
# =============================================================================

def part2_perfect_correlation():
    """
    Illustrate issues when variables are constants or perfectly correlated
    """
    print("\n" + "=" * 70)
    print("PART 2: PERFECT MULTICOLLINEARITY")
    print("=" * 70)
    
    # Set seed for reproducibility
    np.random.seed(826)
    
    # Define covariance matrix and mean vector for x1 and e
    sigma = np.array([
        [4, 0],
        [0, 1]
    ])
    
    mu = np.array([10, 0])
    
    # Generate multivariate normal data
    data = generate_mvnorm(1000, mu, sigma)
    
    # Create DataFrame with x1 and e
    df = pd.DataFrame()
    df['x1'] = data[:, 0]
    df['e'] = data[:, 1]
    
    # Create x2 as multiple of x1
    df['x2'] = 7 * df['x1']
    
    # Display sample statistics
    print("\nCovariance Matrix:")
    print(df.cov())
    print("\nCorrelation Matrix:")
    print(df.corr())
    
    # Create scatterplot matrix
    scatterplot_matrix(df, title="Perfect Correlation between x1 and x2")
    
    # Generate true y (outcome)
    df['y'] = 10 - 6*df['x1'] + 5*df['x2'] + df['e']
    
    # Plot relationships
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(df['x1'], df['y'], alpha=0.5)
    plt.title('y vs x1')
    plt.xlabel('x1')
    plt.ylabel('y')
    
    plt.subplot(1, 2, 2)
    plt.scatter(df['x2'], df['y'], alpha=0.5)
    plt.title('y vs x2')
    plt.xlabel('x2')
    plt.ylabel('y')
    
    plt.tight_layout()
    plt.show()
    
    # Run linear regression with perfect multicollinearity
    try:
        X = df[['x1', 'x2']]
        y = df['y']
        run_ols_analysis(X, y, title="Linear Regression with Perfect Multicollinearity")
    except np.linalg.LinAlgError:
        print("\nLinear regression failed due to perfect multicollinearity between x1 and x2")
        print("x2 = 7 * x1, so they are perfectly correlated")
    
    # Mean of y and intercept-only model
    print(f"\nMean of y: {np.mean(df['y'])}")
    X_intercept = np.ones((len(df), 1))
    run_ols_analysis(X_intercept, y, title="Intercept-Only Model")
    
    # Constant variable case
    print("\n" + "=" * 70)
    print("CASE: x2 IS A CONSTANT")
    print("=" * 70)
    
    # Generate new data where x2 is constant
    sigma = np.array([
        [4, 0, 0],
        [0, 0, 0],
        [0, 0, 1]
    ])
    
    mu = np.array([10, 3, 0])
    
    # Generate multivariate normal data
    data = generate_mvnorm(1000, mu, sigma)
    
    # Create DataFrame
    df2 = pd.DataFrame(data, columns=['x1', 'x2', 'e'])
    
    # Display sample statistics
    print("\nCovariance Matrix:")
    print(df2.cov())
    print("\nCorrelation Matrix:")
    print(df2.corr())
    
    # Create scatterplot matrix
    scatterplot_matrix(df2, title="x2 is Constant")
    
    # Generate true y (outcome)
    df2['y'] = 10 - 6*df2['x1'] + 5*df2['x2'] + df2['e']
    
    # Run linear regression with constant variable
    X = df2[['x1', 'x2']]
    y = df2['y']
    run_ols_analysis(X, y, title="Linear Regression with Constant Variable")

# =============================================================================
# PART 3: HETEROSKEDASTICITY
# =============================================================================

def part3_heteroskedasticity():
    """
    Illustrate effects of heteroskedastic errors
    """
    print("\n" + "=" * 70)
    print("PART 3: HETEROSKEDASTICITY")
    print("=" * 70)
    
    # Set seed for reproducibility
    np.random.seed(826)
    
    # Define covariance matrix and mean vector
    sigma = np.array([
        [4, 1],
        [1, 2]
    ])
    
    mu = np.array([10, 3])
    
    # Generate multivariate normal data for x1 and x2
    data = generate_mvnorm(1000, mu, sigma)
    
    # Create DataFrame
    df = pd.DataFrame()
    df['x1'] = data[:, 0]
    df['x2'] = data[:, 1]
    
    # Generate homoskedastic residuals
    df['e1'] = np.random.normal(0, 1, 1000)
    
    # Generate heteroskedastic residuals
    sigma2 = (df['e1']**2) * (df['x1']**2 + df['x2']**2)
    df['e2'] = np.random.normal(0, np.sqrt(sigma2))
    
    # Generate true y (outcome) for both error structures
    df['y1'] = 10 - 6*df['x1'] + 5*df['x2'] + df['e1']
    df['y2'] = 10 - 6*df['x1'] + 5*df['x2'] + df['e2']
    
    # Create scatterplot matrices
    scatterplot_matrix(df[['x1', 'x2', 'e1']], title="Homoskedastic Errors")
    scatterplot_matrix(df[['x1', 'x2', 'e2']], title="Heteroskedastic Errors")
    
    # Plot relationships
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(df['x1'], df['y1'], alpha=0.5)
    plt.title('y1 (Homoskedastic) vs x1')
    plt.xlabel('x1')
    plt.ylabel('y1')
    
    plt.subplot(1, 2, 2)
    plt.scatter(df['x1'], df['y2'], alpha=0.5)
    plt.title('y2 (Heteroskedastic) vs x1')
    plt.xlabel('x1')
    plt.ylabel('y2')
    
    plt.tight_layout()
    plt.show()
    
    # Run linear regressions
    X = df[['x1', 'x2']]
    y1 = df['y1']
    y2 = df['y2']
    
    run_ols_analysis(X, y1, title="Linear Regression with Homoskedastic Errors")
    run_ols_analysis(X, y2, title="Linear Regression with Heteroskedastic Errors")
    
    # Visualize residuals to check for heteroskedasticity
    plt.figure(figsize=(12, 10))
    
    # Model with homoskedastic errors
    model1 = sm.OLS(y1, sm.add_constant(X)).fit()
    
    plt.subplot(2, 2, 1)
    plt.scatter(df['x1'], model1.resid, alpha=0.5)
    plt.title('Residuals vs x1 (Homoskedastic)')
    plt.xlabel('x1')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='r', linestyle='-')
    
    plt.subplot(2, 2, 2)
    plt.scatter(df['x2'], model1.resid, alpha=0.5)
    plt.title('Residuals vs x2 (Homoskedastic)')
    plt.xlabel('x2')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='r', linestyle='-')
    
    # Model with heteroskedastic errors
    model2 = sm.OLS(y2, sm.add_constant(X)).fit()
    
    plt.subplot(2, 2, 3)
    plt.scatter(df['x1'], model2.resid, alpha=0.5)
    plt.title('Residuals vs x1 (Heteroskedastic)')
    plt.xlabel('x1')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='r', linestyle='-')
    
    plt.subplot(2, 2, 4)
    plt.scatter(df['x2'], model2.resid, alpha=0.5)
    plt.title('Residuals vs x2 (Heteroskedastic)')
    plt.xlabel('x2')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='r', linestyle='-')
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# PART 4: SERIAL CORRELATION
# =============================================================================

def part4_serial_correlation():
    """
    Illustrate effects of serially correlated errors
    """
    print("\n" + "=" * 70)
    print("PART 4: SERIAL CORRELATION")
    print("=" * 70)
    
    # Set seed for reproducibility
    np.random.seed(826)
    
    # Define covariance matrix and mean vector
    sigma = np.array([
        [4, 1],
        [1, 2]
    ])
    
    mu = np.array([10, 3])
    
    # Generate multivariate normal data for x1 and x2
    data = generate_mvnorm(1000, mu, sigma)
    
    # Create DataFrame
    df = pd.DataFrame()
    df['x1'] = data[:, 0]
    df['x2'] = data[:, 1]
    
    # Generate independent residuals
    df['e1'] = np.random.normal(0, 1, 1000)
    
    # Generate serially correlated residuals using ARIMA
    ar_params = [0.8]
    ma_params = []
    ar = np.zeros(1000)
    ar[0] = np.random.normal(0, 1)
    
    for t in range(1, 1000):
        ar[t] = ar_params[0] * ar[t-1] + np.random.normal(0, 1)
    
    df['e2'] = ar
    
    # Generate true y (outcome) for both error structures
    df['y1'] = 10 - 6*df['x1'] + 5*df['x2'] + df['e1']
    df['y2'] = 10 - 6*df['x1'] + 5*df['x2'] + df['e2']
    
    # Plot autocorrelation functions
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plot_acf(df['e1'], lags=20, title='ACF of Independent Errors', alpha=0.05)
    
    plt.subplot(1, 2, 2)
    plot_acf(df['e2'], lags=20, title='ACF of Serially Correlated Errors', alpha=0.05)
    
    plt.tight_layout()
    plt.show()
    
    # Run linear regressions
    X = df[['x1', 'x2']]
    y1 = df['y1']
    y2 = df['y2']
    
    run_ols_analysis(X, y1, title="Linear Regression with Independent Errors")
    run_ols_analysis(X, y2, title="Linear Regression with Serially Correlated Errors")
    
    # Visualize residuals over "time" (observation number)
    plt.figure(figsize=(12, 5))
    
    model1 = sm.OLS(y1, sm.add_constant(X)).fit()
    model2 = sm.OLS(y2, sm.add_constant(X)).fit()
    
    plt.subplot(1, 2, 1)
    plt.plot(model1.resid)
    plt.title('Residuals over Time (Independent Errors)')
    plt.xlabel('Observation Number')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='r', linestyle='-')
    
    plt.subplot(1, 2, 2)
    plt.plot(model2.resid)
    plt.title('Residuals over Time (Serially Correlated Errors)')
    plt.xlabel('Observation Number')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='r', linestyle='-')
    
    plt.tight_layout()
    plt.show()
    
    # Show autocorrelation of residuals
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plot_acf(model1.resid, lags=20, title='ACF of Residuals (Independent Errors)', alpha=0.05)
    
    plt.subplot(1, 2, 2)
    plot_acf(model2.resid, lags=20, title='ACF of Residuals (Serially Correlated Errors)', alpha=0.05)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# PART 5: ENDOGENEITY (X CORRELATED WITH ERROR)
# =============================================================================

def part5_endogeneity():
    """
    Illustrate effects of endogeneity (X correlated with error term)
    """
    print("\n" + "=" * 70)
    print("PART 5: ENDOGENEITY (X CORRELATED WITH ERROR)")
    print("=" * 70)
    
    # CASE 5a: X variables directly correlated with error
    print("\nCASE 5a: X VARIABLES DIRECTLY CORRELATED WITH ERROR")
    
    # Set seed for reproducibility
    np.random.seed(826)
    
    # Create a random positive definite covariance matrix
    n = 3
    A = np.random.uniform(-1, 1, (n, n)) * 2 - 1
    sigma = A.T @ A
    
    mu = np.array([10, 3, 0])
    
    # Generate multivariate normal data
    data = generate_mvnorm(1000, mu, sigma)
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=['x1', 'x2', 'e1'])
    
    # Display covariance and correlation matrix
    print("\nCovariance Matrix:")
    print(df.cov())
    print("\nCorrelation Matrix:")
    print(df.corr())
    
    # Create scatterplot matrix
    scatterplot_matrix(df, title="X Variables Correlated with Error")
    
    # Generate true y (outcome)
    df['y1'] = 10 - 6*df['x1'] + 5*df['x2'] + df['e1']
    
    # Run linear regression
    X = df[['x1', 'x2']]
    y = df['y1']
    run_ols_analysis(X, y, title="Linear Regression with Endogenous Variables")
    
    # CASE 5b: Omitted Variable Bias
    print("\n" + "=" * 70)
    print("CASE 5b: OMITTED VARIABLE BIAS")
    
    # Set seed for reproducibility
    np.random.seed(826)
    
    # Define covariance matrix and mean vector
    sigma = np.array([
        [4, 1, 0],
        [1, 2, 0],
        [0, 0, 1]
    ])
    
    mu = np.array([10, 3, 0])
    
    # Generate multivariate normal data
    data = generate_mvnorm(1000, mu, sigma)
    
    # Create DataFrame
    df2 = pd.DataFrame(data, columns=['x1', 'x2', 'e'])
    
    # Generate true y (outcome)
    df2['y'] = 10 - 6*df2['x1'] + 5*df2['x2'] + df2['e']
    
    # Run regression with x1 only (omitting x2)
    X_omit = df2[['x1']]
    y = df2['y']
    run_ols_analysis(X_omit, y, title="Regression with Omitted Variable (x2)")
    
    # Run regression with x2 only (omitting x1)
    X_omit = df2[['x2']]
    run_ols_analysis(X_omit, y, title="Regression with Omitted Variable (x1)")
    
    # Run correct regression with both variables
    X_full = df2[['x1', 'x2']]
    run_ols_analysis(X_full, y, title="Correct Regression with Both Variables")
    
    # CASE 5c: Autocorrelated dependent variable
    print("\n" + "=" * 70)
    print("CASE 5c: AUTOCORRELATED DEPENDENT VARIABLE")
    
    # Set seed for reproducibility
    np.random.seed(826)
    
    # Generate AR(1) process
    n = 1000
    y = np.zeros(n)
    y[0] = np.random.normal(10, 1)  # Initial value
    
    # Generate errors
    eps = np.random.normal(0, 1, n-1)
    
    # Generate AR(1) process: y_t = 10 + 0.4*y_{t-1} + eps_t
    for t in range(1, n):
        y[t] = 10 + 0.4 * y[t-1] + eps[t-1]
    
    # Fit ARIMA model
    model = ARIMA(y, order=(1, 0, 0))
    results = model.fit()
    print(results.summary())
    
    # Plot series
    plt.figure(figsize=(12, 6))
    plt.plot(y)
    plt.title('AR(1) Process: y_t = 10 + 0.4*y_{t-1} + eps_t')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()
    
    # Plot ACF
    plt.figure(figsize=(12, 6))
    plot_acf(y, lags=20, title='Autocorrelation Function of AR(1) Process', alpha=0.05)
    plt.tight_layout()
    plt.show()

# =============================================================================
# MAIN FUNCTION
# =============================================================================

if __name__ == "__main__":
    print("\nPython version of BiasEfficiencyGaussMarkov.R")
    print("=" * 70)
    
    # Set random seed for reproducibility
    np.random.seed(123)
    
    # Run individual parts (uncomment to run specific sections)
    part1_nonlinear_model()
    part2_perfect_correlation()
    part3_heteroskedasticity()
    part4_serial_correlation()
    part5_endogeneity()