#!/usr/bin/env python
# -*- coding: utf-8 -*-

# =============================================================================
# LAW OF LARGE NUMBERS AND CENTRAL LIMIT THEOREM FOR SAMPLE STATISTICS
# =============================================================================

# Import required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import norm, chi2
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_skewness(data):
    """
    Calculate skewness of a data array
    
    Parameters:
    -----------
    data : array-like
        Data array
        
    Returns:
    --------
    float
        Skewness
    """
    return stats.skew(data)

def calculate_kurtosis(data):
    """
    Calculate excess kurtosis of a data array
    
    Parameters:
    -----------
    data : array-like
        Data array
        
    Returns:
    --------
    float
        Excess kurtosis (normal = 0)
    """
    return stats.kurtosis(data, fisher=True)  # Fisher=True for excess kurtosis

def run_regression(X, y):
    """
    Run simple linear regression and return coefficients
    
    Parameters:
    -----------
    X : array-like
        Independent variable
    y : array-like
        Dependent variable
        
    Returns:
    --------
    tuple
        (intercept, slope)
    """
    # Add constant for intercept
    X_sm = sm.add_constant(X)
    
    # Fit regression model
    model = sm.OLS(y, X_sm)
    results = model.fit()
    
    # Extract coefficients
    intercept = results.params[0]
    slope = results.params[1]
    
    return intercept, slope

def plot_histogram_with_line(data, line_value, title, xlabel, bins=50):
    """
    Plot histogram with vertical line at specified value
    
    Parameters:
    -----------
    data : array-like
        Data to plot
    line_value : float
        Value to draw vertical line at
    title : str
        Plot title
    xlabel : str
        X-axis label
    bins : int
        Number of histogram bins
    """
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, alpha=0.7, edgecolor='black')
    plt.axvline(x=line_value, color='red', linestyle='--', linewidth=2)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_density_comparison(empirical_data, theoretical_x, theoretical_y, 
                           title, xlabel, pop_value, sample_mean=None):
    """
    Plot empirical density vs theoretical density
    
    Parameters:
    -----------
    empirical_data : array-like
        Empirical data
    theoretical_x : array-like
        X values for theoretical density
    theoretical_y : array-like
        Y values for theoretical density
    title : str
        Plot title
    xlabel : str
        X-axis label
    pop_value : float
        Population parameter value
    sample_mean : float, optional
        Sample mean value
    """
    plt.figure(figsize=(10, 6))
    
    # Plot empirical density
    sns.kdeplot(empirical_data, label='Bootstrap Density')
    
    # Plot theoretical density
    plt.plot(theoretical_x, theoretical_y, 'r-', linewidth=2, label='Theoretical Density')
    
    # Add vertical lines
    if sample_mean is not None:
        plt.axvline(x=sample_mean, color='black', linestyle='--', linewidth=1, label='Bootstrap Mean Value')
    plt.axvline(x=pop_value, color='red', linestyle='--', linewidth=2, label='Population Value')
    
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def demonstrate_lln_sample_moments(X, pop_mean, pop_sd, pop_skew, pop_kurt):
    """
    Demonstrate Law of Large Numbers for sample moments
    
    Parameters:
    -----------
    X : array-like
        Population data
    pop_mean : float
        Population mean
    pop_sd : float
        Population standard deviation
    pop_skew : float
        Population skewness
    pop_kurt : float
        Population excess kurtosis
    """
    print("\n" + "=" * 70)
    print("DEMONSTRATION OF LAW OF LARGE NUMBERS (LLN)")
    print("=" * 70)
    
    # Sample sizes to demonstrate
    sample_sizes = [10, 100, 1000, 10000]
    
    # Create empty lists to store results
    mean_results = []
    sd_results = []
    skew_results = []
    kurt_results = []
    
    # Calculate moments for each sample size
    for size in sample_sizes:
        sample = np.random.choice(X, size=size, replace=True)
        mean_results.append(np.mean(sample))
        sd_results.append(np.std(sample, ddof=1))
        skew_results.append(calculate_skewness(sample))
        kurt_results.append(calculate_kurtosis(sample))
    
    # Create a results DataFrame
    results = pd.DataFrame({
        'Sample Size': sample_sizes,
        'Mean': mean_results,
        'Std Dev': sd_results,
        'Skewness': skew_results,
        'Kurtosis': kurt_results
    })
    
    # Add population values for comparison
    pop_values = pd.DataFrame({
        'Sample Size': ['Population'],
        'Mean': [pop_mean],
        'Std Dev': [pop_sd],
        'Skewness': [pop_skew],
        'Kurtosis': [pop_kurt]
    })
    
    results = pd.concat([results, pop_values])
    
    print("\nSample Moments as Sample Size Increases:")
    print(results.to_string(index=False))
    
    # Plot convergence
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(sample_sizes, mean_results, 'o-', linewidth=2)
    plt.axhline(y=pop_mean, color='r', linestyle='--', linewidth=2)
    plt.xscale('log')
    plt.title('Mean Convergence', fontsize=14)
    plt.xlabel('Sample Size (log scale)', fontsize=12)
    plt.ylabel('Sample Mean', fontsize=12)
    
    plt.subplot(2, 2, 2)
    plt.plot(sample_sizes, sd_results, 'o-', linewidth=2)
    plt.axhline(y=pop_sd, color='r', linestyle='--', linewidth=2)
    plt.xscale('log')
    plt.title('Standard Deviation Convergence', fontsize=14)
    plt.xlabel('Sample Size (log scale)', fontsize=12)
    plt.ylabel('Sample Std Dev', fontsize=12)
    
    plt.subplot(2, 2, 3)
    plt.plot(sample_sizes, skew_results, 'o-', linewidth=2)
    plt.axhline(y=pop_skew, color='r', linestyle='--', linewidth=2)
    plt.xscale('log')
    plt.title('Skewness Convergence', fontsize=14)
    plt.xlabel('Sample Size (log scale)', fontsize=12)
    plt.ylabel('Sample Skewness', fontsize=12)
    
    plt.subplot(2, 2, 4)
    plt.plot(sample_sizes, kurt_results, 'o-', linewidth=2)
    plt.axhline(y=pop_kurt, color='r', linestyle='--', linewidth=2)
    plt.xscale('log')
    plt.title('Kurtosis Convergence', fontsize=14)
    plt.xlabel('Sample Size (log scale)', fontsize=12)
    plt.ylabel('Sample Kurtosis', fontsize=12)
    
    plt.tight_layout()
    plt.show()

def demonstrate_clt_sample_moments(X, pop_mean, pop_sd, pop_skew, pop_kurt, realz=10000, N=1000):
    """
    Demonstrate Central Limit Theorem for sample moments
    
    Parameters:
    -----------
    X : array-like
        Population data
    pop_mean : float
        Population mean
    pop_sd : float
        Population standard deviation
    pop_skew : float
        Population skewness
    pop_kurt : float
        Population excess kurtosis
    realz : int
        Number of bootstrap realizations
    N : int
        Sample size for each realization
    """
    print("\n" + "=" * 70)
    print(f"DEMONSTRATION OF CENTRAL LIMIT THEOREM (CLT) WITH N={N}")
    print("=" * 70)
    
    # Initialize empty lists for storing results
    mean_list = []
    sd_list = []
    skew_list = []
    kurt_list = []
    
    # Take repeated samples and calculate statistics
    print(f"\nTaking {realz} samples of size {N}...")
    for _ in range(realz):
        sample = np.random.choice(X, size=N, replace=True)
        mean_list.append(np.mean(sample))
        sd_list.append(np.std(sample, ddof=1))
        skew_list.append(calculate_skewness(sample))
        kurt_list.append(calculate_kurtosis(sample))
    
    # Convert lists to arrays
    mean_array = np.array(mean_list)
    sd_array = np.array(sd_list)
    skew_array = np.array(skew_list)
    kurt_array = np.array(kurt_list)
    
    # Calculate theoretical distributions
    # For mean: Normal with mean=pop_mean and sd=pop_sd/sqrt(N)
    x_mean = np.linspace(pop_mean - 4*pop_sd/np.sqrt(N), 
                         pop_mean + 4*pop_sd/np.sqrt(N), 1000)
    y_mean = norm.pdf(x_mean, loc=pop_mean, scale=pop_sd/np.sqrt(N))
    
    # For standard deviation: derived from chi-square distribution
    x_sd = np.linspace(max(0.001, pop_sd - 4*pop_sd/np.sqrt(2*N)), 
                       pop_sd + 4*pop_sd/np.sqrt(2*N), 1000)
    # Approximation for large N - standard deviation of sample sd
    sd_of_sd = pop_sd / np.sqrt(2*N)
    y_sd = norm.pdf(x_sd, loc=pop_sd, scale=sd_of_sd)
    
    # For skewness: Normal with mean=pop_skew and sd=sqrt(6/N)
    x_skew = np.linspace(pop_skew - 4*np.sqrt(6/N), 
                         pop_skew + 4*np.sqrt(6/N), 1000)
    y_skew = norm.pdf(x_skew, loc=pop_skew, scale=np.sqrt(6/N))
    
    # For kurtosis: Normal with mean=pop_kurt and sd=sqrt(24/N)
    x_kurt = np.linspace(pop_kurt - 4*np.sqrt(24/N), 
                         pop_kurt + 4*np.sqrt(24/N), 1000)
    y_kurt = norm.pdf(x_kurt, loc=pop_kurt, scale=np.sqrt(24/N))
    
    # Display summary statistics
    print("\nSummary of Bootstrap Distributions:")
    print(f"Mean: {np.mean(mean_array):.6f} (Theoretical: {pop_mean})")
    print(f"Std Dev: {np.mean(sd_array):.6f} (Theoretical: {pop_sd})")
    print(f"Skewness: {np.mean(skew_array):.6f} (Theoretical: {pop_skew})")
    print(f"Kurtosis: {np.mean(kurt_array):.6f} (Theoretical: {pop_kurt})")
    
    # Plot histograms
    plot_histogram_with_line(
        mean_array, pop_mean, 
        f"CLT of Sample Mean (N={N})", 
        f"Mean of {realz} samples from X of size {N}"
    )
    
    plot_histogram_with_line(
        sd_array, pop_sd, 
        f"CLT of Sample Standard Deviation (N={N})", 
        f"Std Dev of {realz} samples from X of size {N}"
    )
    
    plot_histogram_with_line(
        skew_array, pop_skew, 
        f"CLT of Sample Skewness (N={N})", 
        f"Skewness of {realz} samples from X of size {N}"
    )
    
    plot_histogram_with_line(
        kurt_array, pop_kurt, 
        f"CLT of Sample Kurtosis (N={N})", 
        f"Kurtosis of {realz} samples from X of size {N}"
    )
    
    # Plot density comparisons
    plot_density_comparison(
        mean_array, x_mean, y_mean, 
        "Distribution of the Mean", "Mean", 
        pop_mean, np.mean(mean_array)
    )
    
    plot_density_comparison(
        sd_array, x_sd, y_sd, 
        "Distribution of the Standard Deviation", "Standard Deviation", 
        pop_sd, np.mean(sd_array)
    )
    
    plot_density_comparison(
        skew_array, x_skew, y_skew, 
        "Distribution of the Skewness", "Skewness", 
        pop_skew, np.mean(skew_array)
    )
    
    plot_density_comparison(
        kurt_array, x_kurt, y_kurt, 
        "Distribution of the Kurtosis", "Kurtosis", 
        pop_kurt, np.mean(kurt_array)
    )

def demonstrate_clt_non_normal(realz=10000, N=1000):
    """
    Demonstrate CLT for non-normal distribution (Bernoulli)
    
    Parameters:
    -----------
    realz : int
        Number of bootstrap realizations
    N : int
        Sample size for each realization
    """
    print("\n" + "=" * 70)
    print(f"DEMONSTRATION OF CLT FOR NON-NORMAL DISTRIBUTION WITH N={N}")
    print("=" * 70)
    
    # Create Bernoulli population (coin flips)
    population = np.random.choice([0, 1], size=1000000, replace=True)
    pop_mean = np.mean(population)
    
    # Plot population distribution
    plt.figure(figsize=(10, 6))
    plt.hist(population, bins=3, alpha=0.7, edgecolor='black')
    plt.axvline(x=pop_mean, color='red', linestyle='--', linewidth=2)
    plt.title("Non-Normal Population Distribution (Bernoulli)", fontsize=14)
    plt.xlabel("Value", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Calculate sample means
    mean_list = []
    for _ in range(realz):
        sample = np.random.choice(population, size=N, replace=True)
        mean_list.append(np.mean(sample))
    
    # Convert to array
    mean_array = np.array(mean_list)
    
    # Theoretical distribution for sample mean
    x_mean = np.linspace(pop_mean - 4*np.sqrt(0.25/N), 
                         pop_mean + 4*np.sqrt(0.25/N), 1000)
    y_mean = norm.pdf(x_mean, loc=pop_mean, scale=np.sqrt(0.25/N))
    
    # Plot histogram
    plot_histogram_with_line(
        mean_array, pop_mean, 
        f"CLT of Sample Mean for Bernoulli Distribution (N={N})", 
        f"Mean of {realz} samples from Bernoulli of size {N}"
    )
    
    # Plot density comparison
    plot_density_comparison(
        mean_array, x_mean, y_mean, 
        "Distribution of the Mean (Bernoulli)", "Mean", 
        pop_mean, np.mean(mean_array)
    )

def demonstrate_regression_lln_clt(X, realz=10000, N=1000):
    """
    Demonstrate LLN and CLT for regression coefficients
    
    Parameters:
    -----------
    X : array-like
        Independent variable values
    realz : int
        Number of bootstrap realizations
    N : int
        Sample size for each realization
    """
    print("\n" + "=" * 70)
    print("DEMONSTRATION OF LLN AND CLT FOR REGRESSION COEFFICIENTS")
    print("=" * 70)
    
    # Set true parameter values
    beta0 = 0.5
    beta1 = 0.8
    
    # Generate error term
    eps = np.random.normal(0, 2, size=len(X))
    
    # Generate dependent variable
    Y = beta0 + beta1 * X + eps
    
    # Combine X and Y into a single array for sampling
    Z = np.column_stack((Y, X))
    
    # Run regression on population
    intercept, slope = run_regression(X, Y)
    print("\nRegression on Population:")
    print(f"Intercept: {intercept:.6f} (True: {beta0})")
    print(f"Slope: {slope:.6f} (True: {beta1})")
    
    # Demonstrate LLN for regression coefficients
    print("\nLLN for Regression Coefficients:")
    sample_sizes = [10, 100, 1000, 10000]
    
    for size in sample_sizes:
        # Sample from population
        indices = np.random.choice(len(Z), size=size, replace=True)
        sample_z = Z[indices]
        sample_y = sample_z[:, 0]
        sample_x = sample_z[:, 1]
        
        # Run regression
        intercept, slope = run_regression(sample_x, sample_y)
        print(f"Sample Size {size}: Intercept = {intercept:.6f}, Slope = {slope:.6f}")
    
    # Demonstrate CLT for regression coefficients
    print("\nCLT for Regression Coefficients:")
    b0_list = []
    b1_list = []
    
    for _ in range(realz):
        # Sample from population
        indices = np.random.choice(len(Z), size=N, replace=True)
        sample_z = Z[indices]
        sample_y = sample_z[:, 0]
        sample_x = sample_z[:, 1]
        
        # Run regression
        intercept, slope = run_regression(sample_x, sample_y)
        b0_list.append(intercept)
        b1_list.append(slope)
    
    # Convert to arrays
    b0_array = np.array(b0_list)
    b1_array = np.array(b1_list)
    
    # Plot histograms
    plot_histogram_with_line(
        b0_array, beta0, 
        f"Distribution of Regression Intercepts (N={N})", 
        f"Intercepts from {realz} samples of size {N}"
    )
    
    plot_histogram_with_line(
        b1_array, beta1, 
        f"Distribution of Regression Slopes (N={N})", 
        f"Slopes from {realz} samples of size {N}"
    )

# =============================================================================
# MAIN FUNCTION
# =============================================================================

if __name__ == "__main__":
    print("\nPython version of LLN_CLT_SampleStats.R")
    print("=" * 70)
    
    # Set random seed for reproducibility
    np.random.seed(12345)
    
    # Define population parameters
    pop_mean = 2.5
    pop_sd = 0.5
    pop_skew = 0.0
    pop_kurt = 0.0  # Excess kurtosis (normal = 0)
    
    # Generate population data
    X = np.random.normal(loc=pop_mean, scale=pop_sd, size=1000000)
    
    # Plot population distribution
    plt.figure(figsize=(10, 6))
    plt.hist(X, bins=100, alpha=0.7, edgecolor='black')
    plt.axvline(x=pop_mean, color='red', linestyle='--', linewidth=2)
    plt.title(f"Population Distribution (N={len(X):,})", fontsize=14)
    plt.xlabel("Value", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Run demonstrations
    demonstrate_lln_sample_moments(X, pop_mean, pop_sd, pop_skew, pop_kurt)
    
    # For CLT, can adjust N to 10, 100, 1000, 10000 to see the effect of sample size
    demonstrate_clt_sample_moments(X, pop_mean, pop_sd, pop_skew, pop_kurt, 
                                  realz=10000, N=1000)
    
    # Demonstrate CLT for non-normal distribution
    demonstrate_clt_non_normal(realz=10000, N=1000)
    
    # Demonstrate LLN and CLT for regression coefficients
    demonstrate_regression_lln_clt(X, realz=10000, N=1000)