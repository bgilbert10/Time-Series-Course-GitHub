"""
Sign-restricted VAR models - Python Version

This script demonstrates how to implement sign-restricted VAR models in Python,
which is a translation of the R examples using VARsignR and bsvarSIGNs packages.

Note: There is no direct Python equivalent of the VARsignR package, so we implement
a basic version of the sign restrictions approach using Bayesian methods.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from scipy import stats
import seaborn as sns
from datetime import datetime
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

# Set random seed for reproducibility
np.random.seed(12345)

# Function to fetch FRED data
def get_fred_data(series_ids, start_date='1990-01-01', end_date=None):
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    data = web.DataReader(series_ids, 'fred', start_date, end_date)
    return data

# Read in data from FRED for examples
print("Fetching data from FRED...")
series_ids = [
    'MCOILWTICO',    # WTI oil price
    'IPN213111N',    # Oil & gas drilling index
    'MHHNGSP',       # Henry Hub natural gas price
    'INDPRO',        # Industrial Production
    'PCU483111483111'  # Ocean Freight PPI
]

try:
    fred_data = get_fred_data(series_ids)
    print("Data fetched successfully!")
    
    # Merge data and calculate returns/changes
    oilgas = fred_data.dropna()
    print("\nData overview:")
    print(oilgas.head())
    
    # Calculate log differences for prices, and changes for indices
    doil = np.log(oilgas['MCOILWTICO']).diff().dropna()
    dgas = np.log(oilgas['MHHNGSP']).diff().dropna()
    dwell = oilgas['IPN213111N'].diff().dropna()
    dind = oilgas['INDPRO'].diff().dropna()
    dfre = oilgas['PCU483111483111'].diff().dropna()
    
    # Align series to have the same start and end dates
    common_idx = doil.index.intersection(dgas.index).intersection(dwell.index).intersection(dind.index).intersection(dfre.index)
    doil = doil.loc[common_idx]
    dgas = dgas.loc[common_idx]
    dwell = dwell.loc[common_idx]
    dind = dind.loc[common_idx]
    dfre = dfre.loc[common_idx]
    
    # Create dataset similar to the Kilian-like example
    kil_df = pd.DataFrame({
        'dwell': dwell, 
        'dfre': dfre,
        'dind': dind,
        'doil': doil
    })
    
    print(f"\nData aligned: {len(common_idx)} observations from {common_idx[0]} to {common_idx[-1]}")
except Exception as e:
    print(f"Error fetching data: {e}")
    # Create some simulated data for demonstration if FRED fetch fails
    print("Creating simulated data for demonstration...")
    
n_obs = 300

# Simulate a VAR(2) process with 4 variables
# Create coefficient matrices
A1 = 0.5*np.array([
    [0.6, 0.2, 0.1, 0.0],
    [0.1, 0.5, 0.2, 0.1],
    [0.0, 0.1, 0.7, 0.2],
    [0.2, 0.1, 0.1, 0.6]
])

A2 = np.array([
    [0.2, 0.1, 0.0, 0.0],
    [0.1, 0.2, 0.1, 0.0],
    [0.0, 0.1, 0.1, 0.1],
    [0.1, 0.0, 0.1, 0.2]
])

# Define variance-covariance matrix of errors
sigma = np.array([
    [1.0, 0.3, 0.2, 0.1],
    [0.3, 1.0, 0.3, 0.2],
    [0.2, 0.3, 1.0, 0.3],
    [0.1, 0.2, 0.3, 1.0]
])

# Initialize data
y = np.zeros((n_obs + 2, 4))

# Generate random multivariate normal errors
errors = np.random.multivariate_normal(mean=np.zeros(4), cov=sigma, size=n_obs)

# Generate VAR(2) process
for t in range(2, n_obs + 2):
    y[t] = A1 @ y[t-1] + A2 @ y[t-2] + errors[t-2]

# Discard burn-in observations
y = y[2:, :]

# Create a DataFrame
dates = pd.date_range(start='2000-01-01', periods=n_obs, freq='M')
kil_df = pd.DataFrame(y, index=dates, columns=['dwell', 'dfre', 'dind', 'doil'])

# Plot our dataset
plt.figure(figsize=(14, 10))
kil_df.plot(subplots=True, figsize=(14, 10))
plt.tight_layout()
plt.savefig('signvar_data.png')
plt.close()

# Basic VAR model for our data
print("\nFitting VAR model...")
var_model = VAR(kil_df)
results = var_model.select_order(maxlags=4)
print("Lag order selection:")
print(results.summary())

# Fit VAR(2) model as in the R example
var_fitted = var_model.fit(2)
print("\nVAR model fitting results:")
print(var_fitted.summary())

# Extract components for sign restriction
B = var_fitted.coefs  # VAR coefficients
Sigma = var_fitted.sigma_u  # Residual covariance matrix
k = var_fitted.neqs  # Number of variables
p = var_fitted.k_ar  # Number of lags

print(f"\nModel has {k} variables and {p} lags")

# Function to draw a random orthogonal matrix using QR decomposition
def random_orthogonal(n):
    """Generate a random n x n orthogonal matrix."""
    A = np.random.randn(n, n)
    Q, R = np.linalg.qr(A)
    # Ensure Q is orthogonal: Q'Q = I
    return Q

# Implement a basic version of sign restrictions
def sign_restricted_irf(var_fitted, horizon, restrictions, n_draws=1000):
    """
    Generate impulse responses with sign restrictions.
    
    Parameters:
    -----------
    var_fitted : VAR model fit
        The fitted VAR model
    horizon : int
        Number of periods for impulse response
    restrictions : dict
        Dictionary of sign restrictions {variable_index: {response_index: sign}}
        where sign is 1 for positive, -1 for negative, 0 for unrestricted
    n_draws : int
        Number of random rotation matrices to try
    
    Returns:
    --------
    accepted_irfs : list
        List of accepted impulse response matrices
    """
    
    k = var_fitted.neqs  # Number of variables
    Sigma = var_fitted.sigma_u  # Residual covariance matrix
    
    # Cholesky decomposition of Sigma
    chol = np.linalg.cholesky(Sigma)
    
    # Container for accepted IRFs
    accepted_irfs = []
    
    for _ in range(n_draws):
        # Generate a random orthogonal matrix
        Q = random_orthogonal(k)
        
        # Compute impact matrix
        impact = chol @ Q
        
        # Check if restrictions are satisfied
        satisfies_restrictions = True
        
        for shock_index, responses in restrictions.items():
            for response_index, sign in responses.items():
                if sign != 0:  # Skip unrestricted
                    if sign > 0 and impact[response_index, shock_index] <= 0:
                        satisfies_restrictions = False
                        break
                    elif sign < 0 and impact[response_index, shock_index] >= 0:
                        satisfies_restrictions = False
                        break
            
            if not satisfies_restrictions:
                break
        
        if satisfies_restrictions:
            # Compute IRFs for all horizons
            irf = var_fitted.irf(horizon)
            # Replace the impact matrix with our sign-restricted one
            irf_modified = np.copy(irf.orth_irfs)
            
            # Apply the impact matrix to all horizons
            for h in range(horizon + 1):
                irf_modified[h] = irf.irfs[h] @ impact
            
            accepted_irfs.append(irf_modified)
    
    return accepted_irfs

# Define sign restrictions for our example
# Similar to the Uhlig assumptions in the R script:
# We assume that a shock to oil prices (index 3):
# 1. Has a positive impact on itself (index 3)
# 2. Has a negative impact on industrial production (index 2)
restrictions = {
    3: {  # Shock to oil prices
        3: 1,   # Positive impact on oil prices
        2: -1   # Negative impact on industrial production
    }
}

# Generate sign-restricted IRFs
print("\nGenerating sign-restricted IRFs...")
horizon = 24
accepted_irfs = sign_restricted_irf(var_fitted, horizon, restrictions, n_draws=1000)
print(f"Found {len(accepted_irfs)} IRFs satisfying the restrictions")

# If we found any satisfying IRFs, plot them
if accepted_irfs:
    # Calculate median, 16th, and 84th percentiles across draws
    all_irfs = np.array(accepted_irfs)
    median_irf = np.median(all_irfs, axis=0)
    lower_irf = np.percentile(all_irfs, 16, axis=0)
    upper_irf = np.percentile(all_irfs, 84, axis=0)
    
    # Plot the impulse responses
    variable_names = kil_df.columns
    
    # Plot IRFs for the shock to oil prices
    shock_index = 3  # Oil price shock
    
    fig, axes = plt.subplots(k, 1, figsize=(14, 12), sharex=True)
    x = np.arange(horizon + 1)
    
    for i in range(k):
        ax = axes[i]
        ax.plot(x, median_irf[:, i, shock_index], 'b-', linewidth=2)
        ax.fill_between(x, lower_irf[:, i, shock_index], upper_irf[:, i, shock_index], 
                         color='b', alpha=0.2)
        ax.axhline(y=0, color='r', linestyle='-', linewidth=0.5)
        ax.set_title(f'Response of {variable_names[i]} to {variable_names[shock_index]} shock')
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('sign_restricted_irf.png')
    plt.close()
    
    print("\nPlotted sign-restricted impulse responses")
else:
    print("No IRFs satisfying the restrictions were found. Try more draws or different restrictions.")

print("\nAdditional sign restriction example with a different shock...")

# Define sign restrictions for another example
# Assume a shock to drilling activity (index 0):
# 1. Has a positive impact on itself
# 2. Has a positive impact on freight rates (index 1)
restrictions_alt = {
    0: {  # Shock to drilling
        0: 1,   # Positive impact on drilling
        1: 1    # Positive impact on freight rates
    }
}

# Generate sign-restricted IRFs
accepted_irfs_alt = sign_restricted_irf(var_fitted, horizon, restrictions_alt, n_draws=1000)
print(f"Found {len(accepted_irfs_alt)} IRFs satisfying the alternative restrictions")


# If we found any satisfying IRFs, plot them
if accepted_irfs_alt:
    # Calculate median, 16th, and 84th percentiles across draws
    all_irfs = np.array(accepted_irfs_alt)
    median_irf = np.median(all_irfs, axis=0)
    lower_irf = np.percentile(all_irfs, 16, axis=0)
    upper_irf = np.percentile(all_irfs, 84, axis=0)
    
    # Plot IRFs for the shock to drilling
    shock_index = 0  # Drilling shock
    
    fig, axes = plt.subplots(k, 1, figsize=(14, 12), sharex=True)
    x = np.arange(horizon + 1)
    
    for i in range(k):
        ax = axes[i]
        ax.plot(x, median_irf[:, i, shock_index], 'b-', linewidth=2)
        ax.fill_between(x, lower_irf[:, i, shock_index], upper_irf[:, i, shock_index], 
                         color='b', alpha=0.2)
        ax.axhline(y=0, color='r', linestyle='-', linewidth=0.5)
        ax.set_title(f'Response of {variable_names[i]} to {variable_names[shock_index]} shock')
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('sign_restricted_irf_alt.png')
    plt.close()
    
    print("\nPlotted alternative sign-restricted impulse responses")
else:
    print("No IRFs satisfying the alternative restrictions were found.")




# Define sign restrictions for another example
# Assume a shock to oil price (index 3):
# 1. Has a positive impact on drilling (index 0)
# 2. Has a positive impact on freight rates (index 1)
restrictions_alt = {
    3: {  # Shock to drilling
        0: 1,   # Positive impact on drilling
        1: 1    # Positive impact on freight rates
    }
}

# Generate sign-restricted IRFs
accepted_irfs_alt = sign_restricted_irf(var_fitted, horizon, restrictions_alt, n_draws=1000)
print(f"Found {len(accepted_irfs_alt)} IRFs satisfying the alternative restrictions")

# If we found any satisfying IRFs, plot them
if accepted_irfs_alt:
    # Calculate median, 16th, and 84th percentiles across draws
    all_irfs = np.array(accepted_irfs_alt)
    median_irf = np.median(all_irfs, axis=0)
    lower_irf = np.percentile(all_irfs, 16, axis=0)
    upper_irf = np.percentile(all_irfs, 84, axis=0)
    
    # Plot IRFs for the shock to oil
    shock_index = 3  # Oil shock
    
    fig, axes = plt.subplots(k, 1, figsize=(14, 12), sharex=True)
    x = np.arange(horizon + 1)
    
    for i in range(k):
        ax = axes[i]
        ax.plot(x, median_irf[:, i, shock_index], 'b-', linewidth=2)
        ax.fill_between(x, lower_irf[:, i, shock_index], upper_irf[:, i, shock_index], 
                         color='b', alpha=0.2)
        ax.axhline(y=0, color='r', linestyle='-', linewidth=0.5)
        ax.set_title(f'Response of {variable_names[i]} to {variable_names[shock_index]} shock')
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('sign_restricted_irf_alt.png')
    plt.close()
    
    print("\nPlotted alternative sign-restricted impulse responses")
else:
    print("No IRFs satisfying the alternative restrictions were found.")

print("\nComparing with standard orthogonalized IRFs:")

# Standard orthogonalized IRFs for comparison
irf_standard = var_fitted.irf(horizon)
plt.figure(figsize=(14, 12))
irf_standard.plot(orth=True)
plt.suptitle('Standard Orthogonalized Impulse Responses', fontsize=16)
plt.savefig('standard_irf.png')
plt.close()

print("""
Note: This implementation provides a basic version of sign-restricted VARs.
Full implementations like the ones in R packages VARsignR and bsvarSIGNs would include:
- More efficient sampling algorithms
- Support for narrative sign restrictions
- Forecast error variance decomposition with sign restrictions
- Sign restrictions at multiple horizons
- More sophisticated methods for inference

For more advanced applications, consider implementing a full Bayesian VAR with sign restrictions
or using a more specialized package if available.
""")

print("\nAnalysis complete!")