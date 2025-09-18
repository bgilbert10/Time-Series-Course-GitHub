"""
Delving into Impulse Response Functions - Python Version

This script demonstrates how to calculate and interpret impulse response functions
for vector autoregression (VAR) models in Python, translating the R examples.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.svar_model import SVAR
from statsmodels.stats.diagnostic import acorr_ljungbox
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

# Function to fetch FRED data
def get_fred_data(series_ids, start_date='1990-01-01', end_date=None):
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    data = web.DataReader(series_ids, 'fred', start_date, end_date)
    return data

# Read in data from FRED
print("Fetching data from FRED...")
series_ids = [
    'MCOILWTICO',    # WTI oil price
    'IPN213111N',    # Oil & gas drilling index
    'MHHNGSP'        # Henry Hub natural gas price
]

fred_data = get_fred_data(series_ids)
print("Data fetched successfully!")

# Merge oil, gas, and drilling data
oilgas = fred_data.dropna()
print("\nOil, gas, and drilling data:")
print(oilgas.head())

# Plot the data
plt.figure(figsize=(14, 10))
oilgas.plot(subplots=True, figsize=(14, 10))
plt.tight_layout()
plt.savefig('irf_data_plot.png')
plt.close()

# Calculate log differences for oil and gas prices, and changes for the drilling index
doil = np.log(oilgas['MCOILWTICO']).diff().dropna()
dgas = np.log(oilgas['MHHNGSP']).diff().dropna()
dwell = oilgas['IPN213111N'].diff().dropna()

# Align all series to have the same start and end dates
common_idx = doil.index.intersection(dgas.index).intersection(dwell.index)
doil = doil.loc[common_idx]
dgas = dgas.loc[common_idx]
dwell = dwell.loc[common_idx]
print(f"\nData aligned: {len(common_idx)} observations from {common_idx[0]} to {common_idx[-1]}")

# Create DataFrame for 2-equation oil & gas price return VAR
og_df = pd.DataFrame({'doil': doil, 'dgas': dgas})
ow_df = pd.DataFrame({'doil': doil, 'dwell': dwell})

# Plot the multivariate time series for oil and gas
plt.figure(figsize=(14, 8))
og_df.plot(subplots=True, figsize=(14, 8))
plt.tight_layout()
plt.savefig('og_returns.png')
plt.close()

# Estimate the VAR(2) model
var_og = VAR(og_df)
var_og_fitted = var_og.fit(2)
print("\nVAR(2) model fitting results:")
print(var_og_fitted.summary())

# Compute and plot non-orthogonalized impulse response function
irf_og = var_og_fitted.irf(6)
plt.figure(figsize=(14, 10))
irf_og.plot()
plt.suptitle('Non-Orthogonalized Impulse Response Function', fontsize=16)
plt.savefig('og_irf_nonortho.png')
plt.close()

# Extract VAR coefficients to manually calculate the IRF
A1 = var_og_fitted.coefs[0]  # First lag coefficients
A2 = var_og_fitted.coefs[1]  # Second lag coefficients

# Create companion form matrix F
# This is the equivalent of the F matrix in the R code
F = np.zeros((4, 4))
F[0:2, 0:2] = A1
F[0:2, 2:4] = A2
F[2:4, 0:2] = np.eye(2)  # Add identity matrix in lower left
print("\nCompanion form matrix F:")
print(F)

# Calculate 2-period-ahead IRF (non-orthogonalized)
F2 = np.matmul(F, F)
irfF = F2[0:2, 0:2]
print("\n2-period-ahead IRF (manually calculated):")
print(irfF)
print("\n2-period-ahead IRF (from statsmodels):")
print(irf_og.irfs[2])

# 3-period ahead
F3 = np.matmul(F, F2)
irfF3 = F3[0:2, 0:2]
print("\n3-period-ahead IRF (manually calculated):")
print(irfF3)
print("\n3-period-ahead IRF (from statsmodels):")
print(irf_og.irfs[3])

# Compute and plot orthogonalized impulse response function
irf_og_orth = var_og_fitted.irf(6)
plt.figure(figsize=(14, 10))
irf_og_orth.plot(orth=True)
plt.suptitle('Orthogonalized Impulse Response Function', fontsize=16)
plt.savefig('og_irf_ortho.png')
plt.close()

# Calculate the covariance matrix of VAR residuals
Omega = var_og_fitted.sigma_u
print("\nCovariance matrix of VAR residuals:")
print(Omega)

# Cholesky decomposition (lower triangular)
P = np.linalg.cholesky(Omega)
print("\nCholesky factor (P):")
print(P)

# Check that P.T @ P equals Omega
print("\nVerification: P.T @ P = Omega?")
print(P.T @ P)
print("Difference:")
print(Omega - (P.T @ P))

# Calculate ADA' form
Dhalf = np.diag(np.diag(P))
A = P.T @ np.linalg.inv(Dhalf)
print("\nA matrix:")
print(A)

# Check that A @ Dhalf² @ A.T equals Omega
Dhalf_squared = Dhalf @ Dhalf
print("\nVerification: A @ Dhalf² @ A.T = Omega?")
print(A @ Dhalf_squared @ A.T)
print("Difference:")
print(Omega - (A @ Dhalf_squared @ A.T))

# Effect of one standard deviation increase (orthogonalized IRF)
irfForth = irfF @ P.T
print("\nOrthogonalized IRF (manual calculation for 2-period-ahead):")
print(irfForth)
print("\nOrthogonalized IRF (from statsmodels for 2-period-ahead):")
print(irf_og_orth.orth_irfs[2])

# Effect of one unit increase (orthogonalized IRF with A)
irfForthA = irfF @ A
print("\nOrthogonalized IRF using A matrix (manual calculation for 2-period-ahead):")
print(irfForthA)

# Structural VAR
# Create a matrix of zeros and ones as in the R example
A_matrix = np.array([[1, 0], ['E', 1]])
B_matrix = np.identity(2)  # Identity matrix for B

# Fit SVAR model
svar_og_model = SVAR(var_og_fitted.endog, svar_type='A', A=A_matrix, B=B_matrix)
svar_og_results = svar_og_model.fit(maxlags=var_og_fitted.k_ar, solver='lbfgs')
# Workaround for the AttributeError
svar_og_results.k_exog_user = 0
print("\nSVAR results:")
print(svar_og_results.summary())

# The estimated A matrix
print("\nEstimated A matrix:")
print(svar_og_results.A)

# Transform to the A matrix from orthogonalized IRFs 
A_struct = np.linalg.inv(svar_og_results.A)
print("\nA_struct (inverse of estimated A):")
print(A_struct)
print("\nComparison with A from orthogonalization:")
print(A)

# IRF for SVAR model
irf_svar_og = svar_og_results.irf(6)
plt.figure(figsize=(14, 10))
irf_svar_og.plot()
plt.suptitle('Structural VAR Impulse Responses', fontsize=16)
plt.savefig('og_svar_irf.png')
plt.close()

# Simple regression of gas returns on oil returns for comparison
X = sm.add_constant(og_df['doil'])
model = sm.OLS(og_df['dgas'], X)
results = model.fit()
print("\nRegression of gas returns on oil returns:")
print(results.summary())

# Repeat analysis for oil returns and drilling
print("\n======== Oil returns and drilling changes ========")

# Plot the data
plt.figure(figsize=(14, 8))
ow_df.plot(subplots=True, figsize=(14, 8))
plt.tight_layout()
plt.savefig('ow_data.png')
plt.close()

# Estimate the VAR(2) model
var_ow = VAR(ow_df)
var_ow_fitted = var_ow.fit(2)
print("\nVAR(2) model fitting results:")
print(var_ow_fitted.summary())

# Compute IRF
irf_ow = var_ow_fitted.irf(12)
plt.figure(figsize=(14, 10))
irf_ow.plot()
plt.suptitle('Non-Orthogonalized Impulse Response Function', fontsize=16)
plt.savefig('ow_irf_nonortho.png')
plt.close()

# Extract VAR coefficients for manual IRF calculation
A1_ow = var_ow_fitted.coefs[0]  # First lag coefficients
A2_ow = var_ow_fitted.coefs[1]  # Second lag coefficients

# Create companion form matrix F
F_ow = np.zeros((4, 4))
F_ow[0:2, 0:2] = A1_ow
F_ow[0:2, 2:4] = A2_ow
F_ow[2:4, 0:2] = np.eye(2)

# Calculate 2-period-ahead IRF (non-orthogonalized)
F2_ow = np.matmul(F_ow, F_ow)
irfF_ow = F2_ow[0:2, 0:2]
print("\n2-period-ahead IRF (manually calculated):")
print(irfF_ow)
print("\n2-period-ahead IRF (from statsmodels):")
print(irf_ow.irfs[2])

# Orthogonalized IRF
irf_ow_orth = var_ow_fitted.irf(2)
plt.figure(figsize=(14, 10))
irf_ow_orth.plot(orth=True)
plt.suptitle('Orthogonalized Impulse Response Function', fontsize=16)
plt.savefig('ow_irf_ortho.png')
plt.close()

# Calculate the covariance matrix of VAR residuals
Omega_ow = var_ow_fitted.sigma_u
print("\nCovariance matrix of VAR residuals:")
print(Omega_ow)

# Cholesky decomposition
P_ow = np.linalg.cholesky(Omega_ow)
print("\nCholesky factor (P):")
print(P_ow)

# Calculate ADA' form
Dhalf_ow = np.diag(np.diag(P_ow))
A_ow = P_ow.T @ np.linalg.inv(Dhalf_ow)
print("\nA matrix:")
print(A_ow)

# Effect of one standard deviation increase
irfForth_ow = irfF_ow @ P_ow.T
print("\nOrthogonalized IRF (manual calculation for 2-period-ahead):")
print(irfForth_ow)
print("\nOrthogonalized IRF (from statsmodels for 2-period-ahead):")
print(irf_ow_orth.orth_irfs[2])

# Effect of one unit increase
irfForthA_ow = irfF_ow @ A_ow
print("\nOrthogonalized IRF using A matrix (manual calculation for 2-period-ahead):")
print(irfForthA_ow)

# Structural VAR for oil and drilling
A_matrix_ow = np.array([[1, 0], ['E', 1]])
B_matrix_ow = np.identity(2)

svar_ow_model = SVAR(var_ow_fitted.endog, svar_type='A', A=A_matrix_ow, B=B_matrix_ow)
svar_ow_results = svar_ow_model.fit(maxlags=var_ow_fitted.k_ar,solver='lbfgs')
# Workaround for the AttributeError
svar_ow_results.k_exog_user = 0
print("\nSVAR results:")
print(svar_ow_results.summary())

print("\nEstimated A matrix:")
print(svar_ow_results.A)

# Transform to the A matrix from orthogonalized IRFs
A_struct_ow = np.linalg.inv(svar_ow_results.A)
print("\nA_struct (inverse of estimated A):")
print(A_struct_ow)

# IRF for SVAR model
irf_svar_ow = svar_ow_results.irf(6)
plt.figure(figsize=(14, 10))
irf_svar_ow.plot()
plt.suptitle('Structural VAR Impulse Responses', fontsize=16)
plt.savefig('ow_svar_irf.png')
plt.close()

# Simple regression of drilling changes on oil returns for comparison
X_ow = sm.add_constant(ow_df['doil'])
model_ow = sm.OLS(ow_df['dwell'], X_ow)
results_ow = model_ow.fit()
print("\nRegression of drilling changes on oil returns:")
print(results_ow.summary())

print("\nAnalysis complete!")