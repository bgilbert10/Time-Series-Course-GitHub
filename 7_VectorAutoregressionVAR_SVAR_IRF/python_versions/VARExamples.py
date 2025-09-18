"""
Multivariate examples:
VAR examples:
    - oil & gas prices plus drilling index
    - Brent oil vs. Asia LNG prices
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
from datetime import datetime
import seaborn as sns
from matplotlib.ticker import MaxNLocator
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
# WTI oil price
# Oil & gas drilling index
# Henry Hub natural gas price
# Industrial Production
print("Fetching data from FRED...")
series_ids = [
    'MCOILWTICO',    # WTI oil price
    'IPN213111N',    # Oil & gas drilling index
    'MHHNGSP',       # Henry Hub natural gas price
    'INDPRO',        # Industrial Production
    'MCOILBRENTEU',  # Brent oil price
    'PNGASJPUSDM'    # LNG Asia price
]

fred_data = get_fred_data(series_ids)
print("Data fetched successfully!")

# Prepare data for Brent and LNG example
LNGdata = fred_data[['MCOILBRENTEU', 'PNGASJPUSDM']].dropna()
print("\nLNG and Brent data:")
print(LNGdata.head())

# Plot the LNG data
plt.figure(figsize=(14, 7))
LNGdata.plot()
plt.title('Brent Crude Oil and Japan LNG Prices')
plt.ylabel('USD')
plt.savefig('LNGdata_plot.png')
plt.close()

# Calculate logarithmic returns for Brent and LNG
dbrent = np.log(LNGdata['MCOILBRENTEU']).diff().dropna()
dlng = np.log(LNGdata['PNGASJPUSDM']).diff().dropna()

# Merge oil, gas, and drilling data and calculate returns/changes
oilgas = fred_data[['MHHNGSP', 'MCOILWTICO', 'IPN213111N', 'INDPRO']].dropna()
print("\nOil, gas, drilling, and industrial production data:")
print(oilgas.head())

# Plot the data
plt.figure(figsize=(14, 10))
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(14, 10), sharex=True)
for i, col in enumerate(oilgas.columns):
    oilgas[col].plot(ax=axes[i])
    axes[i].set_title(col)
    axes[i].grid(True)
plt.tight_layout()
plt.savefig('oilgas_plot.png')
plt.close()

# Calculate log differences for oil and gas prices, and changes for the drilling index
doil = np.log(oilgas['MCOILWTICO']).diff().dropna()
dgas = np.log(oilgas['MHHNGSP']).diff().dropna()
dwell = oilgas['IPN213111N'].diff().dropna()
dind = oilgas['INDPRO'].diff().dropna()

# Align all series to have the same start and end dates
common_idx = doil.index.intersection(dgas.index).intersection(dwell.index).intersection(dind.index)
doil = doil.loc[common_idx]
dgas = dgas.loc[common_idx]
dwell = dwell.loc[common_idx]
dind = dind.loc[common_idx]
print(f"\nData aligned: {len(common_idx)} observations from {common_idx[0]} to {common_idx[-1]}")

# Create DataFrames for different VAR analyses
og_df = pd.DataFrame({'doil': doil, 'dgas': dgas})
god_df = pd.DataFrame({'dwell': dwell, 'doil': doil, 'dgas': dgas})
kil_df = pd.DataFrame({'dwell': dwell, 'dind': dind, 'doil': doil})

# Align Brent and LNG returns
common_lng_idx = dbrent.index.intersection(dlng.index)
dbrent = dbrent.loc[common_lng_idx]
dlng = dlng.loc[common_lng_idx]
brlng_df = pd.DataFrame({'dbrent': dbrent, 'dlng': dlng})

######## Example 1: VAR for oil & gas price returns ########
print("\n======== Example 1: VAR for oil & gas price returns ========")

# Plot the series
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
og_df['doil'].plot(ax=axes[0], title='Oil Price Returns (log diff)')
og_df['dgas'].plot(ax=axes[1], title='Gas Price Returns (log diff)')
for ax in axes:
    ax.grid(True)
plt.tight_layout()
plt.savefig('oil_gas_returns.png')
plt.close()

# Summary statistics
print("\nSummary statistics:")
print(og_df.describe())

# Select lag length using information criteria
var_model = VAR(og_df)
max_lags = 15
results = var_model.select_order(maxlags=max_lags, trend='n')
print("\nLag order selection:")
print(results.summary())

# Estimate VAR model with lag order from FPE or AIC (in this case, let's use 11 like the R example)
var_fitted = var_model.fit(maxlags=13, ic='fpe', trend='n')
print("\nVAR model fitting results:")
print(var_fitted.summary())

# Plot diagnostics: residuals and autocorrelation
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
residuals = var_fitted.resid

# Plot residuals time series
for i, col in enumerate(og_df.columns):
    axes[0, 0].plot(residuals[col], label=col)
axes[0, 0].set_title('Residuals')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Plot residuals histogram
for i, col in enumerate(og_df.columns):
    sns.histplot(residuals[col], kde=True, ax=axes[0, 1], label=col)
axes[0, 1].set_title('Residuals Distribution')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Plot autocorrelation
for i, col in enumerate(og_df.columns):
    sm.graphics.tsa.plot_acf(residuals[col], lags=20, ax=axes[1, 0], title=f'ACF of {col} Residuals')
    
# Plot partial autocorrelation
for i, col in enumerate(og_df.columns):
    sm.graphics.tsa.plot_pacf(residuals[col], lags=20, ax=axes[1, 1], title=f'PACF of {col} Residuals')

plt.tight_layout()
plt.savefig('var_og_diagnostics.png')
plt.close()

# Smaller model for comparison (VAR(1))
var_fitted1 = var_model.fit(1, trend='n')
print("\nVAR(1) model fitting results:")
print(var_fitted1.summary())

# Check stability (eigenvalues should be < 1 in modulus)
roots = var_fitted.roots
print("\nVAR model stability (eigenvalues):")
print("All eigenvalues inside unit circle:", np.all(np.abs(roots) < 1))
print(roots)

# Granger causality tests
print("\nGranger causality test - Oil -> Gas:")
granger_oil_to_gas = var_fitted.test_causality('dgas', ['doil'])
print(granger_oil_to_gas)

print("\nGranger causality test - Gas -> Oil:")
granger_gas_to_oil = var_fitted.test_causality('doil', ['dgas'])
print(granger_gas_to_oil)

# Forecasting
forecast_steps = 48
forecasts = var_fitted.forecast(og_df.values[-var_fitted.k_ar:], forecast_steps)
forecast_index = pd.date_range(start=og_df.index[-1] + pd.Timedelta(days=30), periods=forecast_steps)
forecast_df = pd.DataFrame(forecasts, index=forecast_index, columns=og_df.columns)

# Plot forecasts with confidence intervals
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
for i, col in enumerate(og_df.columns):
    axes[i].plot(og_df.index[-50:], og_df[col].values[-50:], 'b', label='Observed')
    axes[i].plot(forecast_df.index, forecast_df[col], 'r', label='Forecast')
    axes[i].set_title(f'Forecast for {col}')
    axes[i].legend()
    axes[i].grid(True)
plt.tight_layout()
plt.savefig('og_forecast.png')
plt.close()

# Impulse Response Functions
irf_periods = 24
irf = var_fitted.irf(irf_periods)

# Non-orthogonalized IRF
plt.figure(figsize=(14, 10))
irf.plot()
plt.savefig('og_irf.png')
plt.close()

# Orthogonalized IRF
plt.figure(figsize=(14, 10))
irf.plot(orth=True)
plt.savefig('og_irf_ortho.png')
plt.close()

# Specific IRF: oil -> gas
plt.figure(figsize=(10, 6))
irf.plot(orth=True, impulse='doil', response='dgas')
plt.title('Orthogonalized Impulse Response: Oil Price -> Gas Price')
plt.savefig('og_irf_oil_to_gas.png')
plt.close()

# Forecast Error Variance Decomposition
fevd = var_fitted.fevd(irf_periods)
plt.figure(figsize=(14, 10))
fevd.plot()
plt.savefig('og_fevd.png')
plt.close()

# Compute the covariance matrix of VAR residuals
Omega = var_fitted.sigma_u
print("\nCovariance matrix of VAR residuals:")
print(Omega)

# Cholesky decomposition 
P = np.linalg.cholesky(Omega)
print("\nCholesky factor (P):")
print(P)

# Check: P.T @ P should equal Omega
print("\nVerification: P.T @ P = Omega?")
print(P.T @ P)
print("Difference:")
print(Omega - P.T @ P)

# ADA' form
Dhalf = np.diag(np.diag(P))
A = P.T @ np.linalg.inv(Dhalf)
print("\nA matrix:")
print(A)

# Check: A @ Dhalf² @ A.T should equal Omega
Dhalf_squared = Dhalf @ Dhalf
print("\nVerification: A @ Dhalf² @ A.T = Omega?")
print(A @ Dhalf_squared @ A.T)
print("Difference:")
print(Omega - (A @ Dhalf_squared @ A.T))

# Structural VAR
# In Python, we need to implement this slightly differently
# We specify the matrix Amat where we want to estimate NA values
print("\n======== SVAR Estimation ========")
# Create a matrix of zeros and ones as in the R example (using statsmodels SVAR)
A_matrix = np.array([[1, 0], ['E', 1]])
B_matrix = np.identity(2)  # Identity matrix for B

from statsmodels.tsa.vector_ar.svar_model import SVAR

# Fit SVAR model
svar_model = SVAR(var_fitted.endog, svar_type='A', A=A_matrix, B=B_matrix)
svar_results = svar_model.fit(maxlags=var_fitted.k_ar, solver='lbfgs')
# Workaround for the AttributeError
svar_results.k_exog_user = 0
print("\nSVAR results:")
print(svar_results.summary())

# The estimated A matrix
print("\nEstimated A matrix:")
print(svar_results.A)

# IRF for SVAR model
irf_svar = svar_results.irf(irf_periods)
plt.figure(figsize=(14, 10))
irf_svar.plot()
plt.suptitle('Structural VAR Impulse Responses', fontsize=16)
plt.savefig('og_svar_irf.png')
plt.close()

######## Example 2: Oil prices, gas prices, and drilling ########
print("\n======== Example 2: Oil prices, gas prices, and drilling ========")

# Plot the series
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
god_df['dwell'].plot(ax=axes[0], title='Drilling Index Changes')
god_df['doil'].plot(ax=axes[1], title='Oil Price Returns (log diff)')
god_df['dgas'].plot(ax=axes[2], title='Gas Price Returns (log diff)')
for ax in axes:
    ax.grid(True)
plt.tight_layout()
plt.savefig('god_series.png')
plt.close()

# Summary statistics
print("\nSummary statistics:")
print(god_df.describe())

# Select lag length
var_god_model = VAR(god_df)
max_lags = 13
results_god = var_god_model.select_order(maxlags=max_lags, trend='n')
print("\nLag order selection:")
print(results_god.summary())

# Estimate VAR model 
var_god_fitted = var_god_model.fit(maxlags=13, ic='fpe', trend='n')
print("\nVAR model fitting results:")
print(var_god_fitted.summary())

# Plot diagnostics: residuals and autocorrelation
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
residuals = var_god_fitted.resid

# Plot residuals time series
for i, col in enumerate(god_df.columns):
    axes[0, 0].plot(residuals[col], label=col)
axes[0, 0].set_title('Residuals')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Plot residuals histogram
for i, col in enumerate(god_df.columns):
    sns.histplot(residuals[col], kde=True, ax=axes[0, 1], label=col)
axes[0, 1].set_title('Residuals Distribution')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Plot autocorrelation
for i, col in enumerate(god_df.columns):
    sm.graphics.tsa.plot_acf(residuals[col], lags=20, ax=axes[1, 0], title=f'ACF of {col} Residuals')
    
# Plot partial autocorrelation
for i, col in enumerate(god_df.columns):
    sm.graphics.tsa.plot_pacf(residuals[col], lags=20, ax=axes[1, 1], title=f'PACF of {col} Residuals')

plt.tight_layout()
plt.savefig('var_god_diagnostics.png')
plt.close()

# Check stability
roots_god = var_god_fitted.roots
print("\nVAR model stability (eigenvalues):")
print("All eigenvalues inside unit circle:", np.all(np.abs(roots_god) < 1))
print(roots_god)

# Granger causality
print("\nGranger causality test - Oil -> Others:")
granger_oil_to_others = var_god_fitted.test_causality(['dwell', 'dgas'], ['doil'])
print(granger_oil_to_others)

# Forecasting
forecast_steps = 48
forecasts_god = var_god_fitted.forecast(god_df.values[-var_god_fitted.k_ar:], forecast_steps)
forecast_god_index = pd.date_range(god_df.index[-1] + pd.Timedelta(days=30), periods=forecast_steps)
forecast_god_df = pd.DataFrame(forecasts_god, index=forecast_god_index, columns=god_df.columns)

# Plot forecasts
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
for i, col in enumerate(god_df.columns):
    axes[i].plot(god_df.index[-50:], god_df[col].values[-50:], 'b', label='Observed')
    axes[i].plot(forecast_god_df.index, forecast_god_df[col], 'r', label='Forecast')
    axes[i].set_title(f'Forecast for {col}')
    axes[i].legend()
    axes[i].grid(True)
plt.tight_layout()
plt.savefig('god_forecast.png')
plt.close()

# Impulse Response Functions
irf_god = var_god_fitted.irf(irf_periods)

# Non-orthogonalized IRF
plt.figure(figsize=(14, 12))
irf_god.plot()
plt.savefig('god_irf.png')
plt.close()

# Orthogonalized IRF
plt.figure(figsize=(14, 12))
irf_god.plot(orth=True)
plt.savefig('god_irf_ortho.png')
plt.close()

# Specific IRF: oil -> drilling
plt.figure(figsize=(10, 6))
irf_god.plot(orth=True, impulse='doil', response='dwell')
plt.title('Orthogonalized Impulse Response: Oil Price -> Drilling')
plt.savefig('god_irf_oil_to_drilling.png')
plt.close()

# Cumulative IRF: oil -> drilling
plt.figure(figsize=(10, 6))
irf_god.plot_cum_effects(orth=True, impulse='doil', response='dwell')
plt.title('Cumulative Orthogonalized Impulse Response: Oil Price -> Drilling')
plt.savefig('god_irf_oil_to_drilling_cum.png')
plt.close()

# FEVD
fevd_god = var_god_fitted.fevd(irf_periods)
plt.figure(figsize=(14, 12))
fevd_god.plot()
plt.savefig('god_fevd.png')
plt.close()

# Compute the covariance matrix of VAR residuals
Omega_god = var_god_fitted.sigma_u
print("\nCovariance matrix of VAR residuals:")
print(Omega_god)

# Cholesky decomposition
P_god = np.linalg.cholesky(Omega_god)
print("\nCholesky factor (P):")
print(P_god)

# Check: P.T @ P should equal Omega
print("\nVerification: P.T @ P = Omega?")
print(P_god.T @ P_god)
print("Difference:")
print(Omega_god - P_god.T @ P_god)

# ADA' form
Dhalf_god = np.diag(np.diag(P_god))
A_god = P_god.T @ np.linalg.inv(Dhalf_god)
print("\nA matrix:")
print(A_god)

# Structural VAR - 3 variable case
A_matrix_god = np.array([[1, 0, 0], ['E', 1, 0], ['E', 'E', 1]])
B_matrix_god = np.identity(3)

svar_god_model = SVAR(var_god_fitted.endog, svar_type='A', A=A_matrix_god, B=B_matrix_god)
svar_god_results = svar_god_model.fit(maxlags=var_god_fitted.k_ar)
# Workaround for the AttributeError
svar_god_results.k_exog_user = 0
print("\nSVAR results:")
print(svar_god_results.summary())

print("\nEstimated A matrix:")
print(svar_god_results.A)

# IRF for SVAR model
irf_svar_god = svar_god_results.irf(irf_periods)
plt.figure(figsize=(14, 12))
irf_svar_god.plot()
plt.suptitle('Structural VAR Impulse Responses', fontsize=16)
plt.savefig('god_svar_irf.png')
plt.close()

######## Example 2a: Kilian-like example ########
print("\n======== Example 2a: Kilian-like example ========")

# Plot the series
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
kil_df['dwell'].plot(ax=axes[0], title='Drilling Index Changes')
kil_df['dind'].plot(ax=axes[1], title='Industrial Production Changes')
kil_df['doil'].plot(ax=axes[2], title='Oil Price Returns (log diff)')
for ax in axes:
    ax.grid(True)
plt.tight_layout()
plt.savefig('kil_series.png')
plt.close()

# Summary statistics
print("\nSummary statistics:")
print(kil_df.describe())

# Select lag length
var_kil_model = VAR(kil_df)
results_kil = var_kil_model.select_order(maxlags=13, trend='n')
print("\nLag order selection:")
print(results_kil.summary())

# Estimate VAR model
var_kil_fitted = var_kil_model.fit(maxlags=13, ic='aic', trend='n')
print("\nVAR model fitting results:")
print(var_kil_fitted.summary())

# IRF for Kilian example
irf_kil = var_kil_fitted.irf(irf_periods)
plt.figure(figsize=(14, 12))
irf_kil.plot(orth=True)
plt.savefig('kil_irf_ortho.png')
plt.close()

# Specific IRF: oil -> drilling with cumulative effect
plt.figure(figsize=(10, 6))
irf_kil.plot_cum_effects(orth=True, impulse='doil', response='dwell')
plt.title('Cumulative Orthogonalized Impulse Response: Oil Price -> Drilling')
plt.savefig('kil_irf_oil_to_drilling_cum.png')
plt.close()

######## Example 3: Brent and LNG ########
print("\n======== Example 3: Brent and LNG ========")

# Plot the series
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
brlng_df['dbrent'].plot(ax=axes[0], title='Brent Oil Price Returns (log diff)')
brlng_df['dlng'].plot(ax=axes[1], title='LNG Price Returns (log diff)')
for ax in axes:
    ax.grid(True)
plt.tight_layout()
plt.savefig('brlng_series.png')
plt.close()

# Summary statistics
print("\nSummary statistics:")
print(brlng_df.describe())

# Select lag length
var_br_model = VAR(brlng_df)
results_br = var_br_model.select_order(maxlags=30, trend='n')
print("\nLag order selection:")
print(results_br.summary())

# Estimate VAR model
var_br_fitted = var_br_model.fit(maxlags=30, ic='bic', trend='n')
print("\nVAR model fitting results:")
print(var_br_fitted.summary())

# Granger causality
print("\nGranger causality test - Brent -> LNG:")
granger_brent_to_lng = var_br_fitted.test_causality('dlng', ['dbrent'])
print(granger_brent_to_lng)

print("\nGranger causality test - LNG -> Brent:")
granger_lng_to_brent = var_br_fitted.test_causality('dbrent', ['dlng'])
print(granger_lng_to_brent)

# Impulse Response Functions
irf_br = var_br_fitted.irf(irf_periods)

# Orthogonalized IRF
plt.figure(figsize=(14, 8))
irf_br.plot(orth=True)
plt.savefig('brlng_irf_ortho.png')
plt.close()

# Specific IRF: Brent -> LNG
plt.figure(figsize=(10, 6))
irf_br.plot(orth=True, impulse='dbrent', response='dlng')
plt.title('Orthogonalized Impulse Response: Brent Price -> LNG Price')
plt.savefig('brlng_irf_brent_to_lng.png')
plt.close()

# Structural VAR
A_matrix_br = np.array([[1, 0], ['E', 1]])
B_matrix_br = np.identity(2)

svar_br_model = SVAR(var_br_fitted.endog, svar_type='A', A=A_matrix_br, B=B_matrix_br)
svar_br_results = svar_br_model.fit(maxlags=var_br_fitted.k_ar, solver='lbfgs')
# Workaround for the AttributeError
svar_br_results.k_exog_user = 0
print("\nSVAR results:")
print(svar_br_results.summary())

print("\nEstimated A matrix:")
print(svar_br_results.A)

# IRF for SVAR model
irf_svar_br = svar_br_results.irf(irf_periods)
plt.figure(figsize=(14, 8))
irf_svar_br.plot()
plt.suptitle('Structural VAR Impulse Responses', fontsize=16)
plt.savefig('brlng_svar_irf.png')
plt.close()

print("\nAnalysis complete!")