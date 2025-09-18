# Python Versions of VAR (Vector Autoregression) Analysis Scripts

This directory contains Python translations of the R scripts found in the parent directory. These scripts provide examples and analyses of Vector Autoregression (VAR) models, Structural VAR (SVAR) models, and Impulse Response Functions (IRF).

## Files Included

- `VARExamples.py` - Python version of various VAR examples including oil and gas price analysis
- `ImpulseResponseExample.py` - Detailed exploration of Impulse Response Functions
- `SignVars.py` - Implementation of sign-restricted VAR models
- `VARExamplesFRE.py` - VAR examples including freight rate analysis

## Dependencies

These Python scripts require the following packages:
- pandas - For data manipulation and analysis
- numpy - For numerical operations
- matplotlib - For plotting
- statsmodels - For time series analysis and VAR models
- pandas_datareader - For retrieving financial data from FRED
- scipy - For scientific computing functions
- seaborn - For enhanced visualizations

To install these dependencies:
```
pip install pandas numpy matplotlib statsmodels pandas_datareader scipy seaborn
```

## Key Differences from R Versions

1. **Data Retrieval:**
   - Uses `pandas_datareader` to fetch data from FRED instead of R's `quantmod`
   - Data handling with pandas DataFrames instead of xts/zoo objects

2. **VAR Implementation:**
   - Uses `statsmodels.tsa.vector_ar.var_model` for VAR estimation instead of R's `vars` package
   - Different syntax for selecting lag length, model specification, and diagnostics

3. **Visualization:**
   - Uses matplotlib and seaborn for plotting instead of R's base plotting system
   - Different approach for creating impulse response function plots

4. **Structural Analysis:**
   - Uses statsmodels' SVAR implementation with some adjustments for compatibility
   - Different approach for Cholesky decomposition and orthogonalization

## Usage

Each script can be run independently. The scripts retrieve data from FRED and perform similar analyses to their R counterparts, but using Python's statistical packages and syntax.