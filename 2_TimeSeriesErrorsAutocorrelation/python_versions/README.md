# Python Translation of Time Series Errors and Autocorrelation Analysis

This directory contains Python translations of the R scripts from the Time Series Errors and Autocorrelation module. The translations maintain the functionality of the original R scripts while using Python's ecosystem for time series analysis.

## Files

**TimeSeriesErrorsAutocorrelation.py**
- Demonstrates handling of autocorrelation in time series data
- Tests for autocorrelation using ACF, PACF, and Ljung-Box tests
- Implements HC and HAC robust standard errors for regression models
- Shows different approaches to modeling autocorrelated data:
  - ARIMA models with external regressors
  - Dynamic regression models with lagged variables
- Uses an oil & gas drilling example with data from FRED

## Key Python Libraries Used

- **pandas**: For data manipulation and analysis
- **numpy**: For numerical operations
- **matplotlib/seaborn**: For data visualization
- **statsmodels**: For statistical models and tests
- **pandas-datareader**: For fetching data from FRED
- **scipy**: For statistical functions

## Installation

You can install all the required packages with:

```bash
pip install numpy pandas matplotlib seaborn statsmodels pandas-datareader scipy yfinance
```

## Usage

Run the Python script directly:

```bash
python TimeSeriesErrorsAutocorrelation.py
```

## Differences from R Version

While the Python translation follows the same analytical approach as the R script, there are some differences:

1. **Data Retrieval**: Uses pandas-datareader instead of quantmod for FRED data
2. **Visualization**: Uses matplotlib and seaborn instead of base R plotting
3. **Time Series Models**: Uses statsmodels instead of forecast, MTS, and dynlm packages
4. **Standard Errors**: Implements HAC using statsmodels' cov_hac instead of sandwich package
5. **Output Format**: Console output is formatted differently
6. **Dynamic Models**: Implements lags using pandas shift() method rather than a specialized package

Despite these implementation differences, the statistical concepts and analysis procedures remain the same.

## Note

The Python version might show some differences in numerical results due to:
1. Different implementations of statistical methods
2. Different data retrieval methods
3. Potential differences in handling of missing values
4. Different optimization algorithms used in model fitting