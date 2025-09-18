# Python Versions - ARIMA and Lag Selection

This folder contains Python translations of the R programs in the `3_ARIMAandLagSelection` directory. These programs demonstrate ARIMA modeling, lag selection techniques, and Granger causality testing using Python's extensive time series analysis libraries.

## 📋 Contents

| Python File | Original R File | Description |
|-------------|----------------|-------------|
| `ARMAGoldRegressionExample.py` | `ARMAGoldRegressionExample.R` | Gold returns regression with macro variables |
| `ARandMAsimulationExamples.py` | `ARandMAsimulationExamples.R` | AR/MA process simulation and root analysis |
| `InClassExampleARMAGold.py` | `InClassExampleARMAGold.R` | Interactive classroom activity |
| `LagLengthGrangerCauseExample.py` | `LagLengthGrangerCauseExample.R` | Lag selection and Granger causality tests |

## 🛠 Required Python Packages

Install the required packages using pip:

```bash
pip install pandas numpy matplotlib seaborn
pip install statsmodels scipy scikit-learn
pip install yfinance pandas-datareader  # For financial data
```

### Core Libraries Used:

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib/seaborn**: Data visualization  
- **statsmodels**: Statistical modeling and time series analysis
- **scipy**: Scientific computing and statistical tests
- **yfinance**: Yahoo Finance data download
- **pandas-datareader**: FRED and other financial data sources (no API key required for FRED)

## 📊 Data Sources

### FRED Data (No API Key Required)
All programs now use **pandas-datareader** to access FRED data without requiring an API key:

- Programs automatically download real economic data from FRED
- No registration or API key setup needed
- Includes comprehensive error handling for network issues

### Yahoo Finance Data
Gold price data is downloaded directly from Yahoo Finance using **yfinance**:
- No API key required
- Real-time and historical data available
- Robust error handling for market hours and data availability

## 📚 Program Descriptions

### 1. ARMAGoldRegressionExample.py
**Purpose**: Analyzes relationships between gold returns and macroeconomic variables

**Key Features**:
- Downloads gold futures data from Yahoo Finance
- Downloads real macroeconomic data from FRED (no API key needed)
- Visual stationarity analysis with ACF plots
- ARIMA model selection for individual series
- Regression analysis with HAC standard errors
- Serial correlation testing and correction

**Learning Objectives**:
- Data acquisition from multiple sources
- Stationarity testing and transformations
- ARIMA model identification
- Regression with time series data
- Residual diagnostics

### 2. ARandMAsimulationExamples.py
**Purpose**: Educational simulation of AR and MA processes

**Key Features**:
- Simulates AR(2) processes with real vs. complex roots
- Characteristic polynomial root analysis
- Cycle length calculation for complex roots
- Model identification from simulated data
- Forecasting behavior comparison
- AR vs. MA vs. White Noise comparison

**Learning Objectives**:
- Understanding AR/MA process behavior
- Root analysis and cyclical patterns
- Model identification techniques
- Forecasting properties of different models

### 3. InClassExampleARMAGold.py
**Purpose**: Interactive classroom activity for teaching ARIMA concepts

**Key Features**:
- Step-by-step guided analysis
- Student exercises and discussion points
- Visual learning with extensive plots
- Simplified explanations for beginners
- Instructor notes and teaching points

**Pedagogical Design**:
- Clear learning outcomes
- Progressive difficulty
- Interactive elements
- Economic context and interpretation

### 4. LagLengthGrangerCauseExample.py
**Purpose**: Demonstrates lag selection and Granger causality testing

**Key Features**:
- Downloads real energy market data from FRED
- Sequential F-tests for lag selection
- HAC standard errors for robust inference
- Granger causality tests between energy markets
- Model diagnostics and validation
- Economic interpretation of results

**Advanced Topics**:
- VAR model framework
- Energy market relationships
- Lead-lag analysis
- Policy implications

## 🚀 Getting Started

### Quick Start - Run Any Program:

```python
# Example: Run the simulation examples
python ARandMAsimulationExamples.py

# Example: Run the classroom activity
python InClassExampleARMAGold.py
```

### For Real Data Analysis:

**No setup required!** All programs now automatically download real data:

1. **FRED Data**: Automatically downloaded using pandas-datareader (no API key needed)
2. **Yahoo Finance**: Gold price data downloaded using yfinance
3. **Error Handling**: Programs will clearly indicate if data cannot be downloaded and suggest solutions

**Note**: An internet connection is required for data downloads. Programs will stop with clear error messages if data cannot be accessed, rather than using synthetic data.

## 📊 Expected Outputs

Each program generates:

- **Multiple visualizations**: Time series plots, ACF/PACF plots, diagnostic plots
- **Statistical results**: Model summaries, test statistics, information criteria
- **Economic interpretations**: Practical implications and policy insights
- **Educational content**: Step-by-step explanations and learning points

## 🔧 Customization Options

### Modify Time Periods:
```python
start_date = "2010-01-01"  # Change analysis period
end_date = "2023-12-31"
```

### Adjust Model Parameters:
```python
max_lags = 8  # Change maximum lag length
confidence_level = 0.95  # Adjust confidence intervals
```

### Add New Variables:
```python
# Example: Add more FRED series
new_series = fred.get_series('SERIES_ID', start=start_date, end=end_date)
```

## 📖 Educational Use

### For Instructors:
- Use `InClassExampleARMAGold.py` for interactive lessons
- Modify synthetic data generation to create specific scenarios
- Add breakpoints for student discussions
- Customize difficulty levels

### For Students:
- Start with `ARandMAsimulationExamples.py` to understand concepts
- Progress to `ARMAGoldRegressionExample.py` for practical applications
- Use classroom activity for guided practice
- Experiment with different parameters and data

## 🐛 Troubleshooting

### Import and Compatibility Issues:

1. **statsmodels Import Errors**:
   - **Problem**: `ImportError: cannot import name 'acorr_ljungbox' from 'statsmodels.tsa.stattools'`
   - **Solution**: The programs now include robust import handling with fallbacks:
   ```python
   try:
       from statsmodels.stats.diagnostic import acorr_ljungbox
   except ImportError:
       try:
           from statsmodels.tsa.stattools import acorr_ljungbox
       except ImportError:
           acorr_ljungbox = None  # Graceful degradation
   ```

2. **ARIMA Model Import Issues**:
   - **Problem**: Different statsmodels versions have ARIMA in different locations
   - **Solution**: Programs try multiple import locations:
   ```python
   try:
       from statsmodels.tsa.arima.model import ARIMA
   except ImportError:
       from statsmodels.tsa.arima_model import ARIMA
   ```

3. **Seaborn Plotting Style Errors**:
   - **Problem**: `OSError: 'seaborn-v0_8' style not found`
   - **Solution**: Automatic fallback to compatible styles:
   ```python
   try:
       plt.style.use('seaborn-v0_8')
   except OSError:
       plt.style.use('seaborn')  # Fallback
   ```

### Package Installation:

4. **Missing Core Packages**:
   ```bash
   # Essential packages
   pip install --upgrade pandas numpy matplotlib statsmodels scipy
   
   # Financial data packages
   pip install yfinance pandas-datareader
   
   # Visualization
   pip install seaborn
   ```

5. **Version Compatibility Matrix**:
   - **Python**: 3.8+ recommended
   - **pandas**: 1.3+ 
   - **numpy**: 1.20+
   - **matplotlib**: 3.3+
   - **statsmodels**: 0.12+ (latest recommended)
   - **scipy**: 1.7+

### API and Data Issues:

6. **FRED Data Download Issues**:
   - Verify internet connection is working
   - FRED servers may occasionally be unavailable
   - Some series may be discontinued - programs will indicate this clearly
   - No API key or rate limits with pandas-datareader

7. **Yahoo Finance Data Download Errors**:
   - May timeout during market hours
   - Some tickers change over time  
   - Programs will stop with clear error messages if downloads fail

8. **Data Download Failures**:
   - Programs will stop execution with clear error messages if data cannot be downloaded
   - No synthetic data fallbacks - requires real data for meaningful analysis
   - Check internet connection and try again if downloads fail
   - Programs automatically align dates across series when data is successfully downloaded

### Runtime and Performance Issues:

9. **Memory Issues with Large Datasets**:
   ```python
   # Downsample data if memory constrained
   data = data.resample('M').last()  # Monthly instead of daily
   
   # Or limit date range
   data = data['2010':'2020']
   ```

10. **Convergence Warnings**:
    ```python
    import warnings
    warnings.filterwarnings('ignore')  # Suppress convergence warnings
    
    # Or target specific warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    ```

11. **Plotting Display Issues**:
    ```python
    # For headless systems or Jupyter issues
    import matplotlib
    matplotlib.use('Agg')
    
    # Force inline plots in Jupyter
    %matplotlib inline
    
    # Adjust figure sizes for better display
    plt.rcParams['figure.figsize'] = [12, 8]
    ```

### Error Recovery and Diagnostics:

12. **Function Availability Checking**:
    - Programs now check if functions are available before using them
    - Graceful degradation when optional features unavailable
    - Clear error messages indicating what's missing

13. **Debug Mode**:
    ```python
    # Enable detailed error reporting
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Check what functions are available
    print(f"ACF available: {acf is not None}")
    print(f"PACF available: {pacf is not None}")
    print(f"ARIMA available: {ARIMA is not None}")
    ```

### Performance Optimization:
```python
# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Optimize pandas operations
pd.set_option('mode.chained_assignment', None)

# For large datasets, consider resampling
data = data.resample('M').last()  # Monthly resampling
```

## 📈 Extensions and Advanced Topics

### Possible Extensions:
1. **Multivariate GARCH models** using `arch` package
2. **Bayesian VAR** using `pymc` or `arviz`
3. **Machine learning forecasting** with `scikit-learn`
4. **High-frequency analysis** with `pandas` datetime tools
5. **Regime switching models** using `statsmodels`

### Research Applications:
- Central bank policy analysis
- Financial risk management
- Commodity price forecasting
- Economic policy evaluation
- Market microstructure analysis

## 🤝 Contributing

To contribute improvements:
1. Follow PEP 8 style guidelines
2. Add comprehensive docstrings
3. Include error handling
4. Test with different data sources
5. Maintain educational focus

## 📄 License and Citation

These materials are created for educational purposes in econometrics and time series analysis. When using in academic work, please cite appropriately and acknowledge the original R implementations.

---

**Note**: These Python translations maintain the pedagogical structure and economic insights of the original R programs while leveraging Python's rich ecosystem for time series analysis and data visualization.