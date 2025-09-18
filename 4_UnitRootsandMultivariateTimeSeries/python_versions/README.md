# Python Versions - Unit Roots and Multivariate Time Series

This folder contains Python translations of the R programs in the `4_UnitRootsandMultivariateTimeSeries` directory. These programs demonstrate advanced unit root testing methodologies, seasonal unit root analysis, and multivariate time series concepts using Python's comprehensive econometrics and time series libraries.

## 📋 Contents

| Python File | Original R File | Description |
|-------------|----------------|-------------|
| `UnitRootTestingExamples.py` | `UnitRootTestingExamples.R` | Comprehensive unit root testing methodology |
| `InClassExampleUnitRootsGold.py` | `InClassExampleUnitRootsGold.R` | Interactive classroom activity with gold/macro data |
| `SeasonalityUnitRootsGasInventories.py` | `SeasonalityUnitRootsGasInventories.R` | Seasonal unit root testing with gas inventory data |

## 🛠 Required Python Packages

Install the required packages using pip:

```bash
pip install pandas numpy matplotlib seaborn
pip install statsmodels scipy scikit-learn
pip install yfinance pandas-datareader  # For financial data
```

### Core Libraries Used:

- **pandas**: Data manipulation and time series handling
- **numpy**: Numerical computing and array operations
- **matplotlib/seaborn**: Advanced data visualization and statistical plots
- **statsmodels**: Econometric modeling, unit root tests, and time series analysis
- **scipy**: Statistical tests and scientific computing
- **scikit-learn**: Machine learning tools and metrics
- **yfinance**: Yahoo Finance data download
- **pandas-datareader**: FRED and other economic data sources

## 📊 Data Sources

### FRED Data (No API Key Required)
All programs use **pandas-datareader** to access FRED data without requiring an API key:

- Automatic download of real macroeconomic data from FRED
- No registration or API key setup needed
- Comprehensive error handling for network connectivity issues
- Includes key economic indicators: oil prices, drilling activity, monetary policy variables

### Yahoo Finance Data
Financial market data downloaded directly using **yfinance**:
- Gold futures and spot prices
- Real-time and historical financial data
- Robust error handling for market hours and data availability

### Synthetic Data Generation
When real data is unavailable, programs create realistic synthetic datasets:
- Maintains statistical properties of actual time series
- Educational value preserved through proper data generation
- Clear documentation of synthetic vs. real data usage

## 📚 Program Descriptions

### 1. UnitRootTestingExamples.py
**Purpose**: Comprehensive demonstration of unit root testing methodology and specification issues

**Key Features**:
- **Enhanced ADF Testing**: Custom function with comprehensive output and interpretation
- **Specification Analysis**: Over-specification vs. under-specification impacts
- **Case Comparisons**: Case 1 (no intercept), Case 2 (intercept), Case 4 (intercept + trend)
- **Synthetic Data Generation**: Creates AR(1) processes with and without intercepts
- **Educational Framework**: Step-by-step methodology with clear interpretations

**Advanced Topics**:
- Joint hypothesis testing methodology
- CADFtest equivalent functionality with covariate inclusion
- Bootstrap confidence intervals for test statistics
- Power analysis of different test specifications

**Learning Objectives**:
- Master unit root testing methodology
- Understand specification testing impacts on conclusions
- Learn proper test selection for different data generating processes
- Develop intuition for persistence vs. stationarity

### 2. InClassExampleUnitRootsGold.py  
**Purpose**: Interactive classroom activity using real macroeconomic and gold market data

**Key Features**:
- **Real Data Integration**: Downloads actual FRED macroeconomic data
- **Gold Market Analysis**: Uses Yahoo Finance for gold futures prices
- **Comparative Analysis**: Oil/macro variables vs. gold/monetary policy variables
- **Interactive Elements**: Student exercises and discussion prompts
- **Visual Learning**: Comprehensive plots and diagnostic charts

**Educational Design**:
- Progressive difficulty from basic concepts to advanced applications
- Economic context and market interpretation throughout
- Clear instructor notes and teaching points
- Student-friendly explanations with practical examples

**Data Coverage**:
- WTI oil prices and drilling activity data
- Federal funds rates and monetary policy indicators  
- Gold futures prices and precious metals markets
- Comprehensive data validation and error handling

### 3. SeasonalityUnitRootsGasInventories.py
**Purpose**: Advanced seasonal unit root testing using natural gas storage inventory data

**Key Features**:
- **Fourier Term Analysis**: Creates sine/cosine terms for seasonal modeling
- **Deseasonalization Techniques**: Residualization methods for seasonal adjustment
- **SARIMA vs. Fourier Comparison**: Alternative approaches to seasonal modeling
- **Holdout Sample Validation**: Rigorous forecast evaluation methodology
- **Energy Market Applications**: Practical applications to commodity markets

**Advanced Methodologies**:
- CADFtest equivalent with seasonal covariates
- Model comparison through forecast accuracy metrics
- Seasonal unit root vs. deterministic seasonality distinction
- Advanced diagnostic testing for seasonal time series

**Practical Applications**:
- Energy market forecasting and inventory management
- Seasonal adjustment in commodity price analysis
- Risk management for energy trading and storage
- Policy analysis for strategic petroleum reserves

## 🚀 Getting Started

### Quick Start - Run Any Program:

```python
# Example: Run unit root testing examples
python UnitRootTestingExamples.py

# Example: Run classroom activity
python InClassExampleUnitRootsGold.py

# Example: Run seasonal analysis  
python SeasonalityUnitRootsGasInventories.py
```

### For Real Data Analysis:

**No setup required!** All programs automatically download real data:

1. **FRED Data**: Downloaded using pandas-datareader (no API key needed)
2. **Yahoo Finance**: Financial data downloaded using yfinance
3. **Error Handling**: Clear error messages if data cannot be accessed
4. **Fallback Options**: Synthetic data generation when real data unavailable

**Note**: Internet connection required for data downloads. Programs provide clear guidance when data access fails.

## 📊 Expected Outputs

Each program generates:

- **Comprehensive Testing Results**: Detailed ADF test results with interpretation
- **Advanced Visualizations**: Time series plots, ACF/PACF plots, diagnostic charts
- **Statistical Summaries**: Test statistics, critical values, p-values
- **Economic Interpretations**: Practical implications for policy and markets  
- **Educational Content**: Step-by-step explanations and methodological insights
- **Model Diagnostics**: Residual analysis and specification testing

## 🔧 Customization Options

### Modify Data Periods:
```python
start_date = "2010-01-01"  # Change analysis period
end_date = "2023-12-31"
```

### Adjust Test Parameters:
```python
max_lags = 12         # Change maximum lag length for ADF tests
alpha_level = 0.05    # Adjust significance level
regression_type = 'c' # Change test specification ('c', 'ct', 'ctt', 'nc')
```

### Customize Seasonal Analysis:
```python
fourier_terms = 6     # Number of Fourier term pairs
seasonal_period = 52  # Seasonal frequency (weekly = 52)
```

### Add New Economic Variables:
```python
# Example: Add more FRED series
new_series = fred.get_series('SERIES_ID', start=start_date, end=end_date)

# Example: Add more financial data
new_data = yf.download('TICKER', start=start_date, end=end_date)
```

## 📖 Educational Use

### For Instructors:
- Use `InClassExampleUnitRootsGold.py` for interactive lessons
- Modify synthetic data parameters in `UnitRootTestingExamples.py` to create specific scenarios
- Add breakpoints for student discussions and exercises
- Customize difficulty levels and economic applications
- Use seasonal analysis to demonstrate advanced concepts

### For Students:
- Start with `UnitRootTestingExamples.py` to master basic concepts
- Progress to `InClassExampleUnitRootsGold.py` for real-world applications  
- Use `SeasonalityUnitRootsGasInventories.py` for advanced seasonal methods
- Experiment with different test specifications and data periods
- Practice interpreting test results in economic context

### Research Applications:
- Central bank policy analysis and monetary transmission
- Commodity market efficiency and price discovery
- Financial risk management and volatility modeling
- Energy market regulation and storage policy
- International finance and exchange rate dynamics

## 🐛 Troubleshooting

### Import and Compatibility Issues:

1. **statsmodels Version Compatibility**:
   - **Problem**: Different versions have functions in different locations
   - **Solution**: Programs include robust import handling with fallbacks:
   ```python
   try:
       from statsmodels.tsa.arima.model import ARIMA
   except ImportError:
       try:
           from statsmodels.tsa.arima_model import ARIMA
       except ImportError:
           from statsmodels.tsa.arima_model import ARMA as ARIMA
   ```

2. **Unit Root Test Function Issues**:
   - **Problem**: `adfuller` function parameter changes across versions
   - **Solution**: Enhanced wrapper function handles version differences:
   ```python
   def enhanced_adf_test(series, regression='c', autolag='AIC'):
       # Handles multiple statsmodels versions
       return adfuller(series, regression=regression, autolag=autolag)
   ```

3. **SARIMAX Import Problems**:
   - **Problem**: SARIMAX location varies by statsmodels version
   - **Solution**: Multiple import attempts with graceful degradation:
   ```python
   try:
       from statsmodels.tsa.statespace.sarimax import SARIMAX
   except ImportError:
       SARIMAX = None  # Skip SARIMA analysis
   ```

### Data Access and API Issues:

4. **FRED Data Download Failures**:
   - Verify internet connection is stable
   - FRED servers may be temporarily unavailable
   - Some economic series may be discontinued or revised
   - Programs provide clear error messages and fallback suggestions

5. **Yahoo Finance Timeout Issues**:
   - May timeout during high-volume trading periods
   - Some financial instruments may have limited historical data
   - Programs include retry logic and alternative data sources

6. **Missing Data Handling**:
   - Programs automatically align data across different frequencies
   - Clear indication when synthetic data is used instead of real data
   - Robust handling of missing observations and data gaps

### Performance and Memory Issues:

7. **Large Dataset Memory Usage**:
   ```python
   # Downsample for memory-constrained environments
   data = data.resample('M').last()  # Monthly instead of daily/weekly
   
   # Limit analysis period
   data = data['2010':'2020']
   ```

8. **Computational Intensity Warnings**:
   ```python
   # Suppress convergence warnings for cleaner output
   import warnings
   warnings.filterwarnings('ignore', category=UserWarning)
   
   # Reduce maximum iterations if needed
   max_iter = 100
   ```

### Plotting and Display Issues:

9. **Matplotlib Backend Problems**:
   ```python
   # For headless systems
   import matplotlib
   matplotlib.use('Agg')
   
   # Force inline plots in Jupyter
   %matplotlib inline
   
   # Adjust figure sizes for different displays
   plt.rcParams['figure.figsize'] = [15, 10]
   ```

10. **Seaborn Style Compatibility**:
    ```python
    # Automatic fallback for style issues
    try:
        plt.style.use('seaborn-v0_8')
    except OSError:
        try:
            plt.style.use('seaborn')
        except OSError:
            plt.style.use('default')
    ```

### Statistical and Econometric Issues:

11. **Convergence Problems in ARIMA Models**:
    ```python
    # Use different optimization methods
    model = ARIMA(data, order=(p,d,q)).fit(method='css-mle')
    
    # Adjust tolerance levels
    model = ARIMA(data, order=(p,d,q)).fit(maxiter=1000, method='mle')
    ```

12. **Seasonal Modeling Complexity**:
    - Weekly seasonality (period=52) can be computationally intensive
    - Consider using Fourier terms instead of full SARIMA for long series
    - Monitor convergence warnings in seasonal models

### Debug and Diagnostics Tools:

13. **Enhanced Error Reporting**:
    ```python
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Check function availability
    print(f"ADF test available: {adfuller is not None}")
    print(f"SARIMAX available: {SARIMAX is not None}")
    ```

14. **Data Quality Checks**:
    ```python
    # Verify data integrity before analysis
    print(f"Data shape: {data.shape}")
    print(f"Missing values: {data.isnull().sum()}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    ```

## 📈 Extensions and Advanced Applications

### Possible Extensions:
1. **Multivariate Unit Root Tests** using Johansen cointegration framework
2. **Regime-Switching Unit Root Tests** with structural breaks
3. **Panel Unit Root Tests** for cross-sectional time series
4. **Fractional Integration Models** using long-memory techniques
5. **Bayesian Unit Root Testing** with informative priors

### Research Applications:
- **Monetary Policy Analysis**: Testing for persistence in interest rates and inflation
- **Energy Economics**: Unit roots in oil prices and renewable energy adoption
- **Financial Econometrics**: Testing market efficiency and price discovery
- **Environmental Economics**: Persistence in climate variables and emissions
- **International Finance**: Unit roots in exchange rates and capital flows

### Professional Applications:
- **Risk Management**: Understanding persistence in financial returns
- **Commodity Trading**: Seasonal patterns in agricultural and energy markets  
- **Central Banking**: Monetary policy transmission and effectiveness
- **Investment Strategy**: Long-term vs. short-term investment horizons
- **Regulatory Analysis**: Market structure and efficiency evaluation

## 🤝 Contributing

To contribute improvements:
1. Follow PEP 8 style guidelines for Python code
2. Add comprehensive docstrings to all functions
3. Include robust error handling and fallback mechanisms
4. Test with different data sources and time periods
5. Maintain educational focus with clear explanations
6. Provide economic context and practical interpretations

## 📄 License and Citation

These materials are created for educational purposes in econometrics and time series analysis. When using in academic work, please cite appropriately and acknowledge the original R implementations.

The programs demonstrate state-of-the-art unit root testing methodologies and seasonal analysis techniques commonly used in:
- Academic research in economics and finance
- Central bank policy analysis
- Financial industry risk management
- Regulatory analysis and oversight
- International economic policy evaluation

---

**Note**: These Python translations maintain the rigorous econometric methodology and educational structure of the original R programs while leveraging Python's extensive ecosystem for time series analysis, data visualization, and scientific computing. The implementations follow best practices for reproducible research and provide comprehensive documentation for both instructional and research use.