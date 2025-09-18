# Skills Inventory: ARIMA Modeling and Lag Selection

This document catalogs all the conceptual modeling, interpretation, and coding skills demonstrated across the R and Python programs in the `3_ARIMAandLagSelection` folder.

## 📊 Data Acquisition and Preprocessing

### External Data Access
- **Download financial data from Yahoo Finance** (gold futures, stock prices)
- **Retrieve economic data from FRED** (unemployment, inflation, interest rates, drilling activity)
- **Handle missing observations and data alignment** across multiple time series
- **Merge time series with different frequencies** (daily, monthly, quarterly)
- **Select specific date ranges** for analysis windows
- **Convert between data formats** (xts, ts, pandas DataFrame, numpy arrays)

### Data Preparation
- **Apply stationarity transformations** (log differences, simple differences)
- **Create stationary series** for econometric modeling
- **Handle irregular time series data** and missing values
- **Align multiple time series** on common observation dates
- **Calculate returns and growth rates** (continuously compounded vs simple)

## 📈 Visual Analysis and Diagnostics

### Time Series Visualization
- **Plot time series in levels** and identify trends/structural breaks
- **Create multivariate time series plots** for comparative analysis
- **Generate ACF (Autocorrelation Function) plots** with confidence bands
- **Generate PACF (Partial Autocorrelation Function) plots** with confidence bands
- **Interpret ACF/PACF patterns** for model identification (AR vs MA signatures)
- **Plot sequential Box-Ljung test p-values** for increasing lag lengths
- **Create comprehensive diagnostic plot arrays** (tsdiag and tsdisplay style)

### Stationarity Assessment
- **Visually identify non-stationary behavior** (trends, unit roots)
- **Conduct Augmented Dickey-Fuller tests** for unit roots
- **Interpret stationarity test results** and select appropriate transformations
- **Assess seasonal patterns** and structural breaks

## 🔧 ARIMA Modeling Framework

### Model Identification
- **Apply Box-Jenkins methodology** (Identification → Estimation → Diagnostics → Forecasting)
- **Use ACF/PACF patterns** to suggest ARIMA orders
- **Implement automatic model selection** algorithms
- **Compare models using information criteria** (AIC, BIC)
- **Select optimal lag lengths** using sequential testing procedures

### Model Estimation
- **Fit ARIMA(p,d,q) models** with various specifications
- **Estimate AR(p) models** using maximum likelihood
- **Estimate MA(q) models** using maximum likelihood  
- **Fit ARMA(p,q) models** for stationary series
- **Handle model convergence issues** and parameter constraints
- **Implement constrained ARIMA models** with fixed parameters

### Advanced ARIMA Applications
- **Fit ARIMA models with external regressors** (ARIMAX)
- **Model regression equations with ARIMA errors** 
- **Conduct automatic ARIMA order selection** across multiple specifications
- **Compare nested and non-nested ARIMA models**
- **Handle seasonal ARIMA models** and seasonal differencing

## 🧪 Model Diagnostics and Validation

### Residual Analysis
- **Generate standardized residuals** from fitted models
- **Plot residuals over time** to detect patterns
- **Test residual normality** using Q-Q plots and histograms
- **Overlay theoretical normal distributions** on residual histograms
- **Calculate and interpret residual ACF/PACF** for model adequacy

### Statistical Tests
- **Conduct Box-Ljung tests** for residual serial correlation
- **Perform sequential Box-Ljung tests** with increasing lag lengths
- **Implement Ljung-Box tests with degrees-of-freedom corrections**
- **Interpret p-values and significance levels** in time series context
- **Test for remaining autocorrelation** in model residuals

### Comprehensive Diagnostics
- **Create professional diagnostic plot arrays** (2×3 and 3×1 layouts)
- **Generate tsdiag-style diagnostics** (residuals, ACF, Box-Ljung p-values)
- **Produce tsdisplay-style plots** (residuals, ACF, PACF)
- **Assess model adequacy** using multiple diagnostic criteria
- **Identify and correct model specification problems**

## 📊 Regression with Time Series Data

### Basic Regression Analysis
- **Fit OLS regression models** with time series data
- **Interpret regression coefficients** in economic context
- **Assess statistical vs economic significance** of parameters
- **Handle multicollinearity** in time series regressions
- **Create regression summary tables** and interpret R-squared

### Serial Correlation Handling
- **Detect serial correlation** in regression residuals
- **Apply HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors**
- **Compare OLS vs HAC standard errors** and interpret differences
- **Calculate standard error ratios** to assess autocorrelation impact
- **Model regression errors using ARIMA structures** (ARIMA errors approach)

### Advanced Regression Methods
- **Joint estimation of regression and ARIMA error parameters**
- **Automatic selection of ARIMA error structures** in regression
- **Compare ARIMA errors vs HAC approaches** for handling serial correlation
- **Interpret when to use each serial correlation correction method**
- **Create side-by-side method comparison tables**

## 📏 Lag Selection and Model Choice

### Sequential Testing Procedures
- **Implement general-to-specific lag selection** using F-tests
- **Conduct F-tests with HAC standard errors** for robust inference
- **Test joint significance of lag coefficients** across multiple variables
- **Use sequential F-tests** to reduce model complexity
- **Balance model fit vs parsimony** in lag selection

### Information Criteria Methods
- **Compare models using AIC and BIC** across different lag lengths
- **Create lag selection summary tables** with multiple criteria
- **Implement automated lag selection loops** testing multiple specifications
- **Select optimal lag lengths** balancing fit and complexity
- **Document lag selection decision processes**

## 🔄 Granger Causality Analysis

### Causality Testing Framework
- **Understand Granger causality concept** (predictive vs structural causality)
- **Formulate Granger causality hypotheses** (null vs alternative)
- **Test whether X Granger-causes Y** using F-tests
- **Conduct joint significance tests** on lagged variables
- **Use HAC standard errors** in Granger causality tests

### Economic Interpretation
- **Interpret Granger causality results** in economic context
- **Distinguish statistical from economic causality** 
- **Assess lead-lag relationships** between economic variables
- **Apply causality testing** to energy markets, financial data, macro variables
- **Draw policy implications** from causality test results

## 🔮 Forecasting and Prediction

### ARIMA Forecasting
- **Generate point forecasts** from ARIMA models
- **Compute forecast confidence intervals** (80% and 95% levels)
- **Plot forecasts with confidence bands** matching R's forecast package
- **Handle different forecast horizons** (short-term vs long-term)
- **Generate multi-step ahead predictions**

### Forecast Evaluation
- **Implement holdout sample validation** for forecast accuracy
- **Calculate forecast error metrics** (ME, RMSE, MAE, MPE, MAPE)
- **Compare forecast accuracy** across multiple models
- **Split data into training and holdout samples** properly
- **Assess forecast performance** using multiple criteria

### Advanced Forecasting
- **Create forecast comparison tables** across different models
- **Generate forecasts from models with external regressors**
- **Plot forecast vs actual comparisons** with confidence intervals
- **Evaluate forecasting approaches** for different model types
- **Document forecast accuracy assessment procedures**

## 🔄 Characteristic Roots and Cycle Analysis

### Root Analysis
- **Extract AR coefficients** from fitted models
- **Construct characteristic polynomials** from AR parameters
- **Find polynomial roots** using numerical methods
- **Calculate root moduli** to assess stationarity conditions
- **Identify complex conjugate pairs** in characteristic roots

### Business Cycle Analysis
- **Compute cycle lengths** from complex roots
- **Calculate periods of oscillation** in economic time series
- **Interpret cyclical components** in ARIMA models
- **Assess stability conditions** using root moduli
- **Connect root structure to forecast behavior** (smooth vs cyclical)

## 💻 Programming and Implementation Skills

### R-Specific Skills
- **Use quantmod package** for financial data retrieval
- **Implement forecast package** functions for ARIMA modeling
- **Apply lmtest and sandwich packages** for robust inference
- **Use vars package** for VAR estimation and testing
- **Create professional plots** using base R graphics
- **Handle xts and ts time series objects**
- **Use linearHypothesis()** for joint coefficient tests

### Python-Specific Skills
- **Import and manage statsmodels** for econometric modeling
- **Use pandas-datareader** for economic data access
- **Handle ARIMA model estimation** in statsmodels
- **Create matplotlib visualizations** with professional styling
- **Manage pandas DataFrame and Series objects** for time series
- **Implement robust error handling** for model estimation failures
- **Use numpy for numerical computations** and array operations

### Cross-Platform Skills
- **Translate econometric methods** between R and Python
- **Handle package compatibility issues** across different versions
- **Implement consistent diagnostic approaches** across languages
- **Create reproducible analysis workflows**
- **Document code for educational purposes**

## 🎓 Econometric Theory and Concepts

### Time Series Fundamentals
- **Understand stationarity concepts** (weak vs strong stationarity)
- **Apply Wold decomposition theorem** to ARMA representations
- **Interpret MA and AR polynomial forms** and their invertibility/stationarity conditions
- **Connect theoretical ARMA properties** to empirical estimation
- **Understand Box-Jenkins methodology** as systematic modeling approach

### Statistical Inference
- **Apply maximum likelihood estimation** to time series models
- **Understand information criteria** (AIC, BIC) for model selection
- **Implement robust standard errors** (HAC) for time series regression
- **Conduct hypothesis testing** in time series context
- **Interpret confidence intervals** for parameters and forecasts

### Economic Applications
- **Model financial returns** using appropriate transformations
- **Analyze commodity price relationships** and market interactions
- **Study macroeconomic variable relationships** using VAR methods
- **Apply time series methods** to energy market analysis
- **Interpret economic causality** vs statistical associations

## 📚 Educational and Presentation Skills

### Documentation and Reporting
- **Create comprehensive analysis reports** with clear interpretations
- **Document modeling decisions** and their economic rationale
- **Present results in accessible formats** for different audiences
- **Compare alternative methodological approaches**
- **Provide decision frameworks** for method selection

### Interactive Teaching
- **Design in-class exercises** with progressive difficulty
- **Create student discussion points** and interactive questions
- **Develop step-by-step tutorials** for complex procedures
- **Provide economic context** for statistical methods
- **Connect theory to real-world applications**

---

## Summary Statistics

**Total Skills Demonstrated:** 150+ distinct capabilities  
**Programming Languages:** R, Python  
**Key Packages/Libraries:** quantmod, forecast, vars, lmtest, sandwich (R); statsmodels, pandas, matplotlib, numpy (Python)  
**Economic Applications:** Financial markets, energy markets, macroeconomics, commodity pricing  
**Statistical Methods:** ARIMA modeling, Granger causality, robust inference, forecasting, diagnostic testing

---

*This inventory serves as a comprehensive reference for the time series econometric skills developed across all programs in this folder. Each skill is demonstrated with real economic data and includes both conceptual understanding and practical implementation.*