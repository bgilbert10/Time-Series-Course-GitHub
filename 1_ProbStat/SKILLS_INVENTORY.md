# Skills Inventory: Probability and Statistics Foundations

This document catalogs all the conceptual modeling, interpretation, and coding skills demonstrated across the R and Python programs in the `1_ProbStat` folder.

## 📊 Probability Theory and Distributions

### Fundamental Distributions
- **Generate random variables from normal distributions** (univariate and multivariate)
- **Create multivariate normal distributions** with specified covariance structures
- **Understand covariance matrix specification** and correlation vs independence
- **Generate non-normal distributions** (t-distribution, chi-squared, uniform)
- **Sample from populations** with and without replacement
- **Control random number generation** using seeds for reproducibility

### Distribution Properties
- **Calculate population moments** (mean, variance, skewness, kurtosis)
- **Understand relationship between sample and population parameters**
- **Interpret distribution shape parameters** (skewness for asymmetry, kurtosis for tail thickness)
- **Generate populations with known characteristics** for simulation studies
- **Create mixture distributions** and compound distributions

## 📈 Statistical Inference Fundamentals

### Hypothesis Testing
- **Formulate null and alternative hypotheses** for means, variances, and distributions
- **Conduct one-sample t-tests** for mean equality
- **Perform two-sided and one-sided tests** with appropriate critical values
- **Interpret p-values and significance levels** in context
- **Apply multiple testing corrections** when needed

### Confidence Intervals
- **Construct confidence intervals for means** using t-distribution
- **Calculate confidence intervals for variances** using chi-squared distribution
- **Interpret confidence interval coverage** and relationship to hypothesis tests
- **Understand relationship between confidence level and interval width**
- **Create bootstrap confidence intervals** for complex statistics

### Normality Testing
- **Apply Jarque-Bera tests** for normality of distributions
- **Interpret Q-Q plots** for visual normality assessment
- **Compare empirical vs theoretical distributions** using visual methods
- **Assess normality through skewness and kurtosis** analysis
- **Test joint hypothesis of normality** using Jarque-Bera framework

## 🎯 Asymptotic Theory

### Law of Large Numbers (LLN)
- **Demonstrate convergence of sample means** to population means
- **Show convergence of sample moments** (variance, skewness, kurtosis)
- **Illustrate LLN for various sample sizes** (10, 100, 1000, 10000)
- **Apply LLN to regression coefficients** and other estimators
- **Demonstrate consistency of estimators** through simulation

### Central Limit Theorem (CLT)
- **Show distribution of sample means approaches normality**
- **Demonstrate CLT for non-normal parent populations**
- **Illustrate effect of sample size on normality** of sampling distribution
- **Apply CLT to standardized test statistics** (z-scores, t-statistics)
- **Show CLT for sample proportions and other statistics**
- **Understand conditions required for CLT** to hold

## 📉 Financial Returns Analysis

### Return Calculations
- **Calculate simple returns** from price data
- **Compute log returns (continuously compounded)**
- **Convert between different return frequencies** (daily, monthly, annual)
- **Handle missing values** in return calculations
- **Understand relationship between simple and log returns**
- **Calculate cumulative returns** and wealth indexes

### Financial Data Retrieval
- **Download stock price data** from Yahoo Finance (R: quantmod, Python: yfinance)
- **Retrieve ETF and index data** (SPY, XLE, market indices)
- **Handle date ranges and data alignment** across multiple securities
- **Process adjusted closing prices** for stock splits and dividends
- **Create time series objects** from financial data

### Return Distribution Analysis
- **Calculate basic statistics for returns** (mean, variance, standard deviation)
- **Compute higher-order moments** (skewness, kurtosis) for return distributions
- **Test for normality of returns** using multiple methods
- **Compare return distributions across assets** and time periods
- **Analyze tail properties** of return distributions (fat tails, outliers)

## 🔬 Econometric Foundations

### Gauss-Markov Theorem
- **Understand conditions for Best Linear Unbiased Estimator (BLUE)**
- **Demonstrate effects when linearity assumption fails** (nonlinear true models)
- **Show impact of multicollinearity** on coefficient estimation
- **Illustrate consequences of heteroskedasticity** (non-constant variance)
- **Demonstrate serial correlation effects** in regression residuals
- **Show endogeneity bias** when regressors correlate with errors

### Regression Analysis Fundamentals
- **Fit OLS regression models** with multiple regressors
- **Interpret regression coefficients** in economic context
- **Understand R-squared and adjusted R-squared** measures
- **Calculate and interpret standard errors** of coefficients
- **Test joint significance** of multiple coefficients
- **Assess model fit** using residual analysis

### Bias and Efficiency Analysis
- **Demonstrate unbiasedness** of OLS under correct assumptions
- **Show bias introduction** when assumptions are violated
- **Compare efficiency** of different estimators (variance comparison)
- **Illustrate bias-variance tradeoff** in estimation
- **Demonstrate consistency** of estimators through simulation
- **Apply robust estimation methods** when assumptions fail

## 🌡️ Simulation and Monte Carlo Methods

### Simulation Design
- **Design Monte Carlo experiments** to demonstrate statistical concepts
- **Control simulation parameters** (sample sizes, number of replications)
- **Generate data with known properties** for validation studies
- **Create simulation loops** with multiple scenarios
- **Store and analyze simulation results** systematically

### Bias and Efficiency Studies
- **Simulate estimator distributions** under various conditions
- **Calculate empirical bias** from simulation results
- **Measure empirical efficiency** through variance estimation
- **Compare theoretical vs empirical properties** of estimators
- **Demonstrate finite sample vs asymptotic properties**

### Robustness Analysis
- **Test estimator performance** under assumption violations
- **Simulate heteroskedastic error structures**
- **Generate serially correlated residuals**
- **Create endogenous regressor scenarios**
- **Assess estimator breakdown points**

## 📊 Advanced Data Visualization

### Distribution Visualization
- **Create histograms with density overlays** (normal, kernel density)
- **Generate Q-Q plots** for distribution comparison
- **Plot empirical vs theoretical CDFs**

### Multivariate Visualization
- **Create scatterplot matrices** for multivariate data
- **Generate correlation heatmaps** with color coding
- **Design 3D scatter plots** for three-variable relationships
- **Create contour plots** for bivariate densities
- **Generate parallel coordinate plots** for high-dimensional data

### Time Series Visualization
- **Plot time series** with appropriate scaling and labels
- **Create return distribution plots** over time
- **Generate rolling statistics plots** (moving averages, volatility)
- **Design faceted plots** for multiple time series
- **Create interactive plots** for data exploration

### Statistical Plots
- **Generate convergence plots** for LLN and CLT demonstrations
- **Create sampling distribution plots** showing theoretical vs empirical
- **Design bias-efficiency comparison plots**
- **Generate simulation result summaries** with confidence bands
- **Create diagnostic plots** for assumption checking

## 🔄 Joint Distribution Analysis

### Bivariate and Multivariate Analysis
- **Calculate sample correlations** between variables
- **Estimate covariance matrices** from data
- **Generate joint density plots** (contour plots, 3D surfaces)
- **Create marginal distribution plots** from joint distributions
- **Test for independence** between variables

## 💻 Programming and Implementation Skills

### R-Specific Skills
- **Use quantmod package** for financial data retrieval
- **Apply fBasics package** for statistical analysis and testing
- **Implement MASS package functions** for multivariate distributions
- **Use car package** for advanced regression diagnostics
- **Create ggplot2 visualizations** for professional graphics
- **Apply plyr and reshape2** for data manipulation
- **Use RColorBrewer** for color palette selection

### Python-Specific Skills
- **Import and use yfinance** for financial data access
- **Apply numpy** for numerical computing and random generation
- **Use pandas** for data manipulation and time series handling
- **Implement scipy.stats** for statistical distributions and tests
- **Apply statsmodels** for econometric analysis
- **Use matplotlib and seaborn** for statistical visualization
- **Implement scikit-learn** for machine learning applications

### Cross-Platform Skills
- **Translate statistical concepts** between R and Python
- **Handle different data structures** (data.frame vs DataFrame)
- **Implement consistent random number generation** across platforms
- **Create equivalent statistical tests** in both languages
- **Generate comparable visualizations** using different plotting libraries

## 🎓 Statistical Theory and Concepts

### Foundational Concepts
- **Understand population vs sample** distinctions
- **Apply sampling theory** to statistical inference
- **Understand estimator properties** (unbiasedness, consistency, efficiency)
- **Apply method of moments estimation**
- **Understand maximum likelihood principles**

### Financial Econometrics Foundations
- **Understand stylized facts** of financial returns
- **Apply volatility modeling concepts**
- **Understand market efficiency implications**

## 📚 Educational and Pedagogical Skills

### Concept Demonstration
- **Design clear examples** of statistical concepts
- **Create progressive difficulty** in problem sets
- **Demonstrate theory through simulation**
- **Connect abstract theory to real applications**
- **Use visualization to enhance understanding**

### Interactive Learning
- **Create hands-on exercises** with real data
- **Design simulation experiments** for concept illustration
- **Develop step-by-step tutorials**
- **Provide economic interpretation** of statistical results
- **Create assessment questions** for understanding verification

### Documentation and Reporting
- **Write clear code comments** explaining statistical concepts
- **Create comprehensive analysis reports**
- **Document simulation experiment designs**
- **Provide interpretation guidelines** for statistical output
- **Generate reproducible research examples**

---

## Summary Statistics

**Total Skills Demonstrated:** 120+ distinct capabilities  
**Programming Languages:** R, Python  
**Key Packages/Libraries:** quantmod, fBasics, MASS, car, ggplot2 (R); numpy, pandas, scipy, statsmodels, matplotlib, seaborn, yfinance (Python)  
**Statistical Concepts:** Probability theory, asymptotic theory, hypothesis testing, simulation methods, financial econometrics  
**Economic Applications:** Financial returns analysis, portfolio theory, risk modeling, market efficiency

---

*This inventory serves as a comprehensive reference for the probability and statistics skills that form the foundation for advanced time series econometric analysis. These concepts are essential prerequisites for understanding the more advanced techniques demonstrated in subsequent course modules.*