# Python Versions of Time Series Course R Scripts

This directory contains Python translations of the R scripts from the Time Series Course. These translations aim to replicate the functionality of the original R scripts while following Python best practices.

## Files

1. **ReturnsDistributions.py**
   - Analyzes return distributions of financial instruments
   - Calculates summary statistics and tests for normality
   - Creates plots to visualize return distributions
   - Uses yfinance for data retrieval, pandas for data manipulation, and matplotlib/seaborn for visualization

2. **BiasEfficiencyGaussMarkov.py**
   - Illustrates bias and efficiency when Gauss-Markov assumptions fail
   - Demonstrates issues with non-linear models, multicollinearity, heteroskedasticity, serial correlation, and endogeneity
   - Uses numpy for data generation, statsmodels for regression analysis, and matplotlib/seaborn for visualization

3. **JointDistributionReturns.py**
   - Creates heat maps and histograms for joint return distributions
   - Demonstrates different approaches to visualizing return distributions
   - Uses yfinance for data retrieval, pandas for data manipulation, scipy for kernel density estimation, and matplotlib for visualization

4. **LLN_CLT_SampleStats.py**
   - Demonstrates the Law of Large Numbers (LLN) and Central Limit Theorem (CLT)
   - Shows convergence of sample moments (mean, variance, skewness, kurtosis)
   - Illustrates CLT for both normal and non-normal distributions
   - Applies LLN and CLT to regression coefficients
   - Uses numpy for data generation, scipy for statistical functions, and matplotlib/seaborn for visualization

## Usage

Each script can be run directly from the command line:

```bash
python ReturnsDistributions.py
python BiasEfficiencyGaussMarkov.py
python JointDistributionReturns.py
python LLN_CLT_SampleStats.py
```

## Dependencies

The Python scripts require the following packages:

- numpy
- pandas
- matplotlib
- seaborn
- scipy
- statsmodels
- scikit-learn
- yfinance

You can install all dependencies using:

```bash
pip install numpy pandas matplotlib seaborn scipy statsmodels scikit-learn yfinance
```

## Differences from R Versions

While these Python scripts aim to replicate the R functionality, there are some differences:

1. Data retrieval: The Python versions use the `yfinance` package instead of `quantmod`
2. Plotting: Uses `matplotlib` and `seaborn` instead of R's base plotting and `ggplot2`
3. Statistical functions: Uses functions from `scipy.stats` instead of R's `fBasics`
4. Output format: Console output is formatted differently from R
5. Computational methods: Some statistical calculations may use slightly different algorithms

Despite these differences, the concepts and analyses demonstrated are the same.

## Notes

- The random seed is set at the beginning of each script for reproducibility
- Some data may differ slightly from the R versions due to different data sources
- The Python implementations include additional visualizations and explanations in some cases