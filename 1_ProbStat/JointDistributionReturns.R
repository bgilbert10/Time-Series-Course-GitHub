# ======================================================================
# JOINT DISTRIBUTION OF FINANCIAL RETURNS - HEATMAPS AND HISTOGRAMS
# ======================================================================

# --------------------- SETUP AND PACKAGE LOADING ----------------------
# Load required packages for data retrieval, visualization, and analysis
# install.packages("quantmod")  # Uncomment to install if needed
library(quantmod)                # For financial data retrieval
# install.packages("ggplot2")   # Uncomment to install if needed
library(ggplot2)                 # For advanced plotting
# install.packages("plyr")      # Uncomment to install if needed
library(plyr)                    # For data manipulation
# install.packages("epitools")  # Uncomment to install if needed
library(epitools)                # For epidemiological tools
# install.packages("gplots")    # Uncomment to install if needed
library(gplots)                  # For enhanced plotting
# install.packages("reshape2")  # Uncomment to install if needed
library(reshape2)                # For reshaping data
# install.packages("MASS")      # Uncomment to install if needed
library(MASS)                    # For statistical functions
# install.packages("RColorBrewer") # Uncomment to install if needed
library(RColorBrewer)            # For color palettes

# Alternative package loading approach
# if(!require(pacman)) install.packages("pacman")
# pacman::p_load(quantmod, ggplot2, plyr, epitools, gplots, reshape2, MASS, RColorBrewer)

# Store plotting margin defaults for later restoration
par_default <- par()

# --------------------- DATA ACQUISITION -------------------------------
# Fetch market indices and calculate returns

# Download data for Energy sector ETF
getSymbols("XLE", from="2000-01-03")
# XLE is symbol for the SPDR Energy ETF
# Plot price chart
chartSeries(XLE, type=c("line"), theme="white", TA=NULL, 
            main="SPDR Energy ETF (XLE) Prices")
# Calculate returns
xle_daily_log_returns = 100*dailyReturn(XLE, leading=FALSE, type='log')
xle_monthly_log_returns = 100*monthlyReturn(XLE, leading=FALSE, type='log')

# Download data for S&P 500 Index ETF
getSymbols("SPY", from="2000-01-03")
# SPY is the symbol for the S&P 500 Index
# Plot price chart
chartSeries(SPY, type=c("line"), theme="white", TA=NULL,
            main="S&P 500 ETF (SPY) Prices")
# Calculate returns
spy_daily_log_returns = 100*dailyReturn(SPY, leading=FALSE, type='log')
spy_monthly_log_returns = 100*monthlyReturn(SPY, leading=FALSE, type='log')

# Create simple scatterplot of returns
qplot(spy_daily_log_returns, xle_daily_log_returns, 
      main="S&P 500 vs Energy Sector Daily Returns",
      xlab="SPY Returns (%)", ylab="XLE Returns (%)")

# --------------------- DATA PREPARATION -------------------------------
# Convert to time series objects for easier handling

# Transform returns data from "zoo" object to "ts" object 
spy_returns = na.omit(ts(spy_daily_log_returns))
xle_returns = na.omit(ts(xle_daily_log_returns))
# Combine into a single data frame
returns_df = cbind(spy_returns, xle_returns)

# ---------------------- HEAT MAP CREATION ---------------------------
# Create heat map with histograms along the axes

# Create histogram objects for the axes
spy_hist <- hist(spy_returns, breaks=25, plot=FALSE)
xle_hist <- hist(xle_returns, breaks=25, plot=FALSE)

# Calculate maximum count for scaling
top_count = max(spy_hist$counts, xle_hist$counts)

# Function to calculate logarithm of number of points in each bin
bin_log_count <- function(x) log(length(x))

# Create layout for combination heat map/histogram graph
par(mar=c(3, 3, 1, 1))
layout(matrix(c(2, 0, 1, 3), 2, 2, byrow=TRUE), c(3, 1), c(1, 3))

# Create 2D histogram (heat map)
heatmap_data = hist2d(returns_df, nbins=50, FUN=bin_log_count)

# Create histograms along the axes
par(mar=c(0, 2, 1, 0))
barplot(spy_hist$counts, axes=FALSE, ylim=c(0, top_count), space=0, 
        col='red', main="S&P 500 Returns")

par(mar=c(2, 0, 0.5, 1))
barplot(xle_hist$counts, axes=FALSE, xlim=c(0, top_count), space=0, 
        col='red', horiz=TRUE, main="Energy Sector Returns")

# Restore original graphical parameters
par(par_default)

# ---------------------- ALTERNATIVE APPROACH -------------------------
# Create heat map using kernel density estimation

# Convert to data frame and keep column names
df <- data.frame(date = index(returns_df), coredata(returns_df))

# Add binned variables
# Assuming your xts object has columns named "spy_returns" and "xle_returns"
df$spy_bin <- cut(df$spy_returns, breaks=c(-15, -10, -5, 0, 5, 10, 15))
df$xle_bin <- cut(df$xle_returns, breaks=c(-20, -10, 0, 10, 20))

# Add a dummy count column
df$count <- 1

# Define extent of X and Y graph axes
# Reshape with dcast to create a count table
return_table <- dcast(df, spy_bin ~ xle_bin, value.var = "count", fun.aggregate = length)

# Choose color palette from RColorBrewer
color_function <- colorRampPalette(rev(brewer.pal(11, 'Spectral')))
color_palette <- color_function(32)

# Create kernel density estimation
kde_data = kde2d(spy_returns, xle_returns, n=500)

# Plot kernel density image
image(kde_data, col=color_palette, 
      main="Kernel Density Estimation of Joint Returns Distribution",
      xlab="S&P 500 Returns (%)", ylab="Energy Sector Returns (%)")

# Add contour lines
contour(kde_data, add=TRUE, nlevels=10)

# Create combined plot with histograms
# Set up layout 
par(mar=c(3, 3, 1, 1))
layout(matrix(c(2, 0, 1, 3), 2, 2, byrow=TRUE), c(3, 1), c(1, 3))

# Plot kernel density estimation
kde_data_lower = kde2d(spy_returns, xle_returns, n=200)
image(kde_data_lower, col=color_palette,
      main="Joint Distribution of Returns",
      xlab="S&P 500 Returns (%)", ylab="Energy Sector Returns (%)")
contour(kde_data_lower, add=TRUE, nlevels=5)

# Create histograms along the axes
par(mar=c(0, 2, 1, 0))
barplot(spy_hist$counts, axes=FALSE, ylim=c(0, top_count), space=0, 
        col='red', main="S&P 500")

par(mar=c(2, 0, 0.5, 1))
barplot(xle_hist$counts, axes=FALSE, xlim=c(0, top_count), space=0, 
        col='red', horiz=TRUE, main="Energy Sector")

# Restore original graphical parameters
par(par_default)

# ---------------------- CORRELATION ANALYSIS ------------------------
# Calculate correlation and test its significance
correlation_value <- cor(spy_returns, xle_returns, use="complete.obs")
print(paste("Correlation between SPY and XLE returns:", round(correlation_value, 4)))

# Test significance of correlation
correlation_test <- cor.test(spy_returns, xle_returns)
print(correlation_test)