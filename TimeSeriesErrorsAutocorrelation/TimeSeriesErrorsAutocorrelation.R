# Preamble -------------------------------------

# Testing for autocorrelation and plotting ACFs
# Demonstrate HC and HAC standard errors in a regression
# Test for serial correlation in the residuals

require(quantmod)
require(MTS)
require(forecast)
require(sandwich)
require(lmtest)
require(car)
require(dynlm)

# Read in Data from FRED -------------------

# Model for determining price response of oil & gas drilling
# Read wti oil prices, henry hub natural gas prices, and drilling index from FRED
getSymbols("MCOILWTICO",src="FRED")
# Oil & gas drilling index
getSymbols("IPN213111N",src="FRED")
# Henry Hub
getSymbols("MHHNGSP",src="FRED")

# Merge and Plot data ------------------------------------

# verify data are stationary 
# Use eyeballs for now, more rigorous tests later

# Merge and plot prices. Appear nonstationary (we will test this later)
godlevels = merge.xts(MHHNGSP,MCOILWTICO,IPN213111N,all=TRUE)
plot(godlevels)
MTSplot(godlevels)

# Calculate log percentage change - appears stationary
goddiffs = na.omit(diff(log(godlevels)))
plot(goddiffs)
MTSplot(goddiffs)

# Describe autocorrelation in each series -----------------------------------

# Might want to investigate seasonal autocorrelation (later) depending on context

# Little autocorrelation in natural gas price returns, some in oil price returns,
# and more in drilling growth.
acf(goddiffs$MCOILWTICO)
Box.test(goddiffs$MCOILWTICO,lag=10,type='Ljung')
# how many lags? m approximately log(T)
m = log(length(goddiffs))
m
Box.test(goddiffs$MCOILWTICO,lag=7,type='Ljung')

acf(goddiffs$MHHNGSP)
Box.test(goddiffs$MHHNGSP,lag=7,type='Ljung')
acf(goddiffs$IPN213111N)
Box.test(goddiffs$IPN213111N,lag=7,type='Ljung')

# Estimate and examine a model -----------------------------------

# How do prices affect drilling activity
mod <- lm(IPN213111N ~ MCOILWTICO + MHHNGSP,data=goddiffs)
summary(mod)
# very little contemporaneous relationship

# Investigate residuals -----------------------------------


# significant autocorrelation in the residuals at one lag or more
ts.plot(residuals(mod))
acf(residuals(mod))
tsdisplay(residuals(mod))
bt = Box.test(mod$residuals,lag=7) # uses 7 degrees of freedom, because log(T) = 6.8
                              # but we estimated 2 params: 2 coefs.
                              # should use 7 - 2 = 5 degrees of freedom
pv = 1-pchisq(bt$statistic,5) # p-value is still zero.
pv

# HAC standard errors ------------------------------------

# Use HAC standard errors on original model (and show HC for practice)

# notice the HAC-corrected standard errors are quite a bit bigger, 
# except for natural gas (MHHNGSP - monthly Henry Hub Natural Gas Spot Price).
coeftest(mod,vcov=vcovHAC(mod))

# HC standard errors: 
coeftest(mod,vcov=vcovHC(mod))

# linearHypothesis is also a helpful for joint F-tests (doesn't do t-tests). 
# for a single univariate test, the F-stat is the square of the t-stat.
testgas <- linearHypothesis(mod,c("MHHNGSP=0"),vcov=vcovHAC(mod),test="F",data=mod)
testgas
tstat <- sqrt(testgas$F)
tstat

# Adjusting model for autocorrelation -----------------------------------

# residuals are very much autocorrelated. 
# Also, I know that drilling (IPN213111N) is autocorrelated.
# Two options: 
#       - include lags of Y (and maybe X's) using dynlm (dynamic linear model), 
#       - or model the residuals using arima() (autoregressive integrated moving average)
#       - (or both)
#       - dynlm and arima functions do not like xts objects - need to reformat data

godchng = ts(goddiffs,freq=12,start=c(1997,2))  # monthly frequency, starts in Feb 1991

# Model the residuals using arima() -----------------------------------

# column 3 is drilling, column 1 is natural gas price, column 2 is oil price
# include 2 lag of residuals order=c(2,0,0)
mod_ar <- arima(goddiffs[,3],order=c(2,0,0),xreg=cbind(goddiffs[,1],goddiffs[,2]),
                include.mean=T)
summary(mod_ar)
acf(residuals(mod_ar))
tsdisplay(residuals(mod_ar))
bt_ar = Box.test(mod_ar$residuals,lag=7) # uses 7 degrees of freedom, because log(T) = 6.8
                                  # but we estimated 4 params: 2 coefs., 2 lag
                                  # should us 7 - 4 = 3 degrees of freedom
pv = 1-pchisq(bt_ar$statistic,3) 
pv
# p-value is large - accept null of no autocorrelation

coeftest(mod_ar,vcov=vcov(mod_ar))

# don't need HAC standard errors - model for residuals already captures autocorrelation

# However, negative and significant oil price is weird. 
# Is drilling affecting oil prices instead of other way around?
# I might have fixed one problem but still have endogenous regressor. 

# Dynamic linear model ------------------------------------

# Try a dynamic model using dynlm (dynamic linear model)

# Include 2 lags of drilling. Why? We'll dig into this more later. 
dyog <- dynlm(godchng[,"IPN213111N"] ~ L(godchng[,"IPN213111N"],c(1:2))+
                godchng[,"MHHNGSP"]+godchng[,"MCOILWTICO"])
              
summary(dyog)
tsdisplay(residuals(dyog))
# residual autocorrelation is gone, therefore I don't need HAC standard errors.

# just to be sure, let's see coefficients with HC standard errors:
coeftest(dyog,vcov=vcovHC(dyog))
# oil and gas not significant at the 5 percent level. Oil is at 10.

# oil & gas prices are highly correlated, try a joint test
cor(goddiffs$MHHNGSP,goddiffs$MCOILWTICO)
# Notice a syntax issue with the quotes in the variable names:
linearHypothesis(dyog,c("godchng[, "MHHNGSP"]=0","godchng[, "MCOILWTICO"]=0"),
                 vcov=vcovHC(dyog),test="F",data=dyog)
# reestimate same model with different variable names:
dyog <- dynlm(godchng[,3]~L(godchng[,3],c(1:2))+godchng[,1]+godchng[,2])
linearHypothesis(dyog,c("godchng[, 1]=0","godchng[, 2]=0"),vcov=vcovHC(dyog),test="F",data=dyog)
# fail to reject the null at 5 percent - seems oil and gas prices don't matter

# that seems odd! maybe it takes several months for price shocks to matter. 
# try one lag of gas price and several lags of oil price
dyog.lags <- dynlm(godchng[,3]~L(godchng[,3],c(1:2))+L(godchng[,1],c(1))+
                     L(godchng[,2],c(0:4)))
summary(dyog.lags)
tsdisplay(residuals(dyog.lags))
coeftest(dyog.lags,vcov=vcovHAC(dyog.lags))
