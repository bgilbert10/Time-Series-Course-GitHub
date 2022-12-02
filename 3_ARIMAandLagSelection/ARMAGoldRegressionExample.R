###### Preamble ####
# In class activity 
# 1) download gold prices, TIPS, Break-even inflation, Trade-weighted dollar 
#     index and merge them for a common time window of 01-03-2008 to 06-30-2020
# 2) investigate stationarity of levels (no need for formal tests yet)
# 3) evaluate an arima model for the returns/changes of each series
# 4) run a regression of gold returns on the first differences of the other 
#       variables to evaluate how gold returns correlate to macro variables
# 5) evaluate the residuals for autocorrelation and adjust the model as needed

require(forecast)
require(quantmod)
require(caschrono)
require(texreg)
require(sandwich)
require(lmtest)


# 1) Read and Organize Data --------------
getSymbols("GC=F")                          #gold futures
getSymbols("DFII10",src="FRED")             #10y TIPS const maturity
getSymbols("T10YIE",src="FRED")             #10y breakeven inflation
getSymbols("DTWEXBGS",src="FRED")           #trade-wtd USD index (goods & svcs, select which)


grd <- merge.xts(`GC=F`$`GC=F.Adjusted`, DFII10$DFII10, 
                 T10YIE$T10YIE, DTWEXBGS,
                 all=TRUE, fill=NA, retclass="xts")

colnames(grd) <- c("gld","tip", "bei","twd")

#SELECT START AND END DATE
startdate <- "2008-01-03"    
enddate   <- "2020-06-30"    

grd <- window(grd, start = startdate, end = enddate)
grd <- na.omit(grd)


# 2) Investigate autocorrelation -----------------
chartSeries(grd$gld)
acf(grd$gld)
chartSeries(grd$tip)
acf(grd$tip)
chartSeries(grd$bei)
acf(grd$bei)
chartSeries(grd$twd)
acf(grd$twd)


# 3) Autoregressive models for first differences --------------

# Use log differences for gold price for log returns, but simple differences
# for other variables, which are already measured as rates or indices. 
dlgld  = ts(na.omit(diff(log(grd$gld))))
dtip   = ts(na.omit(diff(grd$tip)))
dbei   = ts(na.omit(diff(grd$bei)))
dtwd   = ts(na.omit(diff(grd$twd)))

## Gold returns ---------------

# all approaches agree on white noise for gold returns
armaselect(dlgld)
auto.arima(dlgld)
ar(dlgld)
tsdisplay(dlgld)

## TIPS ------------- 

# armaselect says ARMA(1,2), but uses BIC which penalizes parameters more heavily. 
armaselect(dtip)
# auto.arima says ARMA(1,2)
auto.arima(dtip)
# ar() says AR(8)
ar(dtip)
# Only the AR(8) cleans up the autocorrelation
tsdiag(arima(dtip,order=c(1,0,2),include.mean=F),gof=15)
tsdiag(arima(dtip,order=c(8,0,0),include.mean=F),gof=15)

## Break-even inflation ----------------

# armaselect says ARMA(1,0)
armaselect(dbei)
# auto.arima says ARMA(0,1)
auto.arima(dbei)
# ar() says AR(4)
ar(dbei)
# ARMA(0,1) and AR(2) both do pretty well
tsdiag(arima(dbei,order=c(1,0,0),include.mean=F),gof=15)
tsdiag(arima(dbei,order=c(0,0,1),include.mean=F),gof=15)
tsdiag(arima(dbei,order=c(2,0,0),include.mean=F),gof=15)

## Trade-weighted dollar index ---------------

# armaselect says ARMA(0,0)
armaselect(dtwd)
# auto.arima says ARMA(0,1)
auto.arima(dtwd)
# ar() says AR(1)
ar(dtwd)
# neither of these looks okay
tsdiag(arima(dtwd,order=c(0,0,1),include.mean=F),gof=15)
tsdiag(arima(dtwd,order=c(1,0,0),include.mean=F),gof=15)
# after playing around a bit:
tsdiag(arima(dtwd,order=c(3,0,2),include.mean=F),gof=15)


# 4) Regression -------------------

# currency-related macro variables are significant explanatory factors in
# gold returns. 
mod <- lm(dlgld ~ dtip + dbei + dtwd)
summary(mod)


# 5) Modeling residual serial correlation ----------------

# same results, but easier to evaluate residuals:
mod1 <- arima(dlgld,order=c(0,0,0),xreg=cbind(dtip,dbei,dtwd),include.mean = T)
# clear autocorrelation in residuals from Ljung-Box tests: 
tsdiag(mod1)
tsdisplay(residuals(mod))

# auto.arima says MA(2) errors are optimal
auto.arima(dlgld,xreg=cbind(dtip,dbei,dtwd))

mod2 <- arima(dlgld,order=c(0,0,2),xreg=cbind(dtip,dbei,dtwd),include.mean = T)
# standard errors are very, very similar
mod2

# Ljung-Box tests look better, but still some significant individual ACFs.
# is it enough to worry about? Maybe not. 
tsdiag(mod2, gof=40)
tsdisplay(residuals(mod2))

# I did some trial and error. It takes a large model to get white noise errors. 

# alternatively, compare results from OLS standard errors to HAC standard errors
# HAC standard errors increase a bit. 
coeftest(mod)
coeftest(mod,vcov=vcovHAC(mod))

# which one to report? MA(1) errors, large ARMA errors, or basic model with HAC? 
# my view: just report them all. 


  