############
# Let's look at a natural gas example
# I uploaded NGInventoryEastRegionBCF.csv to Canvas - a time series of natural gas storage inventories

require(tseries)
require(lubridate)
require(forecast)
require(CADFtest)
require(quantmod)
require(fBasics)
require(urca)

# I stored it in the directory below, so let's point R there:
setwd("C:/Users/gilbe/Dropbox/Econometrics/TimeSeriesCourse/Fall2017/TsayCh2/Lecture/")
stg = read.table('NGInventoryEastRegionBCF.csv',header=T,sep=",") 
# Let's save it as a time series object. We need the "lubridate" package to allow
# a non-integer seasonal frequency of 365.25/7 (this is weekly data)
store = ts(stg$EastNGInventoryBCF,freq=365.25/7, start=2010)
ts.plot(store)
# this does not appear to be drifting/trending but there is strong seasonality
# It also doesn't LOOK like it has a unit root, but that can be deceiving

# One advantage of CADFtest is that I can control for other X variables while running the ADF test
# In this case, let's use Fourier terms to control for the seasonality, and 
# test for a unit root in the non-seasonal component of the data
# Here I'm using 4 Fourier terms - you could play around and see if more or less Fourier terms reduces the AIC
cadfstg = CADFtest(store,criterion=c("AIC"),type="drift",X=fourier(store,K=c(4)))
summary(cadfstg)
# Notice that the output has the regression coefficients on the Fourier terms (sines and cosines)
# I fail to reject the null of a unit root - Surprising!

# We can better see the unit root by stripping out the seasonality
# Estimate the Fourier terms and nothing else, and take the residuals:
x = Arima(store,order=c(0,0,0),xreg=fourier(store,K=c(4)))
ts.plot(residuals(x))
tsdiag(x,gof=25)
tsdisplay(residuals(x))
# The residuals look more like a random walk WITHOUT drift (Case 2)
# We can test the residuals using what we know:
cadfres = CADFtest(residuals(x),criterion=c("AIC"),type="drift")
summary(cadfres)

# In both iterations of the CADFtest, the output had one extra lag as well as a unit root
# So, conditional on the Fourier terms, an ARIMA(1,1,0) might be a good model:
x2 = Arima(store,order=c(1,1,0),xreg=fourier(store,K=c(4)))
summary(x2)
tsdiag(x2,gof=25)
tsdisplay(residuals(x2))
# Still some residual autocorrelation, but we're on the right track at least
# auto.arima agreed with me, anyway:
x3 = auto.arima(store,xreg=fourier(store,K=c(4)),seasonal=FALSE)
x3
# let's forecast it and compare it to a holdout sample:
# hold out  the last 52 weeks (1 year) and estimate on the remaining subsample (periods 1 to 352):
store2 = ts(store[-c(352:404)],freq=365.25/7, start=2010)
forecast_holdout <- Arima(store[-c(352:404)],order=c(1,1,0),include.drift = FALSE,
                          xreg=fourier(store2,K=c(4)))
# calculate the forecast from period 352 two years (104 weeks) ahead (52 remaining in data plus 52 into the future)
fcast_holdout <- forecast(forecast_holdout,xreg=fourier(store2,K=c(4),h=104))
# plot both the forecast using plot, and the data using lines:
plot(fcast_holdout,main="ARIMA(1,1,0) with 4th order harmonic regression",include=156)
lines(ts(store))

# takes awhile to run, produces SARIMA(1,0,1)(1,1,0) with drift
auto.arima(store,seasonal=TRUE)
store.sarima <- Arima(store,order = c(1,0,1),seasonal = list(order=c(1,1,0)),include.constant = T)

# let's forecast this one and compare it to a holdout sample:
# hold out  the last 52 weeks (1 year) and estimate on the remaining subsample (periods 1 to 352):
store2 = ts(store[-c(352:404)],freq=365.25/7, start=2010)
forecast_hold_sar <- Arima(store[-c(352:404)],order = c(1,0,1),seasonal = list(order=c(1,1,0)),include.constant = T)
# calculate the forecast from period 352 two years (104 weeks) ahead (52 remaining in data plus 52 into the future)
fcast_hold_sar <- forecast(forecast_hold_sar,h=104)
# plot both the forecast using plot, and the data using lines:
plot(fcast_hold_sar,main="ARIMA(1,0,1) with SARIMA(1,1,0) and drift",include=156)
lines(ts(store))

fcast_hold_full <- forecast(store.sarima,h=104)
# plot both the forecast using plot, and the data using lines:
plot(fcast_hold_full,main="ARIMA(1,0,1) with SARIMA(1,1,0) and drift",include=156)
lines(ts(store))

# try a different model on the pre-hold-out sample
auto.arima(store[-c(352:404)],seasonal=TRUE)
f3 <- Arima(store[-c(352:404)],order = c(3,0,0),include.constant = T)
# calculate the forecast from period 352 two years (104 weeks) ahead (52 remaining in data plus 52 into the future)
fc3 <- forecast(f3,h=104)
# plot both the forecast using plot, and the data using lines:
plot(fc3,main="ARIMA(3,0,0) with SARIMA(0,0,0) and drift",include=156)
lines(ts(store))
