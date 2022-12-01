# Do gold futures have ARCH/GARCH effects?
# investigate the properties of the gold futures price

require(quantmod)
require(forecast)
require(tseries)
require(vars)
require(fGarch)

# 1) Organize data ----------------------

# Read data from Yahoo!Finance
getSymbols("GC=F")
head(`GC=F`)

# Let's pick a fixed window so that answers don't change every day
gold <- `GC=F`$`GC=F.Adjusted`
gold <- gold[paste("2007-01-01","2021-12-05",sep="/")]

goldrtn <- diff(log(gold[,1]))
ts.plot(goldrtn) # obvious volatility clustering

# 2) ARIMA model for the mean ---------------------------

# get a model for the mean that handles autocorrelation, calculate the residuals. 
auto.arima(goldrtn) 
# ARIMA(0,0,0) with non-zero mean, but has autocorrelation
res0 = residuals(arima(na.omit(goldrtn),order=c(0,0,0),include.mean=T))
tsdisplay(res0)
Box.test(res0,lag=10,type='Ljung')
Box.test(res0,lag=20,type='Ljung')
# try ARIMA(1,0,2) - fail to reject null of no autocorrelation
res = residuals(arima(na.omit(goldrtn),order=c(1,0,2),include.mean=T))
tsdisplay(res)
Box.test(res,lag=10,type='Ljung')
Box.test(res,lag=30,type='Ljung')

# 3) GARCH modeling -------------------------

# There is clear autocorrelation in squared residuals
res2 = res^2
acf(res2)
pacf(res2)
Box.test(res2,lag=10,type='Ljung')
Box.test(res2,lag=30,type='Ljung')

goldrtn <- na.remove(goldrtn)
# Estimated daily mean and volatility models jointly
# GARCH(2,1) gets rid of a lot (not all) of the squared residual autocorrelation
# Note garch order is reversed (sig_t AR(1) comes 2nd, sig_t MA(2) comes first)
GoldenArch = garchFit(~arma(1,2)+garch(2,1),data=goldrtn,trace=F,cond.dist='sstd',include.mean=T)


# 4) Evaluate GARCH model ------------------

# still some autocorrelation in squared residuals
summary(GoldenArch)


tsdisplay(residuals(GoldenArch))
acf(GoldenArch@sigma.t)

# mu, ar1, ma1 to ma2 are for the mean. 
# omega, alpha (ARCH coefficents), beta (GARCH coefficients)

# r(t) = 0.00055 - 0.902*r(t-1) + a(t) + 0.872*a(t-1) - 0.0312*a(t-2)

# model for the volatility:
# sig^2(t) = 0.00000063 + 0.96*sig^2(t-1) + 0.013*a^2(t-1) + 0.023*a^2(t-2) + 

# notice how many options with plot
plot(GoldenArch)

# Plot returns alongside volatilities (conditional SDs)
# we can see volatilities spike when return tails get fat
par(mfcol=c(2,1))
plot(GoldenArch, which=1)
plot(GoldenArch, which=8)
par(mfcol=c(1,1))

# Forecast using the fitted ARIMA-GARCH model
ga = predict(GoldenArch,n.ahead=100,plot=T,nx=500,mse=c("cond"),crit_val=1.96)


