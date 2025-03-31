# Does the Gold commodity price have ARCH/GARCH effects?

require(quantmod)
require(forecast)
require(tseries)
require(vars)
require(fGarch)

# investigate the properties of the daily 3 p.m. gold per troy ounce 
# price in London Bullion market, US dollars
getSymbols("GC=F")
head(`GC=F`$`GC=F.Adjusted`)

# FRED does not allow selecting specific dates, so let's pick a fixed window
# so that answers don't change every day
gold <- `GC=F`$`GC=F.Adjusted`
gold <- gold[paste("2007-04-01","2024-10-05",sep="/")]

goldrtn <- diff(log(gold[,1]))
ts.plot(goldrtn) # obvious volatility clustering

auto.arima(goldrtn) # ARMA(0,0)
tsdisplay(residuals(arima(na.omit(goldrtn),order=c(0,0,0),include.mean=T)))
res = residuals(arima(na.omit(goldrtn),order=c(0,0,0),include.mean=T))
res2 = res^2

# residual autocorrelation is not bad
tsdisplay(res)
acf(res)
Box.test(res,lag=10,type='Ljung')
Box.test(res,lag=20,type='Ljung')

# clear autocorrelation in squared residuals
acf(res2)
pacf(res2)
Box.test(res2,lag=10,type='Ljung')
Box.test(res2,lag=20,type='Ljung')

goldrtn <- na.remove(goldrtn)
# Estimated daily mean and volatility models jointly
# GARCH(2,1) gets rid of a lot of the residual autocorrelation
GoldenArch = garchFit(~arma(0,3)+garch(2,1),data=goldrtn,trace=F,cond.dist='sstd',include.mean=T)

# still some autocorrelation in residuals and squared residuals
summary(GoldenArch)


tsdisplay(residuals(GoldenArch))
acf(GoldenArch@sigma.t)

# mu, ma1 to ma3 are for the mean. 
# omega, alpha (ARCH coefficents), beta (GARCH coefficients)
# r(t) = 0.000326 + a(t) - 0.031*a(t-1) - 0.0068*a(t-2) + 0.017*a(t-3)

# model for the volatility:
# sig^2(t) = 0.00000077 + 0.00788*a^2(t-1) + 0.0298*a^2(t-2) +0.958*sig^2(t-1) 

# notice how many options with plot
plot(GoldenArch)

# Plot returns alongside volatilities (conditional SDs)
# we can see volatilities spike when return tails get fat
par(mfcol=c(2,1))
plot(GoldenArch, which=1)
plot(GoldenArch, which=8)
par(mfcol=c(1,1))

ga = predict(GoldenArch,n.ahead=100,plot=T,nx=500,mse=c("cond"),crit_val=1.96)

# Concatenate the fitted model with the prediction, transform to time series
dat <- as.ts(c(sqrt(GoldenArch@h.t), ga = ga$standardDeviation))

# Create the plot
plot(window(dat, start = start(dat), end = 4400), col = "blue",
     xlim = range(time(dat)), ylim = range(dat),
     ylab = "Conditional SD", main = "Prediction based on GARCH model")

par(new=TRUE)

plot(window(dat, start = 4400), col = "red", axes = F, xlab = "", ylab = "", xlim = range(time(dat)), ylim = range(dat))

# Zoomed in on the plot
plot(window(dat, start = 3500, end = 4400), col = "blue",
     xlim = range(3500,4500), ylim = range(dat),
     ylab = "Conditional SD", main = "Prediction based on GARCH model")

par(new=TRUE)

plot(window(dat, start = 4400), col = "red", axes = F, xlab = "", ylab = "", xlim = range(3500,4500), ylim = range(dat))

