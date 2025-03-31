# Some R Time Series Basics

# ts() can transform data frames, vectors, and arrays into ts object
# often a time series function will automatically make a ts object

require(quantmod)
getSymbols("MCOILWTICO",src="FRED")
oil = ts(MCOILWTICO$MCOILWTICO)


# can calculate differences and lags of various lengths
# here, one difference across two months
x2a = diff(oil,lag=2,difference=1)
x2b = oil - lag(oil, -2)
plot(x2a,x2b)

# lm() may not understand the lag() function or other ts() options
# could do more awkward thing, ensure conformable dimensions
lag.oil = oil[-NROW(oil)]
oil.small = oil[-1]
lm(oil.small ~ lag.oil)

# to use lm() could also create dataframe with all the lags
lagoil = lag(oil,-1)
d = ts.union(oil,lagoil)
# or
d = ts.union(oil,lagoil=lag(oil,-1))
lm(oil ~ lagoil,data=d)


# filter() command can do AR and MA filtering

# e.g. input x to y = theta(L)*x for MA (convolution) filter
# e.g. input x to y = phi^(-1)(L)x for AR (recursive) filter
# need to already know coefficients
# use sides=1 for MA/convolution, otherwise will be centered MA
oil.rtn = diff(log(oil))
yMA = filter(oil.rtn,c(1,0.2,-0.35,0.1),method="convolution",sides=1)
yMA2 = oil.rtn+0.2*lag(oil.rtn,-1)-0.35*lag(oil.rtn,-2)+0.1*lag(oil.rtn,-3)
plot(yMA,yMA2)
yAR = filter(oil.rtn,c(0.2,-0.35,0.1),method="recursive")

# in this example, they are pretty close:
plot(oil.rtn,plot.type="s",col=1)
lines(yMA,col=2)
lines(yAR,col=3)


# More advanced filtering, e.g., Kalman filtering
# simple example estimating Nile river flow rates
# from Tusell, 2011, Kalman Filtering in R, Journal of Statistical Software
# paper reviews packages dse, dlm, FKF, KFAS, and sspir (discontinued)
install.packages("dse")
require(dse)
m1.dse = dse::SS(F=matrix(1,1,1),Q=matrix(40,1,1),
                H=matrix(1,1,1),R=matrix(130,1,1),z0=matrix(0,1,1),P0=matrix(10^5,1,1))
data("Nile",package="datasets")
m1b.dse.est = estMaxLik(m1.dse,TSdata(output=Nile))
plot(Nile,plot.type="s",col=1)
lines(m1b.dse.est$estimates$pred,col=2)


# ARIMA modeling

# basic:
ar1 = arima(oil.rtn,order=c(1,0,0))
ma1 = arima(oil.rtn,order=c(0,0,1))

# estimate higher order but set first few coefficients to zero
# ARMA(2,1) with first AR coefficient set to zero 
# 4 total coefficients in the model including intercept
arma21 = arima(oil.rtn,order=c(2,0,1),fixed=c(0,NA,NA,NA))

# for AR only, and letting package pick a reasonable order
# fits AR(1)
ar.oil = ar(oil.rtn,order.max = 5)

# can use fracdiff to do fractional differencing model
require(fracdiff)
fracdiff(log(oil),nar=1,nma=1)


# ARCH/GARCH modeling
# using garch() from tseries package
arch.oil = garch(oil.rtn,order=c(0,3))
garch.oil = garch(oil.rtn,order=c(3,2))
# recover and plot estimates of the volatilities
plot(fitted(arch.oil))
plot(cbind(arch.oil$fitted.values[,1],garch.oil$fitted.values[,1],oil.rtn)
          , plot.type="m")
# study coefficients (a0 is intercept of volatility process)
arch.oil$coef
# b's are the autoregressive volatility (GARCH) coefficients, 
# a's are the moving average squared residual (ARCH) coefficients
garch.oil$coef

# include exogenous variables or AR terms in the mean equation
# (aside: a great rugarch example https://stats.stackexchange.com/questions/93815/fit-a-garch-1-1-model-with-covariates-in-r)
install.packages("rugarch")
require(rugarch)

# here in garchOrder, MA terms are first. So GARCH(3,2) is as below
fitgarch.oil.spec = ugarchspec(variance.model=list(model="sGARCH",
                                              garchOrder=c(2,3)),
                          mean.model = list(armaOrder=c(1,1),
                                            include.mean=TRUE),
                          distribution.model = "norm")
fitgarch.oil = ugarchfit(data=oil.rtn, spec = fitgarch.oil.spec)

# coefficients and standard errors
# omega is the intercept of the volatility process
coef(fitgarch.oil)
diag(vcov(fitgarch.oil))
# confidence intervals
conf.lb = coef(fitgarch.oil)+qnorm(0.025)*diag(vcov(fitgarch.oil))
conf.ub = coef(fitgarch.oil)+qnorm(0.975)*diag(vcov(fitgarch.oil))

# plot coefficients and confidence intervals
plot(coef(fitgarch.oil), pch = 1, col = "red",
     ylim = range(c(conf.lb, conf.ub)),
     xlab = "", ylab = "", axes = FALSE)
box(); axis(1, at = 1:length(coef(fitgarch.oil)), labels = names(coef(fitgarch.oil))); axis(2)
for (i in 1:length(coef(fitgarch.oil))) {
  lines(c(i,i), c(conf.lb[i], conf.ub[i]))
}
legend( "topleft", legend = c("estimate", "confidence interval"),
        col = c("red", 1), pch = c(1, NA), lty = c(NA, 1), inset = 0.01)

# Correlograms
par(mfrow=c(2,1))
acf(oil.rtn,ci.type="ma")
pacf(oil.rtn)

# Predict. Compare predictions of ARMA(1,1) to ARMA(1,1)+GARCH(3,2)
library(forecast)
armaoil = arima(oil.rtn,order=c(1,0,1))
pr.armaoil = predict(armaoil,12)
fc.armaoil <- forecast::forecast(armaoil,h=12)
pr.garchoil = ugarchforecast(fitgarch.oil,n.ahead=12)
par(mfrow=c(2,1))
plot(fc.armaoil,include=101)
plot(pr.garchoil)


# Tests

# Durbin-Watson
require(car)
# input results of a lm() model
# rejects lack of autocorrelation in levels regression, not in returns
durbinWatsonTest(lm(oil.small ~ lag.oil),max.lag=2)
durbinWatsonTest(lm(diff(log(oil.small)) ~ diff(log(lag.oil))),max.lag=2)

# Box tests (Box-Pierce, Ljung-Box)
Box.test(arima(oil.rtn,order=c(1,0,1))$resid,type="Ljung-Box")

# Dickey-Fuller
adf.test(oil)
adf.test(log(oil))


# VAR
# simplest approach is to use ar() on vectors bound together
getSymbols("MCOILBRENTEU",src="FRED")

oil=ts(MCOILWTICO$MCOILWTICO,freq=12,start=1986)
oil=ts(oil[c(73:378)],freq=12,start=1992)
boil=ts(MCOILBRENTEU$MCOILBRENTEU,freq=12,start=1987+(4/12))
boil=ts(boil[c(57:362)],freq=12,start=1992)

oil.rtn = diff(log(oil))
boil.rtn = diff(log(boil))

oilret = cbind(oil.rtn,boil.rtn)

ar(oilret,order=2)
ar(oilret)

# more interesting:
require(vars)
varbo = VAR(oilret,type="none",p=3)
summary(varbo)
plot(varbo,name="oil.rtn")
causality(varbo,cause="boil.rtn")
plot(irf(varbo,n.ahead=24,ortho=T))
plot(fevd(varbo,n.ahead=24))
