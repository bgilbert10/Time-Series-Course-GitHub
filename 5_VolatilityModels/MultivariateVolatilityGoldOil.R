# Examples of multivariate volatility models for dynamic relationships

# First between gold futures returns and changes in TIPS. 
# Second in a Kilian (2009)-like oil supply/demand example.

# Mostly using Tsay's multivariate time series (MTS) package. 

require(MTS)
require(forecast)
require(quantmod)
require(vars)
require(fBasics)
require(tseries)
require(zoo)

# Gold and TIPS example --------------

## 1) Organize Data ---------------
getSymbols("GC=F")                          #gold futures
getSymbols("DFII10",src="FRED")             #10y TIPS const maturity

grd <- merge.xts(`GC=F`$`GC=F.Adjusted`, DFII10$DFII10, 
                 all=TRUE, fill=NA, retclass="xts")

colnames(grd) <- c("gld","tip")

#SELECT START AND END DATE
startdate <- "2019-01-03"    
enddate   <- "2021-11-30"    

grd <- window(grd, start = startdate, end = enddate)
grd <- na.omit(grd)

chartSeries(grd$gld)
chartSeries(grd$tip)

dlgld  = ts(na.omit(diff(log(grd$gld))),freq=252,start = 2019)
dtip   = ts(na.omit(diff(grd$tip)),freq=252,start = 2019)

ret = cbind(dlgld,dtip)

## 2) Fit the VAR and capture the residuals for volatility modeling -------------
VARselect(ret,lag.max=10,type="const")
gldvar = vars::VAR(ret,p=1,type="const")
plot(irf(gldvar,impulse="dtip",response="dlgld",ortho=TRUE))
res = residuals(gldvar)
# check out volatility clustering in both residuals
ts.plot(res[,1])
ts.plot(res[,2])
arch.test(gldvar)
plot(arch.test(gldvar,lags.multi=30))

## 3) Estimate the exponentially weighted moving average volatility model -----------
ewgld = EWMAvol(res)
# column 1 is gold volatility, 4 is tips volatility, 2 and 3 are gold-tips covariance
head(ewgld$Sigma.t)
# gold residual conditional volatility over time:
ts.plot(ewgld$Sigma.t[,1])
# gold and tips residual correlation over time:
ts.plot(ewgld$Sigma.t[,2])
# tips residual conditional volatility over time:
ts.plot(ewgld$Sigma.t[,4])

## 4) Estimate the dynamic conditional correlation volatility model -------------
# dccPre fits the VAR(1) with intercept, and computes univariate residual GARCH
gldvar.garch <- dccPre(ret,p=1,include.mean=TRUE)
# GARCH-standardized residuals: 
head(gldvar.garch$sresi)
# gold residuals from regular VAR(1): 
ts.plot(res[,1])
# univariate GARCH-standardized gold residuals:
ts.plot(gldvar.garch$sresi[,1])

# residual volatility:
head(gldvar.garch$marVol)
# gold equation residual volatility
ts.plot(gldvar.garch$marVol[,1])
# tips equation residual volatility
ts.plot(gldvar.garch$marVol[,2])

# take standardized residuals from univariate GARCH and plug in to DCC model:
gldDCC = dccFit(gldvar.garch$sresi)
head(gldDCC$rho.t)
# cross equation residual correlation: 
ts.plot(gldDCC$rho.t[,2])

## 5) BEKK model (properly multivariate GARCH) -----------------
# takes a long time to run, can handle at most 3 variables/equations
# doesn't converge when using the VAR residuals for some reason
gld.bekk = BEKK11(ret, include.mean=T)
# gold volatility
ts.plot(gld.bekk$Sigma.t[,1])
ts.plot(gld.bekk$Sigma.t[25:710,1])
# tips volatility
ts.plot(gld.bekk$Sigma.t[,4])
ts.plot(gld.bekk$Sigma.t[25:710,4])
# cross equation correlation
ts.plot(gld.bekk$Sigma.t[,2])
ts.plot(gld.bekk$Sigma.t[25:710,2])


# Kilian-like oil VAR example --------------

## 1) Organize Data ---------------
# WTI
getSymbols("MCOILWTICO",src="FRED")
# Oil & gas drilling index
getSymbols("IPN213111N",src="FRED")
# Industrial Production
getSymbols("INDPRO",src="FRED")

# merge oil, gas, and drilling and calculate returns/changes
oilgas= merge.xts(MCOILWTICO,IPN213111N,INDPRO,join="inner")
# calculate log differences as ts() objects, notice start dates
doil = ts(na.omit(diff(log(oilgas$MCOILWTICO))),freq=12,start=1986+1/12)
dwell = ts(na.omit(diff(oilgas$IPN213111N)),freq=12,start=1986+1/12)
dind = ts(na.omit(diff(oilgas$INDPRO)),freq=12,start=1986+1/12)

kil = cbind(dwell,dind,doil)
# show series 
plot(kil,xlab="")


## 2) Fit the VAR and capture the residuals for volatility modeling -------------
VARselect(kil,lag.max=13,type="none")
# SC (BIC) select 1 lags, same as below:
varkil = vars::VAR(kil,p=1,type="none")

plot(irf(varkil,impulse="dwell",response="doil",ortho=TRUE))
res = residuals(varkil)
# check out volatility clustering 
ts.plot(res[,1])
ts.plot(res[,2])
ts.plot(res[,3])
arch.test(varkil)
plot(arch.test(varkil,lags.multi=30))

## 3) Estimate the exponentially weighted moving average volatility model -----------
ewoil = EWMAvol(res)
# this is a covariance matrix in each time period, (1) to (3) are first row, etc.
head(ewoil$Sigma.t)
# drilling residual conditional volatility over time:
ts.plot(ewoil$Sigma.t[,1])
# drilling and ind prod residual correlation over time (2 and 4):
ts.plot(ewoil$Sigma.t[,2])
# drilling and oil return residual correlation over time (3 and 7):
ts.plot(ewoil$Sigma.t[,3])
# ind prod residual conditional volatility over time:
ts.plot(ewoil$Sigma.t[,5])
# ind prod and oil return residual correlation over time (6 and 8):
ts.plot(ewoil$Sigma.t[,8])
# oil return residual conditional volatility over time:
ts.plot(ewoil$Sigma.t[,9])

## 4) Estimate the dynamic conditional correlation volatility model -------------
# dccPre fits the VAR(1) with intercept, and computes univariate residual GARCH
oilvar.garch <- dccPre(kil,p=1,include.mean=TRUE)
head(oilvar.garch$sresi)
# drilling residuals: 
ts.plot(res[,1])
# univariate GARCH-standardized drilling residuals:
ts.plot(oilvar.garch$sresi[,1])
# oil return residuals: 
ts.plot(res[,3])
# univariate GARCH-standardized oil return residuals:
ts.plot(oilvar.garch$sresi[,3])

head(oilvar.garch$marVol)
# drilling equation residual volatility
ts.plot(oilvar.garch$marVol[,1])
# oil return equation residual volatility
ts.plot(oilvar.garch$marVol[,3])

# take standardized residuals from univariate GARCH and plug in to DCC model:
oilDCC = dccFit(oilvar.garch$sresi)
head(oilDCC$rho.t)
# cross equation residual correlation: 
ts.plot(oilDCC$rho.t[,2])

## 5) BEKK model (properly multivariate GARCH) -----------------
# takes a long time to run, can handle at most 3 variables/equations
# doesn't converge when using the VAR residuals for some reason
oil.bekk = BEKK11(kil, include.mean=T)
# drilling volatility
ts.plot(oil.bekk$Sigma.t[,1])
ts.plot(oil.bekk$Sigma.t[25:429,1])
# ind prod volatility
ts.plot(oil.bekk$Sigma.t[,5])
ts.plot(oil.bekk$Sigma.t[25:429,5])
# oil return volatility
ts.plot(oil.bekk$Sigma.t[,9])
ts.plot(oil.bekk$Sigma.t[25:429,9])
# cross equation drilling, ind prod correlation
ts.plot(oil.bekk$Sigma.t[,2])
ts.plot(oil.bekk$Sigma.t[25:429,2])

