##########################
# In class activity 10/26
# 1) download daily gold prices, TIPS, Break-even inflation, Trade-weighted dollar 
#     index, S&P index, and WTI crude oil price and merge them for a common time 
#     window of 01-03-2007 to 09-30-2020
# 2) investigate cointegrating relationships between these series
# 3) estimate the VECM and calculate IRFs

require(forecast)
require(quantmod)
require(caschrono)
require(texreg)
require(fBasics)
require(CADFtest)
require(urca)
require(sandwich)
require(lmtest)
require(nlme)
require(MTS)
require(car)
require(strucchange)
require(vars)


# 1) 
getSymbols("GC=F")   #gold fixing 1030am london
getSymbols("DFII10",src="FRED")             #10y TIPS const maturity
getSymbols("T10YIE",src="FRED")             #10y breakeven inflation
getSymbols("DTWEXBGS",src="FRED")           #trade-wtd USD index (goods & svcs, select which)
getSymbols("SPY")                           #S&P500
getSymbols("DCOILWTICO",src="FRED")         #daily WTI crude oil price

grd <- merge.xts(`GC=F`$`GC=F.Adjusted`, DFII10$DFII10, 
                 T10YIE$T10YIE, DTWEXBGS, SPY$SPY.Close, DCOILWTICO$DCOILWTICO,
                 all=TRUE, fill=NA, retclass="xts")

colnames(grd) <- c("gld","tip", "bei","twd","sp5","oil")

#SELECT START AND END DATE
startdate <- "2007-01-03"    
enddate   <- "2021-10-29"    

grd <- window(grd, start = startdate, end = enddate)
grd <- na.omit(grd)

MTSplot(grd)


# Johansen's test simultaneously estimates the VECM
## How many lags should there be in the VAR in levels (remember there will be
## one fewer lags in the VECM because algebra)?
## If cointegrated, VAR in levels is okay, so I can use it to test for lag length
## Looks like 2 lags of VAR in levels is preferred by HQ and SC(BIC):
VARselect(grd,lag.max=20,type="const")

# Johansen test with K=2 lags in the levels VAR
xjl = ca.jo(grd,type="trace",ecdet="const",K=2,spec="transitory")
summary(xjl)
# looks like 3 cointegrating relationships/vectors

cajools(xjl)  # raw coefficients
cajorls(xjl,r=3) # coefficients from each component of the VECM
plotres(xjl)  # checking residual autocorrelation from each of the 6 equations
### And each cointegrating residual should be stationary:
ts.plot(cajorls(xjl,r=3)$rlm$model$ect1)
ts.plot(cajorls(xjl,r=3)$rlm$model$ect2)
ts.plot(cajorls(xjl,r=3)$rlm$model$ect3)

# translate my VECM in changes into a VAR in levels
x = vec2var(xjl,r=3)

# Forecast levels 200 days ahead
xp = predict(x,n.ahead=200)
plot(xp)

# Notice that impulse responses exhibit permanent effects
irfgold = irf(x,n.ahead=24) 
plot(irfgold)
# hard to see when all y axes are on the same scale. 

# Looks like a one-standard deviation permanent increase in tips
# causes a $2.50 permanent decline in the gold price
plot(irf(x,n.ahead=24,impulse = "tip", response = "gld"))

# we can investigate other pairwise IRFs:
plot(irf(x,n.ahead=24,impulse = "bei", response = "gld"))
plot(irf(x,n.ahead=24,impulse = "twd", response = "gld"))
plot(irf(x,n.ahead=24,impulse = "sp5", response = "gld"))
plot(irf(x,n.ahead=24,impulse = "oil", response = "gld"))

# Question: considering the Orthogonal IRFs, does the ordering of the
# variables in my system of equations make sense or should I have ordered them
# differently?