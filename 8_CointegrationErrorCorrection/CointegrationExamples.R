# Cointegration Examples: testing, estimating, interpreting

# Example 1: Brent crude oil price and Asian LNG price
# Example 2: drilling, oil prices, natural gas prices
# Example 3: fossil fuel prices: crude oil, propane, gasoline, jet fuel


###### Packages we will use ########
require(quantmod)
require(forecast)
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

##### Read in and organize data #########

# Brent
getSymbols("MCOILBRENTEU",src="FRED")
# LNG Asia
getSymbols("PNGASJPUSDM",src="FRED")

# WTI
getSymbols("MCOILWTICO",src="FRED")
# Oil & gas drilling index
getSymbols("IPN213111N",src="FRED")
# Henry Hub
getSymbols("MHHNGSP",src="FRED")

# Propane
getSymbols("MPROPANEMBTX",src="FRED")
# Gasoline, all formulations
getSymbols("GASREGM",src="FRED")
# Jet fuel
getSymbols("MJFUELUSGULF",src="FRED")

# Steps -------------------
# 1) Merge data, plot, and decide whether or not to log the data
# 2) Test each variable for a unit root and ARIMA behavior (we will skip this for now)
# 3) Test for cointegration among nonstationary variables
#   a) Philips-Ouliaris test works well for few variables (2 or 3)
#   b) Johansen's algorithm works better for more variables (3 or more)
#     They are different tests and might not always agree. Use your judgment. 
# 4) For Johansen's algorithm and for estimating VECM:
#   a) Select lag length for the VAR in levels (not differenced)
#   b) decide whether to include a trend, constant, both, or neither
#   c) Use decisions from (a) and (b) in ca.jo()
#   d) investigate outputs to choose number of cointegrating vectors
# 5) Analyse the model - what can we do with the estimated VECM?
#   a) investigate coefficients (cajools and cajorls)
#   b) cointegrating vector parameters, return time coefficients, VAR coefficients
#   c) evaluate residuals of cointegrating vectors and of VECM. 
#   d) transform into level-VAR and calculate IRFs, forecasts
#   e) impose structural restrictions like SVAR, but on short run and long run behavior

# Example 1: Brent crude and Asian LNG -----------
oillng= merge.xts(MCOILBRENTEU,PNGASJPUSDM,join="inner")
colnames(oillng) <- c("brnt","lng")
#SELECT START AND END DATE
startdate <- "1997-01-01"    
enddate   <- "2019-12-01"    
oillng <- window(oillng, start = startdate, end = enddate)
oillng <- na.omit(oillng)
plot(oillng)

# They look cointegrated. Looking at individual series, logs seem
# to have a more linear/less exponential trend. For example:
MTSplot(oillng)
log.oil.lng = log(oillng)
MTSplot(log.oil.lng)
plot(log.oil.lng)
# regression residuals look stationary until about 2012
ts.plot(ts(residuals(lm(lng~brnt,data=log.oil.lng)), freq=12, start=1997))

##### Cointegration testing ####

# Testing cointegration in the levels of the three series (log prices)

# Phillips-Ouliaris test:
xcl = ca.po(log.oil.lng)
summary(xcl)
# We fail to reject the null of a unit root in the residuals
# Conclude cointegration does not exist

# Johansen's test simultaneously estimates the VAR/VECM
## How many lags should there be in the VAR?
## If cointegrated, VAR in levels is okay
## Looks like 2 lags of VAR in levels is preferred by all criteria:
VARselect(log.oil.lng,lag.max=20,type="both")

# Johansen test with K=2  lags
xjl = ca.jo(log.oil.lng,type="trace",ecdet="const",K=2,spec="transitory")
summary(xjl)
# seems like 0 cointegrating vectors
# How do we know: 
#     r = 0 does not reject the null 17.55 < 19.96 critical value. Stop at 0.

# Test for break and try pre/post break
# could they be cointegrated after adjusting for breaks?
log.oil.lng <- ts(log.oil.lng, freq=12, start=1997)
bp.lngoil = breakpoints(lng~brnt,data=log.oil.lng)
# Two breaks is optimal:
breakpoints(bp.lngoil)
plot(bp.lngoil)
summary(bp.lngoil)

fm0 = lm(lng~brnt,data=log.oil.lng)
# breaks in slope and intercept:
fm1 = lm(lng~0+breakfactor(bp.lngoil,breaks=1)/brnt,data=log.oil.lng)
fm2 = lm(lng~0+breakfactor(bp.lngoil,breaks=2)/brnt,data=log.oil.lng)
summary(fm0)
summary(fm1)
summary(fm2)
plot(log.oil.lng[,2])
lines(ts(fitted(fm0),start=1997,freq=12),col=2)
lines(ts(fitted(fm1),start=1997,freq=12),col=3)
lines(ts(fitted(fm2),start=1997,freq=12),col=4)
lines(bp.lngoil,breaks=2)

# residuals are still autocorrelated, but seem stationary. 
tsdisplay(residuals(fm1))
tsdisplay(residuals(fm2))

# Could test for pre-break cointegration
startdate <- "1997-01-01"    
enddate   <- "2011-04-01"    
early <- window(oillng, start = startdate, end = enddate)
early <- na.omit(log(early))
plot(early)

ts.plot(ts(residuals(lm(lng~brnt,data=early)), freq=12, start=1997))

# Phillips-Ouliaris test:
xcl.e = ca.po(early)
summary(xcl.e)
# We fail to reject the null of a unit root in the residuals
# Conclude cointegration does not exist

# Johansen's test simultaneously estimates the VAR/VECM
## How many lags should there be in the VAR?
## If cointegrated, VAR in levels is okay
## Looks like between 1 and 5 lags of VAR in levels:
VARselect(early,lag.max=20,type="both")

# Johansen test with K=3  lags
xjl.e = ca.jo(early,type="trace",ecdet="trend",K=3,spec="transitory")
summary(xjl.e)
# seems like 1 cointegrating vector
# How do we know: 
#     r = 0 rejects the null 31.53 < 25.32 critical value. 
#     r = 1 fails to reject. 


##### Parameters of Cointegrated Model ####

# coefficients of the Vector Error Correction Model (VECM)
cajools(xjl.e)
# These coefficients are not arranged into cointegrating vectors, etc.
# Just the raw coefficients on regressors. 
# Notice that there are two lagged difference (dl1, dl2) of each variable. 
# The VAR in levels had 3 lags, differencing leaves two lags. 
# There is also one lagged level of each (l1) - the error correction term
cajorls(xjl.e)
# now I get the parameters of the error correction model and cointegrating 
# vector separately. $beta is the cointegrating vector. 
# ect1 is the cointegrating residual - it has it's return time coefficient
# on ect1 in each of the VECM equations. 

# Examine residuals and residual autocorrelation 
# from each equation in the VECM:
plotres(xjl.e)

### And ect1 should be stationary
ts.plot(cajorls(xjl.e)$rlm$model$ect1)

# Transform/rotate VECM to its level-var representation in order 
# to use all the var() tools. Note r=1 because 1 cointegrating vector
x = vec2var(xjl.e,r=1)

# Forecast levels
xp = predict(x,n.ahead=24)
plot(xp)
# Notice that impulse responses exhibit permanent effects
plot(irf(x,n.ahead=24))

# Looks like a one-standard deviation permanent increase in Brent
# caused a 6 percent permanent increase in the lng price
plot(irf(x,n.ahead=24,impulse = "brnt", response = "lng"))


####### Example 2: drilling, oil price, natural gas price ####

## merge oil, gas, and drilling and calculate returns/changes ----------
oilgas= merge.xts(IPN213111N,MCOILWTICO,MHHNGSP,join="inner")
colnames(oilgas) <- c("drl","oil","gas")
#SELECT START AND END DATE
startdate <- "1997-01-01"    
enddate   <- "2019-12-01"    
oilgas <- window(oilgas, start = startdate, end = enddate)
oilgas <- na.omit(oilgas)
plot(oilgas)
# calculate log levels as ts() objects, notice start dates
loil = ts(na.omit((log(oilgas$oil))),freq=12,start=1997)
lgas = ts(na.omit((log(oilgas$gas))),freq=12,start=1997)
well = ts(na.omit((oilgas$drl)),freq=12,start=1997)
logw = cbind(well,loil,lgas)

# visual inspection suggests cointegration may exist
plot(oilgas)
MTSplot(logw)
# not clear if these residuals are stationary
ts.plot(residuals(lm(well~loil+lgas,data=logw)))


##### Cointegration testing ####

# Testing cointegration in the levels of the three series (log prices)

# Phillips-Ouliaris test:
xcl = ca.po(logw)
summary(xcl)
# We reject the null of a unit root in the residuals
# Conclude cointegration exists

# Johansen's test simultaneously estimates the VAR/VECM
## How many lags should there be in the VAR?
## If cointegrated, VAR in levels is okay
## Looks like 2 lags of VAR in levels is preferred by all criteria:
VARselect(logw,lag.max=20,type="both")

# Johansen test with K=2  lags
xjl = ca.jo(logw,type="trace",ecdet="const",K=2,spec="transitory")
summary(xjl)
# seems like 1 cointegrating vector
# How do we know: 
#     r = 0 rejects the null 46.94 > 34.91 critical value. 
#     r<=1 does not, 12.11 < 19.96 critical value. Stop at 1. 


##### Parameters of Cointegrated Model ####

# coefficients of the Vector Error Correction Model (VECM)
cajools(xjl)
# These coefficients are not arranged into cointegrating vectors, etc.
# Just the raw coefficients on regressors. 
# Notice that there is one lagged difference (dl1) of each variable. 
# The VAR in levels had 2 lags, differencing leaves one lag. 
# There is also one lagged level of each (l1) - the error correction term
cajorls(xjl)
# now I get the parameters of the error correction model and cointegrating 
# vector separately. $beta is the cointegrating vector. 
# ect1 is the cointegrating residual - it has it's return time coefficient
# on ect1 in each of the VECM equations. 

# Examine residuals and residual autocorrelation 
# from each equation in the VECM:
plotres(xjl)

### And ect1 should be stationary
ts.plot(cajorls(xjl)$rlm$model$ect1)

# Transform/rotate VECM to its level-var representation in order 
# to use all the var() tools. Note r=1 because 1 cointegrating vector
x = vec2var(xjl,r=1)

# Forecast levels
xp = predict(x,n.ahead=24)
plot(xp)
# Notice that impulse responses exhibit permanent effects
plot(irf(x,n.ahead=24))

# Looks like a one-standard deviation permanent increase in drilling
# causes a 6 percent permanent decline in the oil price
plot(irf(x,n.ahead=24,impulse = "well", response = "loil"))


####### Example 3: Four fossil fuel prices ###########

##### merge together data and plot ####
fossil = merge.xts(GASREGM,MJFUELUSGULF,MPROPANEMBTX,MCOILWTICO,join="inner")
#SELECT START AND END DATE
startdate <- "1992-06-01"    
enddate   <- "2019-09-01"    
fossil <- window(fossil, start = startdate, end = enddate)
fossil <- na.omit(fossil)

# They look cointegrated. Looking at individual series, logs seem
# to have a more linear/less exponential trend. For example:
MTSplot(fossil)
lfossil = log(fossil)
MTSplot(lfossil)
plot(lfossil)
# regression residuals look stationary
ts.plot(residuals(lm(GASREGM~MCOILWTICO+MPROPANEMBTX+MJFUELUSGULF,data=lfossil)))

##### cointegration test ####

# For ca.jo, options are "none", "const", "trend". 
# they all share the same trend, so I don't need to add one to partial it out. 
xt = ca.jo(lfossil,ecdet="const")
summary(xt)
# Reject the null of 1 or fewer cointegrating vectors, 
# because 56.73 > 22.00, the 5% critical value.
# Do not reject null of 2 or fewer cointegrating vectors,
# because 11.10 < 15.67, the 5% critical value.
# Conclude two cointegrating vectors.

# same conclusion on cointegration with a trend, 
# but maybe 3 cointegrating vectors.
xo = ca.jo(lfossil,ecdet="trend")
summary(xo)

## What if I control for lags.
## First decide lag order. If cointegrated, VAR in levels is okay
## No clear agreement. use 3 or 4 lags
VARselect(lfossil,lag.max=20,type="both")

## Johansen test in logs, go with 3 lags
## Evidence again of 2 cointegrating vectors when no trend is added
ca.lfossil = ca.jo(lfossil,type="trace",ecdet="const",K=3,spec="transitory")
summary(ca.lfossil)

## Report parameters. "r=2" refers to two cointegrating vectors. 
ca.lfossil2 = cajorls(ca.lfossil,r=2)
ca.lfossil2
## cointegrating vector
ca.lfossil2$beta
## AR coefficients, and return time parameters are ect1 and ect2 in:
ca.lfossil2$rlm

### For cointegrating vectors, we have 
### log.gas = -2.69 - 0.234*log.prop + 0.865*log.oil - 0.000*log.jet + ect1
### log.jet = -4.34 - 0.164*log.prop + 1.193*log.oil - 0.00000000000000011*log.gas + ect2

### a positive shock to ect1 means gas is abnormally high relative to oil and propane.
### a positive shock to ect2 means jet fuel is abnormally high relative to oil and propane

### A 100% shock to ect1 (abnormally high gas) implies gas will return 22.2% 
# of the way to equilibrium each period (i.e., fall by 22.2% all else equal)
### A 100% shock to ect2 (abnormally high jet fuel) implies that gas will 
# increase by 10.7% the following period.

# and ect1 and ect2 should be stationary
ts.plot(ca.lfossil2$rlm$model$ect1)
ts.plot(ca.lfossil2$rlm$model$ect2)

## Translate into a VAR and show forecasts and IRFs
vecfossil = vec2var(ca.lfossil,r=2)
plot(predict(vecfossil,n.ahead=48))
plot(irf(vecfossil,n.ahead=24,ortho=T))
plot(irf(vecfossil,n.ahead=24,ortho=T,impulse = "MCOILWTICO",response="GASREGM"))
### A one-time, permanent one-standard deviation increase in the log oil price
### (in other words a one time permanent 8 percent increase) sd(na.omit(diff(lfossil$MCOILWTICO)))
### predicts about a 2 percent long-run increase in the gasoline price.


# Advanced stuff ----------------

## Return to US oil, gas, drilling example -------------

# if we think the system has prices determining wells, normalize cointegrating
# vector relative to wells
vecmdgo = ca.jo(cbind(well,lgas,loil),type="trace",ecdet="trend",K=2,spec="transitory")
summary(vecmdgo)
vecmdgo2 = cajorls(vecmdgo,r=1)
vecmdgo2
vecmdgol = vec2var(vecmdgo,r=1)
plot(predict(vecmdgol,n.ahead=48))
plot(irf(vecmdgol,n.ahead=24,ortho=T))
plot(irf(vecmdgol,n.ahead=24,ortho=T,impulse = "loil",response="well"))
plot(irf(vecmdgol,n.ahead=24,ortho=T,impulse = "well",response="lgas"))
plot(fevd(vecmdgol,n.ahead=24))

# estimate the structural components of SVECM
# need 0.5K(K-1) = 3 restrictions on B matrix (K=3 equations)
# have k* = r(K-r) = 2 permanent shocks and 1 transitory
# implies set last column to zeros, modify number of restrictions
# so really 0.5k*(k*-1) = 1 restriction on remaining columns
# one idea: wells is transitory (perhaps not true in this example - gas seems not to matter)
# assume oil moves independently - gas only driven by oil, nothing drives oil
# implies set row 1, column 2 = 0 also in long run matrix
# recall that structural residual matrix is decomposed into short run and long run effects.
# set long run impacts matrix
vecmgod = ca.jo(cbind(loil,lgas,well),type="trace",ecdet="trend",K=2,spec="transitory")
BL = matrix(c(NA,NA,NA,0,NA,NA,0,0,0),nrow=3,ncol=3)
BS = matrix(NA,nrow=3,ncol=3)
BS[1,2] = 0
svecmgod = SVEC(vecmgod,r=1,LR=BL,SR=BS,boot=T,runs=200,max.iter=400)
summary(svecmgod)
svec.god = irf(svecmgod,n.ahead=24,boot=T)
svec.godw = irf(svecmgod,response="well",n.ahead=24,boot=T)
plot(svec.god)
plot(svec.godw)
svec.fegod = fevd(svecmgod,n.ahead=24)
plot(svec.fegod)

# Another idea: gas price variation is transitory. Drilling and oil shocks 
# are permanent - wells produce for a long time. But suppose they don't affect 
# oil price in the long run. 
# No short run effect of oil price on gas price 
vecmgod = ca.jo(logw,type="trace",ecdet="trend",K=2,spec="transitory")
BL = matrix(c(NA,0,NA,NA,NA,NA,0,0,0),nrow=3,ncol=3)
BS = matrix(NA,nrow=3,ncol=3)
BS[3,2] = 0
svecmgod = SVEC(vecmgod,r=1,LR=BL,SR=BS,boot=T,runs=200,max.iter=400)
summary(svecmgod)
svec.god = irf(svecmgod,n.ahead=24,boot=T)
svec.godw = irf(svecmgod,response="well",n.ahead=24,boot=T)
plot(svec.god)
plot(svec.godw)
plot(irf(svecmgod,response="lgas",n.ahead=24,boot=T))
svec.fegod = fevd(svecmgod,n.ahead=24)
plot(svec.fegod)