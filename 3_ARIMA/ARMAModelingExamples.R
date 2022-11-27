# Preamble ----------------
# Some Example ARMA Modeling

require(forecast)
require(quantmod)
require(caschrono)

# Read and organize data ----------------

# investigate the industrial production index for drilling
getSymbols("IPN213111S",src="FRED")

# FRED does not allow selecting specific dates, so let's pick a fixed window
# so that answers don't change every day

# one way to select start and end: use "paste": 
drill <- na.omit(IPN213111S[paste("1972-04-01","2022-08-03",sep="/")])

# another way: use "window":
startdate <- "1972-04-01"    
enddate   <- "2022-08-03"   
dril2 <- na.omit(window(IPN213111S, start = startdate, end = enddate))
# drill and drill2 are identical

chartSeries(drill, theme="white")
acf(drill)


# let's remove missing observations while calculating differences
drlchng <- data.frame(na.omit(diff(drill)))
ts.plot(drlchng) # nonstationary or breaks in the mean?

# Model fitting and checking -------------------

# Fit optimal AR(p) model (ignoring MA for now)
ar(drlchng,order.max=20) # 15 lags selected
m1 <- arima(drlchng,order=c(15,0,0)) 
m1
# only lags 1, 8, 9, 14 are significant, AIC 3553.5
tsdiag(m1,gof=36)
# important: R's arima() reports the mean as `intercept`
# need to actually calculate the intercept if important. will do below

# Drop insignificant terms ------------

# create a model that drops insignificant coefficients. Keep p=15. 
c1 <- c(NA,0,0,0,0,0,0,NA,NA,0,0,0,0,NA,NA,NA) # last entry is for intercept
m2 <- arima(drlchng,order=c(15,0,0),fixed=c1) # AIC 3554.49 is worse
m2

# but 2, 3, 4, 5, 10, 12, 13 were close. Re-estimate: 
c1 <- c(NA,NA,NA,NA,NA,0,0,NA,NA,NA,0,NA,NA,NA,NA,NA) # last entry is for intercept
m2 <- arima(drlchng,order=c(15,0,0),fixed=c1) # AIC 3548.89 is improvement
m2

tsdiag(m2,gof=36)
tsdisplay(residuals(m2),main='AR Model With Lags 1-5, 8-10, 12-15 only')
Box.test(m2$residuals,lag=20) # uses 20 df, but should have 20-12=8.
pv = 1-pchisq(3.4468,8) # p-value is 0.9 - lower than before.
pv

# Using auto.arima() ------------------

# auto.arima can handle nonstationary data by finding how much differencing is required.
auto.arima(drill) # ARIMA(1,1,1) without drift

m3 <- auto.arima(drlchng) 
m3
# ARIMA(1,0,1) with zero mean, 
# AIC 3553.18 better than AR15, not as good as constrained AR15
tsdisplay(residuals(m3),main='ARIMA (1,0,1) Model Residuals')
# notice there are some autocorrelations
tsdiag(m3,gof=36)

# by default auto.arima take shortcuts in the numerical solvers to estimate many 
# models quickly, does not search over all models. May not give optimal model. 
# Try to get auto.arima to give me the AR15 which I know is better
# Takes a few minutes to run.
auto.arima(drlchng,max.p=15,max.order=100,seasonal=F,stepwise=F,trace=T,approximation=F) 
# looks like ARIMA (4,0,2) with zero mean is better after all
# AIC 3546.11 - even better than constrained AR15
# could investigate "smaller" models if p and q were large
m4 <- arima(drlchng,order=c(4,0,2),include.mean = F)
m4
tsdiag(m4,gof=36)
tsdisplay(residuals(m4))

# Which models pass all the tests? Could some be "overfitting"?
# Other models have some serial correlations in the residuals?

# Forecasting from fitted ARIMA ---------------

mean(drlchng$IPN213111S)
predict(m4,50) # Prediction
m4p = predict(m4,50)
names(m4p)
lcl = m4p$pred-1.96*m4p$se  # calculate lower 95% interval
ucl = m4p$pred+1.96*m4p$se # calculate upper 95% interval
cl <- cbind(lcl,m4p$pred,ucl)
print(cl)

fcast <- forecast(m4,h=50)
plot(fcast,include=50)

# forecast with a holdout sample from constrained AR15 to see how well we did
drlchng <- as.ts(drlchng)
fit_no_holds <- arima(drlchng[-c(555:604)],order=c(15,0,0),fixed=c1)
fcast_no_holds <- forecast(fit_no_holds,h=50)
plot(fcast_no_holds,main=" ",include=50)
lines(as.ts(drlchng))

# forecast with a holdout sample from ARMA(4,2) to see how well we did
fit_no_holds <- arima(drlchng[-c(555:604)],order=c(4,0,2))
fcast_no_holds <- forecast(fit_no_holds,h=50)
plot(fcast_no_holds,main=" ",include=50)
lines(as.ts(drlchng))
plot(fcast_no_holds,main=" ")
lines(as.ts(drlchng))

# Evaluating roots ---------------

# If there are complex conjugate pairs, there is a cyclical component
# Calculate the average length of stochastic cycles

# try it with our ARMA(4,2) model
cp = c(1,-m4$coef[1:4])
roots = polyroot(cp)
roots
# 1 complex conjugate pair, 1 cycle
# a +/- b*i
# Modulus is sqrt(a^2 + b^2)
Mod(roots)
sqrt((-0.135840)^2+(1.012397)^2)
# average cycle length is 2*pi/acos(a/mod(a,b))
k1 = 2*pi/acos(-0.135840/1.02147)
k1
# about 3 or 4 month cycle

# try with our constrained AR15, model 2
cp = c(1,-m2$coef[1:15])
roots = polyroot(cp)
roots
# 7 complex conjugate pairs

# old
# 15 roots, 14 are complex, 7 complex conjugate pairs, 7 cycles
Mod(roots)
k1 = 2*pi/acos(0.9221662/1.310143)
k2 = 2*pi/acos(-0.7374550/1.144145)
k3 = 2*pi/acos(-0.2075883/1.108865)
k4 = 2*pi/acos(1.1497949/1.169493)
k5 = 2*pi/acos(0.3089135/1.194408)
k6 = 2*pi/acos(-1.0598384/1.188696)
k7 = 2*pi/acos(1.1272146/1.392528)
# 2 to 4 month cycle, 8 month, 10 month, and 35 months 


# Additional example: unemployment ------------------

### Example with unemployment rates
getSymbols("UNRATE",src="FRED")
head(UNRATE)
dim(UNRATE)
# convert the 'xts' formatting to numeric, taking the first column of data from UNRATE
startdate <- "1948-01-01"    
enddate   <- "2020-08-01"   
rate <- na.omit(window(UNRATE, start = startdate, end = enddate))
rate <- as.numeric(rate[,1])

# some detective work on the temporal behavior
ts.plot(rate)
acf(rate)
pacf(rate)
# the unemployment rate doesn't strictly conform to our expected pattern of 
# stationary/nonstationary (acf declining, but not exponentially)
# However, pacf has one lag close to one and rest close to zero. 
# Treat it as nonstationary for now
# calculate the first difference (monthly change in unemployment) to get 
# a stationary series
ratechng = na.omit(diff(rate))
ts.plot(ratechng)

# detective work on ARMA orders
acf(ratechng)
pacf(ratechng)
# AR(4), ARMA(0,1), or ARMA(1,1) 
ar(ratechng)
armaselect(ratechng)
auto.arima(ratechng)

c4 <- c(0,NA,0,NA)
ar4 <- arima(ratechng,order = c(4,0,0), include.mean=F, fixed = c4)
ar4
tsdiag(ar4)
tsdisplay(residuals(ar4))
# Check Box test

arma01 <- arima(ratechng,order = c(0,0,1))
arma01
# Higher AIC than ar4
tsdiag(arma01)
tsdisplay(residuals(arma01))

arma11 <- arima(ratechng,order = c(1,0,1))
arma11
# Higher AIC than ar4
tsdiag(arma11)
tsdisplay(residuals(arma11))


# Stick with ar4. 
#### We can also look at the roots of the lag polynomial 
lagpol_ar4 = c(1,-ar4$coef[1:4])
roots = polyroot(lagpol_ar4)
roots
Mod(roots)
# With only two coefficients, 2 unique roots, 1 complex conjugate pair, 1 cycle.
# All greater than one in modulus (characteristic roots are less than 1)
k1 = 2*pi/acos(1.200711/1.801056)
k1  ## 7.5 months

arima(rate,order=c(4,1,0))
arima4 = arima(rate,order=c(4,1,0),fixed=c4)
lagpol_arima4 = c(1,-arima4$coef[1:4])
roots = polyroot(lagpol_arima4)
roots
Mod(roots)
# Identical to result above
k1 = 2*pi/acos(1.200711/1.801056)
k1  ## 7.5 months

fcast <- forecast(arima4,h=48)
plot(fcast,include=48)

# forecast with a holdout sample to see how well we did
rate <- as.ts(rate)
fit_no_holds <- arima(rate[-c(861:872)],order=c(4,1,0),fixed=c4)
fcast_no_holds <- forecast(fit_no_holds,h=12)
plot(fcast_no_holds,main=" ",include=36)
lines(as.ts(rate))

