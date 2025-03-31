# simulate a couple of AR(2) processes and look at characteristic polynomials
# one with real roots, one with complex roots

# compare AR1 process to MA1 process to White Noise process

# install.packages("forecast")
require(forecast)
# install.packages("quantmod")
require(quantmod)

# simulate an AR(2) with imaginary roots and 
# one with real roots
y1 <- arima.sim(model=list(ar=c(1.3,-0.4)),1000)
# simulates 1000 obs with ph1 =1.3, ph2 = -0.4
acf(y1)
y2 <- arima.sim(model=list(ar=c(0.8,-0.7)),1000)
acf(y2)
# ACF for y1 shows exponential decay, but 
# ACF for y2 shows dampening sine and cosine
plot(y1)
plot(y2)

# calculate the roots
p1 <- c(1,-1.3,0.4) # coefficients of characteristic polynomial
r1 <- polyroot(p1) # calculate roots
r1   # y1 had two real roots
p2 <- c(1,-0.8,0.7)
r2 <- polyroot(p2)
r2   # y2 had a conjugate pair of complex roots

# a = 0.571429, b=1.049781

# since y2 has cyclical behavior,
# let's investigate the average length of the cycle
# two ways to do it
k <- 2*pi/acos(0.571429/sqrt(0.571429^2 + 1.049781^2))

# calculate modulus (absolute value) of the roots
Mod(r2)
# sqrt(a^2 + b^2) = 1.195229
k <- 2*pi/acos(0.571429/1.195229)

# Be aware that the coefficients are all we need to capture
# the cyclical dynamics in our forecast - don't need to 
# work with sine and cosine functions

# suppose I didn't know the true coefficients
m1 <- ar(y1,method="mle",order.max=10) # choose lag order with AIC criterion
m1$order
m1
# notice the AR2 coefficients are very close
pacf(y1)  # compute PACF
t.test(y1)  # testing mean = 0.
Box.test(y1,lag=10,type='Ljung')  # Perform Ljung-Box Q-test.

m2 <- ar(y2,method="mle",order.max=10) 
m2$order
m2
pacf(y2)  # compute PACF
t.test(y2)  # testing mean = 0.
Box.test(y2,lag=10,type='Ljung')  # Perform Ljung-Box Q-test.

# in order to use "forecast" we have to use the model we fit
# and use that model to forecast (not forecast on raw data)
fy2 = forecast(m2,h=100)
plot(fy2)
# you can see the forecast includes the cycles, but let's 
# zoom in by including only the first 100 observations of
# the original series: 
plot(fy2,include=100)
# Important: the long run forecast is just the mean, mu

# y1 is less interesting
fy1 = forecast(m1,h=100)
plot(fy1)
plot(fy1,include=100)



# Let's compare what an MA(1) looks like compared to an AR(1) with same
# coefficient, compared to white noise with same mu and sigma

ar1 <- arima.sim(model=list(ar=c(0.8)),200)
ma1 <- arima.sim(model=list(ma=c(0.8)),200)
wn <- arima.sim(model=list(),200)
plot(ar1)
plot(ma1)
plot(wn)
# notice AR drifts away from mean and drifts back
# MA and WN have same mean reversion, but MA has wider variance.
# WN 95% of observations between 2 and -2 (two standard devations from mean=0)
# MA has more observations outside (-2,2)

# notice that depending on the draw of random numbers,
# auto.arima might think our ar1 is non-stationary because 0.8 close to 1
# if we were to add more observations, or do this multiple times, 
# this would go away
fitAR1 = auto.arima(ar1)
fitAR1
# using ar() command get very close to simulated model
# but can't get MA or I components
fitAR1 = ar(ar1)
fitAR1
fcAR1 = forecast(fitAR1,h=200)
plot(fcAR1)
plot(fcAR1,include=100)

fitMA1 = auto.arima(ma1,approximation=F)
# should get MA1
# could try impose MA1 and zero mean with 
fitMA1 = arima(x=ma1,order=c(0,0,1),include.mean=F)
# with only 200 obs, coefficient is pretty far from correct
# try with 2000 obs. 
# in practice, we don't know

# compare 5-step ahead forecast with MA terms
# vs. same forecast without
fcMA1 = forecast(fitMA1,h=5)
plot(fcMA1)
plot(fcMA1,include = 50)
# notice difference in uncertainty if ignoring MA1
fcMA.naive = forecast(ma1,h=5)
plot(fcMA.naive,include = 50)
# forecast converges to sample mean and variance quickly
# but MA1 forecast allows more precision in short term

# compare to AR(3) or MA(3), see how forecast tapers off.
ar3 <- arima.sim(model=list(ar=c(0.8,-0.3,0.3)),2000)
ma3 <- arima.sim(model=list(ma=c(0.8,-0.3,0.3)),2000)

fitAR3 = ar(ar3)
fitAR3
fcAR3 = forecast(fitAR3,h=200)
plot(fcAR3,include=100)

fitMA3 = auto.arima(ma3,approximation=F)
fitMA3
# might not get MA3
# could try impose MA3 and zero mean with 
fitMA3 = arima(x=ma3,order=c(0,0,3),include.mean=F)
fitMA3
# in practice we don't know

# see 10-step ahead forecast with MA terms
fcMA3 = forecast(fitMA3,h=10)
plot(fcMA3,include = 50)
  