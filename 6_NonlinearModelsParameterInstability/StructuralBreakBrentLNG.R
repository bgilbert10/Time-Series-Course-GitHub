# Structural Breaks Examples

require(quantmod)
require(forecast)
require(fBasics)
require(CADFtest)
require(urca)
require(sandwich)
require(lmtest)
require(nlme)
require(car)
require(strucchange)
require(vars)

# Brent
getSymbols("MCOILBRENTEU",src="FRED")
# LNG Asia
getSymbols("PNGASJPUSDM",src="FRED")

# need to make sure they cover the same time span

LNGdata = merge.xts(MCOILBRENTEU,PNGASJPUSDM,join="inner")
plot(LNGdata)

dbrent = ts(na.omit(diff(log(LNGdata$MCOILBRENTEU))),freq=12,start=1992+1/12)
dlng = ts(na.omit(diff(log(LNGdata$PNGASJPUSDM))),freq=12,start=1992+1/12)

brlng = cbind(dbrent,dlng)
ts.plot(dlng)

# Structural breaks. Is there a break in the LNG return series? 
# In the Brent-LNG returns relationship? In the LNG return variance?
# Here's the basic regression
basicreg <- lm(dlng~dbrent)
coeftest(basicreg,vcov=vcovHAC(basicreg))

# estimate and investigate break points in the LNG returns using the breakpoints
# function in the strucchange package. 
# by default searching as many as five break points:
bp_dlng = breakpoints(dlng~1)
# get confidence intervals around timing of breaks:
ci_dlng = confint(bp_dlng)
# optimal zero break points:
summary(bp_dlng)
# see that BIC lowest at zero breaks:
plot(bp_dlng)
# print the intercept coefficients: 
coef(bp_dlng,breaks=0)
# print if we force it to have one break (just for comparison):
coef(bp_dlng,breaks=1)
# get the standard errors on the intercepts using HAC standard errors:
sqrt(sapply(vcov(bp_dlng, breaks = 1), diag))
# compare graphically the time series to the mean with & without breaks 
# (here we forced a single break when we know the optimal number is zero)
fit0 = lm(dlng~1)
fit1 = lm(dlng~0+breakfactor(bp_dlng,breaks=1))
# notice the coefficients of fit1 are the same as above, but 
# standard errors did not use HAC
summary(fit1)
plot(dlng)
lines(ts(fitted(fit0),start=1992,freq=12),col=3)
lines(ts(fitted(fit1),start=1992,freq=12),col=4)
lines(bp_dlng)
lines(ci_lng)


# estimate and investigate break points in the LNG return variance
# use the squared residuals from a model, e.g., 
lngres = residuals(arima(dlng,order=c(2,0,2)))
lngres2 = lngres^2
# by default searching as many as five break points:
bp_vlng = breakpoints(lngres2~1)
# get confidence intervals around timing of breaks:
ci_vlng = confint(bp_vlng)
# optimal one break point:
summary(bp_vlng)
# see that BIC lowest at one break:
plot(bp_vlng)
# print the intercept coefficients: 
coef(bp_vlng,breaks=1)
# compare graphically the time series to the variance with & without breaks 
fit1 = lm(lngres2~0+breakfactor(bp_vlng,breaks=1))
summary(fit1)
plot(lngres2)
lines(ts(fitted(fit1),start=1992,freq=12),col=4)
lines(bp_vlng)
lines(ci_vlng)


# What if the data is not stationary? It's mean is NOT stable.
# DON'T USE THIS BAI & PERRON METHOD 
# It will tell you there are breaks all over the place because a unit root does
# not have a stable mean. See this example. 

# (notice I'm converting to a ts object inside the command, so dates are recognized)
loglng <- ts(log(PNGASJPUSDM),freq=12,start=1992)
bp_lng = breakpoints(loglng~1)
# get confidence intervals around timing of breaks:
ci_lng = confint(bp_lng)
# optimal 4 break points in 1999, 2005, 2010, 2015:
summary(bp_lng)
# see that BIC lowest at 4 breaks
plot(bp_lng)
# print the intercept coefficients: 
coef(bp_lng,breaks=4)
# get the standard errors on the intercepts using HAC standard errors:
sqrt(sapply(vcov(bp_lng, breaks = 4), diag))
# compare graphically the time series to the mean with & without breaks
fit0 = lm(loglng~1)
fit1 = lm(loglng~0+breakfactor(bp_lng,breaks=4))
# notice the coefficients of fit1 are the same, but standard errors did not use HAC
summary(fit1)
plot(loglng)
lines(ts(fitted(fit0),start=1992,freq=12),col=3)
lines(ts(fitted(fit1),start=1992,freq=12),col=4)
lines(bp_lng)
lines(ci_lng)

# How does the LNG price correspond to Brent oil price?
# (with breaks overlayed, but remember they are spurious)
par(mar=c(5,4,4,5)+0.1)
plot(ts(log(LNGdata$MCOILBRENTEU),freq=12,start=1992))
par(new=T)
plot(loglng,axes=FALSE,ylab="",col="red")
mtext("lng",side=4,line=2.5,col="red")
axis(side=4,col="red",col.axis="red")
lines(bp_lng,col="green")
lines(ci_lng,col="blue")


# let's first use the stationary (returns)
# structural break in the Brent-LNG PRICE relationship
bp.lngoil = breakpoints(dlng~dbrent)
ci.lngoil = confint(bp.lngoil,breaks=1)
breakpoints(bp.lngoil)
# optimal number of breaks is zero - stable relationship
# JUST FOR DISCUSSION, impose a single break:
coef(bp.lngoil,breaks=1)
sqrt(sapply(vcov(bp.lngoil, breaks = 1), diag))
plot(bp.lngoil)
# If we DID impose a break, it's in October 2008:
summary(bp.lngoil)
# no breaks:
fm0 = lm(dlng~dbrent)
# Only break in intercept:
fm1 = lm(dlng~0+dbrent+breakfactor(bp.lngoil,breaks=1))
# Break in slope and intercept:
fm2 = lm(dlng~0+breakfactor(bp.lngoil,breaks=1)/(dbrent))
summary(fm1)
summary(fm2)
plot(dlng)
lines(ts(fitted(fm0),start=1992,freq=12),col=2)
lines(ts(fitted(fm1),start=1992,freq=12),col=3)
lines(ts(fitted(fm2),start=1992,freq=12),col=4)
lines(bp.lngoil,breaks=1)
lines(ci.lngoil)

# now let's use the log prices (why?)
# structural break in the Brent-LNG PRICE relationship
logbrent <- ts(log(LNGdata$MCOILBRENTEU),freq=12,start=1992)
# they LOOK cointegrated:
ts.plot(logbrent,loglng)
# but it's unclear whether the residuals of the price equation are stationary:
tsdisplay(residuals(lm(loglng~logbrent)))
# could they be cointegrated after adjusting for breaks?
bp.lngoil = breakpoints(loglng~logbrent)
ci.lngoil = confint(bp.lngoil)
# Two breaks is optimal:
breakpoints(bp.lngoil)
coef(bp.lngoil,breaks=2)
sqrt(sapply(vcov(bp.lngoil, breaks = 2), diag))
plot(bp.lngoil)
summary(bp.lngoil)
fm0 = lm(loglng~logbrent)
# intercept breaks only:
fm1 = lm(loglng~0+logbrent+breakfactor(bp.lngoil,breaks=2))
# breaks in slope and intercept:
fm2 = lm(loglng~0+breakfactor(bp.lngoil,breaks=2)/logbrent)
summary(fm1)
summary(fm2)
plot(loglng)
lines(ts(fitted(fm0),start=1992,freq=12),col=2)
lines(ts(fitted(fm1),start=1992,freq=12),col=3)
lines(ts(fitted(fm2),start=1992,freq=12),col=4)
lines(bp.lngoil,breaks=2)
lines(ci.lngoil)

# residuals are still autocorrelated, but seem stationary. 
tsdisplay(residuals(fm2))


# more on structural breaks, some from strucchange vignettes in package documentation
# pre-packaged data on Nile river flows
data("Nile")
plot(Nile)
bp.nile = breakpoints(Nile~1)
summary(bp.nile)
fm0 = lm(Nile~1)
fm1 = lm(Nile~breakfactor(bp.nile,breaks=1))
plot(Nile)
lines(ts(fitted(fm0),start=1871),col=3)
lines(ts(fitted(fm1),start=1871),col=4)



# Example on the Phillips Curve from the Tsay textbook
data("PhillipsCurve")
uk <- window(PhillipsCurve,start=1948)

bp.inf <- breakpoints(dp~dp1,data=uk, h=8)
plot(bp.inf)
summary(bp.inf)

fac.inf = breakfactor(bp.inf,breaks=2,label="seg")
summary(fac.inf)
fm.inf <- lm(dp~0+fac.inf/dp1,data=uk)
summary(fm.inf)

coef(bp.inf,breaks=2)
sqrt(sapply(vcov(bp.inf, breaks = 2), diag))

# Phillips Curve
bp.pc = breakpoints(dw ~ dp1 + du + u1,data=uk, h=5,breaks=5)
plot(bp.pc)
summary(bp.pc)
coef(bp.pc,breaks=2)
fac.pc = breakfactor(bp.pc,breaks=2,label="seg")
fm.pc = lm(dw~0+fac.pc/dp1 + du+ u1,data=uk)
fm.pc2 = lm(dw~0+fac.pc/dp1+fac.pc/du+fac.pc/u1,data=uk)
fm.pc3 = lm(dw~fac.pc + dp1 +du +u1,data=uk)
summary(fm.pc)
summary(fm.pc2)
summary(fm.pc3)
