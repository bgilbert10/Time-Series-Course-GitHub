######################
# This is a program to practice some examples of unit root testing
#   Helps give some guidance on which Cases to use when you're not sure, and
#   What are the consequences of using the wrong case. 
#   Bottom line: Constraining coefficients to zero when we should not (true model includes them) 
#                is much worse than including extra terms that might be irrelevant. 

# Here are the packages we will use (install.packages if you don't have them):
require(quantmod)
require(fBasics)
require(tseries)
require(CADFtest)
require(urca)
require(forecast)
require(lubridate)

# First we will simulate some fake data where we know the answers, to see how well we do
# set.seed() lets me simulate the same data every time so I can compare results
set.seed(206)

# Simple example: Persistent AR(1) and AR(2) with zero mean vs Random Walk without drift
# The truth for each of these simulated variables is Case 1 (type = "none"). 
# What happens if we accidentally test using Case 2 (type = "drift")?
# We force the ADF test to include an intercept. Is that a problem? Let's find out
y1AR1 <- arima.sim(model=list(order=c(1,0,0),ar=c(0.85)),n=1000)
ts.plot(y1AR1)

y1AR2 <- arima.sim(model=list(order=c(2,0,0),ar=c(1.2,-0.3)),n=1000)
ts.plot(y1AR2)

y1RW <- arima.sim(model=list(order=c(0,1,0)),n=1000)
ts.plot(y1RW)

# Suppose I want to test each of these series for a unit root.
# I can use Case 1 (type="none") because I KNOW there is no intercept
# Suppose I don't know the correct number of lags
# I can the CADFtest package to get the right lag length of the augmented dickey fuller test
# I can't, however, test joint hypotheses using CADFtest package.

# What happens if I run the AR1 with 0.95? 0.999?
# What happens if I use type=drift?
cadfy1AR1 = CADFtest(y1AR1,criterion=c("AIC"),type = "none")
summary(cadfy1AR1)
# REJECT the null - NOT a unit root (we knew that)
# Coefficient on lag is -0.13
# Original AR(1) simulated: p(t) = 0.85*p(t-1) + a(t)
# Estimated ADF:        (p(t)-p(t-1)) = -0.13*p(t-1) 
# Rearranging the estimated one: p(t) = 0.87*p(t-1) 
# What happens if I run the AR1 with the lag coefficent = 0.95? 0.999? Try it.

cadfy1AR2 = CADFtest(y1AR2,criterion=c("AIC"),type = "none")
summary(cadfy1AR2)
# REJECT the null - NOT a unit root (we knew that)
# Aside: 
# Notice in the AR2 I get the coefficients back if I unwind the test
# ADF test regresses a first difference on a lagged level and possibly lagged first differences
# It is a good exercise to do the algebra to rewrite that regression in levels
# You will see the estimated coefficients from the regression give me my AR(2) back
# Original AR(2) simulated: p(t) = 1.2*p(t-1) - 0.3*p(t-2) + a(t)
# Estimated ADF:        (p(t)-p(t-1)) = -0.09*p(t-1) + 0.33*(p(t-1)-p(t-2))
# Rearranging the estimated one: p(t) = 1.24*p(t-1) - 0.33*p(t-2) 

cadfy1RW = CADFtest(y1RW,criterion=c("AIC"),type = "none")
summary(cadfy1RW)
# FAIL to reject the null - a UNIT ROOT (we knew that)

# I could use the ur.df if I want the joint tests
# CADFtest told me how many lags to include (even though I already knew since I created the data)
# In Case 1, type = "none", there are no joint tests. 
# test statistics for ADF are at the bottom of the output
ury1AR1 <- ur.df(y1AR1,type=c("none"),lags=0)
summary(ury1AR1)
ury1AR2 <- ur.df(y1AR2,type=c("none"),lags=1)
summary(ury1AR2)
ury1RW <- ur.df(y1RW,type=c("none"),lags=0)
summary(ury1RW)

# Suppose I didn't know that these were all mean zero processes, and didn't know which ones had random walks.
# What happens if I use Case 2 instead?
# Answer: there is no problem!
# You SHOULD be doing this with real data any time you think there is no growth/trend/drift.
# using type = "drift" (again, when we believe there is no real drift)

# On the stationary AR(1) first
ury1AR1 <- ur.df(y1AR1,type=c("drift"),lags=0)
summary(ury1AR1)
# Notice I get TWO test statistics at the bottom: 
# tau2: individual t-test of lag (Is there a unit root?)
# phi1: joint F-test of lag AND intercept. 
# I reject both - phi1 tells me EITHER the lag is not zero, OR intercept not zero, or BOTH
# How can I tell if the intercept is significant by itself?
# Under the null, the data has a unit root, so the critical values are strange.
# Under the alternative, the data is stationary and the critical values are normal.
# If I reject the null of a unit root, I can interpret the t-stats of the regression output as I normally would.
# So is the intercept individually significant?
# alternatively, I could estimate the (stationary) AR1 and test the intercept: 
arima(y1AR1,order=c(1,0,0),include.mean = TRUE)

# I get a similar set of results with the stationary AR(2)
ury1AR2 <- ur.df(y1AR2,type=c("drift"),lags=1)
summary(ury1AR2)

# Now try with the Random Walk with no drift. 
# I get a different issue here when I fail to reject, and have a unit root
ury1RW <- ur.df(y1RW,type=c("drift"),lags=0)
summary(ury1RW)
# I fail to reject the null for the individual test of the unit root (-0.7839 less than tau2 critical values)
# I fail to reject the null for the joint test of the lagged level and intercept (0.6668 less than phi1 critical values)
# This is a random walk without drift (I already knew that)
# What if I want to know if the intercept is INDIVIDUALLY significant?
# With a unit root, the t-stats are not normal - can't do a t-test on the intercept from the test output
# HOWEVER, first differences are stationary
# difference the data (stationary), and perform a t-test on the intercept
arima(diff(y1RW),order=c(0,0,0),include.mean = TRUE)
Arima(y1RW,order=c(0,1,0),include.constant = TRUE)


##############
# Next example: The truth is Case 2. 
# Simulate persistence AR(1) and AR(2) with NON-zero mean vs Random Walk WITHOUT drift
y2AR1 <- arima.sim(model=list(order=c(1,0,0),ar=c(0.85)),n=1000) + 10
ts.plot(y2AR1)

y2AR2 <- arima.sim(model=list(order=c(2,0,0),ar=c(1.2,-0.3)),n=1000)+10
ts.plot(y2AR2)

y2RW <- arima.sim(model=list(order=c(0,1,0)),n=1000)
ts.plot(y1RW)
# The truth is a random walk without drift. My eyes suggest a possible drift. I need to be careful!

# Let's do a similar set of investigations
# I can use CADFtest to find the right lag length of the augmented dickey fuller test

# What happens if I apply Case 1 (type=none) when I should not?
# Unlike before, when I included an intercept that was no different from zero,
# here I FORCE the model to have no intercept when it should by using Case 1. This is much worse!
cadfy2AR1 = CADFtest(y2AR1,criterion=c("AIC"),type = "none")
summary(cadfy2AR1)
summary(cadfy1AR1)
# Comparing this output to the previous example, 
# I fail to reject the null when I should reject it!
# I might think this is a random walk, when it is not!
# Also, the R-squared is lower, and I get the wrong coefficients

# Using the "drift" option (Case 2) gives us the right answer when Case 2 is correct
# AND it DOESN'T Give the WRONG answer when Case 1 was correct.
cadfy2AR1 = CADFtest(y2AR1,criterion=c("AIC"),type = "drift")
summary(cadfy2AR1)
# Suppose I mistakenly use the "trend" option (Case 4) even though Case 2 is correct:
cadfy2AR1 = CADFtest(y2AR1,criterion=c("AIC"),type = "trend")
summary(cadfy2AR1)
# No harm! I just included an irrelevant trend term that is not significant
# I reject the null of a unit root, so the data is stationary and coefficients are normally distributed
# I can use t-test on the trend, if I reject the null and believe data is stationary. 
# trend is not statistically significant. 
# This tells me to go back and use Case 2.

# Now let's look at the stationary AR2
# Incorrectly using type = "none"
cadfy2AR2 = CADFtest(y2AR2,criterion=c("AIC"),type = "none")
summary(cadfy2AR2)
summary(cadfy1AR2)
# LUCKILY reject null when I should (doesn't happen in all simulations - no guarantee), 
# but the coefficients are way off, fit (R-squared) is much worse

# Let's correctly use Case 2 (type=drift) to see the right answer
cadfy2AR2 = CADFtest(y2AR2,criterion=c("AIC"),type = "drift")
summary(cadfy2AR2)
# strongly reject the null
# mean = intercept/(1-phi1-phi2) is close to 10, the true mean

# Let's INCORRECTLY use Case 4 (type=trend)
cadfy2AR2 = CADFtest(y2AR2,criterion=c("AIC"),type = "trend")
summary(cadfy2AR2)
# we reject (and we should), so coefficients are normally distributed
# can use t-test on the trend term under the alternative
# The trend term is not statistically significant, which we know from generating the data.

# I can use ur.df to perform the joint tests
# Let's compare drift and trend options (recall Case 2 (drift) is the right one for this data)
ury2AR1 <- ur.df(y2AR1,type=c("drift"),lags=0)
summary(ury2AR1)
# strongly reject joint hypothesis of zero lag and zero intercept
# Good! The data we generated had non-zero mean and no unit root
ury2AR1t <- ur.df(y2AR1,type=c("trend"),lags=0)
summary(ury2AR1t)
# phi2 is joint test of lag, intercept and trend
# phi3 is joint test of lag and trend
# strongly reject all of these hypotheses (look at the bottom of the output): 
#     "Value of test-statistic is: tau3 phi2 phi3" then critical values are below
# BUT phi2 and phi3 mean AT LEAST ONE of the coefficients is not zero
# DOES NOT mean the trend is individually significant. 
# Because we reject the null, t-stat on trend is normal. We can use that to reject the trend
# From this alone, suspect a stationary series with no trend/growth/drift - go back and use Case 2.

# same with the stationary AR2
# Again, I get the result I should have, 
# when stationary, I can test the trend coefficient directly
# Joint tests do NOT tell me that I should keep the trend
ury2AR2 <- ur.df(y2AR2,type=c("drift"),lags=1)
summary(ury2AR2)
ury2AR2t <- ur.df(y2AR2,type=c("trend"),lags=1)
summary(ury2AR2t)

# Now let's look at series that has a true unit root with NO drift (Case 2, but should fail to reject)
ury2RW <- ur.df(y2RW,type=c("drift"),lags=0)
summary(ury2RW)
# drift option estimates intercept, but fail to reject individual and joint hypotheses
# Can't rule out a random walk with zero drift (which is the true process we simulated)

# Suppose I use the "trend" option, Case 4, on this data when Case 2 is the truth
ury2RWt <- ur.df(y2RW,type=c("trend"),lags=0)
summary(ury2RWt)
# Fail to reject individual and joint hypotheses
# because I fail to reject phi3, can conclude trend coefficient is also insignificant
# CANNOT conclude Intercept is insignificant. 

# To be sure, check that the intercept of differenced model is not statistically different from zero: NO DRIFT
Arima(y2RW,order=c(0,1,0),include.constant = TRUE)

####
# Again, if the truth is Case 2, no harm in estimating Case 4 as long as I am careful about interpreting tests.


##########
# Next let's compare when the truth is Case 4 but we estimate using Case 2
# This is worse - using Case 2 FORCES the trend coefficient to be zero when it may not be. 
y3AR1 <- arima.sim(model=list(order=c(1,0,0),ar=c(0.85)),n=1000) + 0.1*seq(1000)
ts.plot(y3AR1)
y3RW <- arima.sim(model=list(order=c(0,1,0)),mean=0.1,n=1000)
ts.plot(y3RW)

# Let's test y3RW first - the truth is random walk with drift (Case 4, use type="trend")
cadfy3RW = CADFtest(y3RW,criterion=c("AIC"),type = "trend")
summary(cadfy3RW)
urtest = ur.df(y3RW,type=c("trend"),lags=0)
summary(urtest)
# fail to reject the individual tau3 
# reject phi2 the joint test of intercept, lag, and trend
# fail to reject phi3, the joint test of trend=0 and lag=0
# From phi3, conclude that trend=0 and lag=0. 
# What makes phi2 reject then? Probably a significant intercept
# Suggests a random walk with drift. 

# If we use Case 2, we force the trend to be zero
cadfy3RW = CADFtest(y3RW,criterion=c("AIC"),type = "drift")
summary(cadfy3RW)
urtest = ur.df(y3RW,type=c("drift"),lags=0)
summary(urtest)
# fail to reject tau2 (lag level), reject phi1 (lag and intercept)
# What makes phi1 reject? Probably significant intercept
# Since we have evidence of a unit root, difference the series and test the intercept using stationary transformation
Arima(y3RW,order=c(0,1,0),include.constant = TRUE)
# Intercept is significant: drift is real. Needed to use Case 4!

# Now look at the stationary AR1 with trend
cadfy3AR1 = CADFtest(y3AR1,criterion=c("AIC"),type = "trend")
summary(cadfy3AR1)
urtest = ur.df(y3AR1,type=c("trend"),lags=0)
summary(urtest)
# reject, reject, reject - as it should be, this is the right case!

# What if we force the trend to be zero when it belongs?
cadfy3AR1 = CADFtest(y3AR1,criterion=c("AIC"),type = "drift")
summary(cadfy3AR1)
urtest = ur.df(y3AR1,type=c("drift"),lags=0)
summary(urtest)
# Fail to reject lag, reject lag and intercept
# ambiguous conclusion - suggests a random walk!!!!
# This is wrong!
Arima(y3AR1,order=c(0,1,0),include.constant = TRUE)



###########
# Example with oil, gas, and drilling data
getSymbols("MCOILWTICO",src="FRED")
# Oil & gas drilling index
getSymbols("IPN213111N",src="FRED")
# Henry Hub
getSymbols("MHHNGSP",src="FRED")

godlevels = merge.xts(MHHNGSP,MCOILWTICO,IPN213111N,all=TRUE,join="inner")
startdate <- "1997-01-01"    
enddate   <- "2020-07-01"   
godlevels <- window(godlevels, start = startdate, end = enddate)

# let's get a sense of unit roots and seasonality
chartSeries(godlevels$MHHNGSP)
chartSeries(godlevels$MCOILWTICO)
chartSeries(godlevels$IPN213111N)


# let's do the augmented dickey fuller tests:
# not going to use logs, seeing a trend in oil prices, not sure about gas prices, not in drilling

ar(na.omit(diff(godlevels$MHHNGSP)))
summary(ur.df(godlevels$MHHNGSP,type=c("trend"),lags=9))
# tau3: fail to reject unit root
# phi2: fail to reject unit root, zero drift, zero trend
# phi3: fail to reject unit root and zero trend. 
summary(ur.df(godlevels$MHHNGSP,type=c("drift"),lags=9))
# tau2: fail to reject unit root
# phi1: fail to reject unit root and zero drift

ar(na.omit(diff(godlevels$MCOILWTICO)))
summary(ur.df(godlevels$MCOILWTICO,type=c("trend"),lags=1))
# tau3: fail to reject unit root
# phi2: fail to reject unit root, zero drift, zero trend
# phi3: fail to reject unit root and zero trend. 
summary(ur.df(godlevels$MCOILWTICO,type=c("drift"),lags=1))
# tau2: fail to reject unit root
# phi1: fail to reject unit root and zero drift

ar(na.omit(diff(godlevels$IPN213111N)))
summary(ur.df(godlevels$IPN213111N,type=c("drift"),lags=4))
# tau2: fail to reject unit root
# phi1: fail to reject unit root and zero drift








