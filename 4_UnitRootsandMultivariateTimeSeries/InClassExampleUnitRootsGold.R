# Program to investigate unit root nonstationarity

require(quantmod)
require(fBasics)
require(tseries)
require(CADFtest)
require(urca)
require(forecast)
require(lubridate)

# Data for HW 2 ------------------------
getSymbols("T10YIE",src="FRED") # 10-year break-even inflation rate (daily)
getSymbols("WPU081",src="FRED") # Lumber PPI (monthly)
getSymbols("WPU10", src="FRED") # Metals PPI (monthly)
getSymbols("MCOILWTICO",src="FRED") # Oil spot price (monthly)

# Convert 10-year inflation to monthly averages for compatibility
TYINF <- apply.monthly(na.omit(T10YIE), FUN=mean)
TYINF = xts(TYINF,order.by = as.yearmon(index(TYINF)))

# Pick a common time window (Jan 2003 to Aug 2021)
startdate <- "2003-01-01"    
enddate   <- "2021-08-01"   
WPU081 <- na.omit(window(WPU081, start = startdate, end = enddate))
WPU10 <- na.omit(window(WPU10, start = startdate, end = enddate))
MCOILWTICO <- na.omit(window(MCOILWTICO, start = startdate, end = enddate))
TYINF <- na.omit(window(TYINF, start = startdate, end = enddate))

# Put all 4 series in one object with column names
data = merge.xts(TYINF,WPU081,WPU10,log(MCOILWTICO),join="inner")
colnames(data) <- c("TYINF","LumberPPI","MetalsPPI","LogWTI")

########## Lumber PPI ############

# this step isn't actually necessary:
lumb <- ts(data$LumberPPI,freq=12,start=2003) # transform to a ts object
ts.plot(lumb)
# Lumber appears to have a trend - use Case 4
CADFlumb = CADFtest(lumb,max.lag.y=48,criterion=c("BIC"),type="trend")
summary(CADFlumb) # fail to reject the null of a unit root, 11 lags. 
# using ur.df() I can test joint hypotheses
urdflumb <- ur.df(lumb,selectlags = c("BIC"),lags=48,type="trend")
summary(urdflumb) # fail to reject phi2. But trend is individually significant. 
# Move to CASE 2, estimate ARIMA with drift
urdflumb2 <- ur.df(lumb,lags=11,type="drift")
summary(urdflumb2) # fail to reject phi1. 
# Random walk without drift. 
# Or Random walk with trend (trend term is significant)
# A little surprising based on picture. 
# Regress lumber first diffs on an intercept, not significant: 
summary(lm(na.omit(diff(lumb))~1))

########## Metals PPI ############

# this step isn't actually necessary:
met <- ts(data$MetalsPPI,freq=12,start=2003) # transform to a ts object
ts.plot(met)
# Metals PPI appears to have a trend - use Case 4
CADFmet = CADFtest(met,max.lag.y=48,criterion=c("BIC"),type="trend")
summary(CADFmet) # fail to reject the null of a unit root. One lag. 
# using ur.df() I can test joint hypotheses
urdfmet <- ur.df(met,selectlags = c("BIC"),lags=48,type="trend")
summary(urdfmet) # fail to reject phi2. Go to CASE 2
# Move to CASE 2, estimate ARIMA with drift
urdfmet2 <- ur.df(met,lags=1,type="drift")
summary(urdfmet2) # fail to reject phi1. 
# Random walk without drift. 
# Regress metals first diffs on an intercept, IS significant: 
summary(lm(na.omit(diff(met))~1))
t.test(na.omit(diff(met)))
# What do you do?

########## Oil WTI ############

# this step isn't actually necessary:
wti <- ts(data$LogWTI,freq=12,start=2003) # transform to a ts object
ts.plot(wti)
# WTI appears NOT to have a trend - use Case 2
CADFwti = CADFtest(wti,max.lag.y=48,criterion=c("BIC"),type="drift")
summary(CADFwti) # REJECT the null of a unit root! One lag. 
# But look at lagged coefficient!
# using ur.df() I can test joint hypotheses
urdfwti <- ur.df(wti,selectlags = c("BIC"),lags=48,type="drift")
summary(urdfwti) # Reject both individual and joint null hypotheses. 


# Data for gold markets and monetary policy ------------
getSymbols("GD=F")   #gold fixing 1030am london
getSymbols("DFII10",src="FRED")             #10y TIPS const maturity
#getSymbols("T10YIE",src="FRED")             #10y breakeven inflation
getSymbols("DTWEXBGS",src="FRED")           #trade-wtd USD index (goods & svcs, select which)


grd <- merge.xts(`GD=F`$`GD=F.Adjusted`, DFII10$DFII10, 
                 T10YIE$T10YIE, DTWEXBGS,join = "inner")

colnames(grd) <- c("gld","tip", "bei","twd")

#SELECT START AND END DATE
startdate <- "2006-01-03"    
enddate   <- "2021-08-30"    

grd <- window(grd, start = startdate, end = enddate)
grd <- na.omit(grd)


########### Gold prices ###########
ts.plot(grd$gld) # appears to have a trend
x1g <- CADFtest(grd$gld,criterion = c("AIC"),type="trend")
summary(x1g)      # fail to reject null of unit root, one additional lags
x2g <- ur.df(grd$gld,type=c("trend"),lags=1)
summary(x2g)
# Fail to reject (accept) phi2. Move to Case 2, estimate ARIMA with drift
x3g <- ur.df(grd$gld,type=c("drift"),lags=1)
summary(x3g)
# Fail to reject phi1. Random Walk without drift. 
# Somewhat surprising based on picture. 
# Regress gold returns on an intercept, intercept is not significant: 
summary(lm(na.omit(diff(log(grd$gld)))~1))

######## TIPS ############
ts.plot(grd$tip) # appears to have a trend
x1t <- CADFtest(grd$tip,criterion = c("AIC"),type="trend")
summary(x1t)  # fail to reject null of a unit root, one additional lag change
x2t <- ur.df(grd$tip,type=c("trend"),lags=1)
summary(x2t)
# Fail to reject (accept) phi2. Move to Case 2, estimate ARIMA with drift
x3t <- ur.df(grd$tip,type=c("drift"),lags=1)
summary(x3t)
# Fail to reject phi1. Random Walk without drift. 
# Somewhat surprising based on picture. 
# Regress TIPS changes on an intercept, intercept is not significant: 
summary(lm(na.omit(diff(grd$tip))~1))

########### Trade-weighted dollar index ##########
ts.plot(grd$twd)    # Again, seems to have a trend
x1d <- CADFtest(grd$twd,criterion = c("AIC"),type="trend")
summary(x1d)    # fail to reject unit root, also include one lag
x2d <- ur.df(grd$twd,type=c("trend"),lags=1)
summary(x2d)
# Again fail to reject phi2, move to Case 2
x3d <- ur.df(grd$twd,type=c("drift"),lags=1)
summary(x3d)
# Fail to reject - random walk without drift. 
