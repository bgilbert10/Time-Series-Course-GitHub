##########################
# 1) download gold prices, TIPS, Break-even inflation, Trade-weighted dollar 
#     index and merge them for a common time window of 01-03-2017 to 09-30-2021
# 2) investigate structural breaks in the mean gold returns
# 3) investigate structural breaks in the relationship

require(forecast)
require(quantmod)
#require(caschrono)
require(texreg)
require(strucchange)
#require(rollRegres)


# 1) 
getSymbols("GC=F")   #gold fixing 1030am london 1968-04-01
getSymbols("DFII10",src="FRED")             #10y TIPS const maturity 2003-01-02
getSymbols("T10YIE",src="FRED")             #10y breakeven inflation 2003-01-02
getSymbols("DTWEXBGS",src="FRED")           #trade-wtd USD index (goods & svcs) 2006-01-02
getSymbols("SP500",src="FRED")              #S&P500 2011-10-03


grd <- merge.xts(`GC=F`$`GC=F.Adjusted`, DFII10$DFII10, 
                 T10YIE$T10YIE, DTWEXBGS, SP500$SP500,
                 all=TRUE, fill=NA, retclass="xts")

colnames(grd) <- c("gld","tip", "bei","twd","sp5")

#SELECT START AND END DATE
startdate <- "2017-01-03"  
enddate   <- "2021-09-30"    

grd <- window(grd, start = startdate, end = enddate)
grd <- na.omit(grd)

chartSeries(grd$gld)
chartSeries(grd$tip)
chartSeries(grd$bei)
chartSeries(grd$twd)
chartSeries(grd$sp5)


dlgld  = ts(na.omit(diff(log(grd$gld))),freq=252,start = 2017)
dtip   = ts(na.omit(diff(grd$tip)),freq=252,start = 2017)
dbei   = ts(na.omit(diff(grd$bei)),freq=252,start = 2017)
dtwd   = ts(na.omit(diff(grd$twd)),freq=252,start = 2017)
dsp5   = ts(na.omit(diff(grd$sp5)),freq=252,start = 2017)


# 2) Estimate and investigate break points in the gold returns using the breakpoints function
ts.plot(dlgld)
bp_gold = breakpoints(dlgld~1)
breakpoints(bp_gold)
summary(bp_gold)
plot(bp_gold)

# 3) Estimate and investigate break points in gold returns relationship with 3 monetary variables 

# no breaks
mod <- lm(dlgld ~ dtip + dbei + dtwd + dsp5)
summary(mod)

bp.gold = breakpoints(dlgld~dtip + dbei + dtwd + dsp5  )
breakpoints(bp.gold)
summary(bp.gold)
plot(bp.gold)
ci.gold = confint(bp.gold,breaks=2)
# optimal number of breaks is 2
coef(bp.gold,breaks=2)
sqrt(sapply(vcov(bp.gold, breaks = 2), diag))

# Only break in intercept:
fm1 = lm(dlgld~0+dtip + dbei + dtwd+dsp5+breakfactor(bp.gold,breaks=2))
# Break in intercept and tips and bei coefficients:
fm2 = lm(dlgld~0 + dtwd+dsp5+breakfactor(bp.gold,breaks=2)/(dtip + dbei))
# Breaks in all coefficients:
fm3 = lm(dlgld~(breakfactor(bp.gold,breaks=2)/(dtip + dbei+ dtwd+dsp5)))
summary(fm1)
summary(fm2)
summary(fm3)
plot(dlgld)
lines(ts(fitted(mod),start=2017,freq=252),col=2)
lines(ts(fitted(fm1),start=2017,freq=252),col=3)
lines(ts(fitted(fm3),start=2017,freq=252),col=4)
lines(bp.gold,breaks=2)
lines(ci.gold)

plot(ts(residuals(mod),start=2017,freq=252))
lines(ts(residuals(fm1),start=2017,freq=252),col=2)
lines(ts(residuals(fm3),start=2017,freq=252),col=4)
lines(bp.gold,breaks=2)
lines(ci.gold)

plot(ts(grd$gld,start=2017,freq=252))
lines(bp.gold,breaks=2)
lines(ci.gold)

plot(ts(grd$tip,start=2017,freq=252))
lines(bp.gold,breaks=2)
lines(ci.gold)

plot(ts(grd$bei,start=2017,freq=252))
lines(bp.gold,breaks=2)
lines(ci.gold)

plot(ts(grd$twd,start=2017,freq=252))
lines(bp.gold,breaks=2)
lines(ci.gold)

plot(ts(grd$sp5,start=2017,freq=252))
lines(bp.gold,breaks=2)
lines(ci.gold)

#Try RE, ME, and rescale = FALSE
#Try OLS-CUSUM, Rec-CUSUM, OLS-MOSUM, Rec-MOSOM
#t <- index(grd)[c(-1)]
efptest.sum <- efp(dlgld~dtip + dbei + dtwd + dsp5,type="OLS-CUSUM")
plot(efptest.sum)
sctest(efptest.sum)

efptest.est <- efp(dlgld~dtip + dbei + dtwd + dsp5,type="RE",rescale=TRUE)
plot(efptest.est)
plot(efptest.est,functional=NULL)

## Rolling regression (rollRegres is no longer supported by CRAN)
width  <- 252L # regression window

xs    = cbind(dtip,dbei,dtwd,dsp5) # X vector for right hand side variables
fitg <- roll_regres(dlgld ~ xs, width = width) # rolling regression
# plot coefficients

ts.plot(fitg$coefs[,1],gpars = list(main="Intercept"))
ts.plot(fitg$coefs[,2],gpars = list(main="beta TIPS"))
ts.plot(fitg$coefs[,3],gpars = list(main="beta BEI"))
ts.plot(fitg$coefs[,4],gpars = list(main="beta TWD"))
ts.plot(fitg$coefs[,5],gpars = list(main="beta S&P"))



