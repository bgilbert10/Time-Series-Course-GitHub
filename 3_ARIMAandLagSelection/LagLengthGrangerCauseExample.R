# Preamble ------------------

# Using sequential F-tests with HAC standard errors, vs. AIC, to select lag length
# Also test for Granger causality of one variable against another

require(quantmod)
require(forecast)
require(fBasics)
require(CADFtest)
require(urca)
# install.packages("sandwich")
require(sandwich)
# install.packages("lmtest")
require(lmtest)
require(nlme)
# install.packages("MTS")
require(MTS)
require(car)
# install.packages("strucchange")
require(strucchange)
# install.packages("vars")
require(vars)

# Read and organize data ---------------

getSymbols("MCOILWTICO",src="FRED")
# Oil & gas drilling index
getSymbols("IPN213111N",src="FRED")
# Henry Hub
getSymbols("MHHNGSP",src="FRED")


data = merge.xts(MHHNGSP,MCOILWTICO,IPN213111N,join="inner")
plot(data)

dgas = ts(na.omit(diff(log(data$MHHNGSP))),freq=12,start=1997+1/12)
doil = ts(na.omit(diff(log(data$MCOILWTICO))),freq=12,start=1997+1/12)
dwell = ts(na.omit(diff(data$IPN213111N)),freq=12,start=1997+1/12)

god = cbind(dgas,doil,dwell)
MTSplot(god)

# Brute force sequential F-tests or AIC ---------------

# sequentially estimate the model with 6 lags, then 5, then 4, etc. and 
# capture the AIC each time, but also do joint F-test of all lags at each lag level
x6 = dynlm(dwell ~ L(dgas,c(1:6)) + L(doil,c(1:6)) + L(dwell,c(1:6)))
a6 = AIC(x6)
F6 = linearHypothesis(x6,c("L(dgas, c(1:6))6=0","L(doil, c(1:6))6=0" ,"L(dwell, c(1:6))6=0"),vcov=vcovHAC(x6),test="F",data=x6)
f6p = F6$`Pr(>F)`

x5 = dynlm(dwell ~ L(dgas,c(1:5)) + L(doil,c(1:5)) + L(dwell,c(1:5)))
a5 = AIC(x5)
F5 = linearHypothesis(x5,c("L(dgas, c(1:5))5=0","L(doil, c(1:5))5=0" ,"L(dwell, c(1:5))5=0"),vcov=vcovHAC(x5),test="F",data=x5)
f5p = F5$`Pr(>F)`

x4 = dynlm(dwell ~ L(dgas,c(1:4)) + L(doil,c(1:4)) + L(dwell,c(1:4)))
a4 = AIC(x4)
F4 = linearHypothesis(x4,c("L(dgas, c(1:4))4=0","L(doil, c(1:4))4=0" ,"L(dwell, c(1:4))4=0"),vcov=vcovHAC(x4),test="F",data=x4)
f4p = F4$`Pr(>F)`

x3 = dynlm(dwell ~ L(dgas,c(1:3)) + L(doil,c(1:3)) + L(dwell,c(1:3)))
a3 = AIC(x3)
F3 = linearHypothesis(x3,c("L(dgas, c(1:3))3=0","L(doil, c(1:3))3=0" ,"L(dwell, c(1:3))3=0"),vcov=vcovHAC(x3),test="F",data=x3)
f3p = F3$`Pr(>F)`

x2 = dynlm(dwell ~ L(dgas,c(1:2)) + L(doil,c(1:2)) + L(dwell,c(1:2)))
a2 = AIC(x2)
F2 = linearHypothesis(x2,c("L(dgas, c(1:2))2=0","L(doil, c(1:2))2=0" ,"L(dwell, c(1:2))2=0"),vcov=vcovHAC(x2),test="F",data=x2)
f2p = F2$`Pr(>F)`

x1 = dynlm(dwell ~ L(dgas,c(1:1)) + L(doil,c(1:1)) + L(dwell,c(1:1)))
a1 = AIC(x1)
F1 = linearHypothesis(x1,c("L(dgas, c(1:1))=0","L(doil, c(1:1))=0" ,"L(dwell, c(1:1))=0"),vcov=vcovHAC(x1),test="F",data=x1)
f1p = F1$`Pr(>F)`

F.results = cbind(f1p,f2p,f3p,f4p,f5p,f6p)
AIC.results = cbind(a1,a2,a3,a4,a5,a6)

# model 4 rejects the null in the F-test, model 6 has the lowest AIC. 

tsdisplay(residuals(x6))
tsdisplay(residuals(x4))
# both get rid of autocorrelationn in the residuals - either one is fine. 

summary(x4)
coeftest(x4,vcov=vcovHAC(x4))

# Loop sequential F-tests and AIC calculation --------------

# doing this in a loop:
fp = list()
a = list()
for (i in 2:6)
{ 
  x = dynlm(dwell ~ L(dgas,c(1:i)) + L(doil,c(1:i)) + L(dwell,c(1:i)))
  a[i] = AIC(x)
  Ft = linearHypothesis(x,c(paste("L(dgas, c(1:i))",i,"=0",sep="")
                                ,paste("L(doil, c(1:i))",i,"=0",sep="")
                                ,paste("L(dwell, c(1:i))",i,"=0",sep=""))
                           ,vcov=vcovHAC(x)
                           ,test="F",data=x)
  fp[i] = Ft$`Pr(>F)`[2]
}

# Test Granger Causality ----------------

# do oil price changes or gas price changes granger-cause drilling activity changes?
linearHypothesis(x4,c("L(doil, c(1:4))1=0","L(doil, c(1:4))2=0" ,"L(doil, c(1:4))3=0","L(doil, c(1:4))4=0"),vcov=vcovHAC(x4),test="F",data=x4)
linearHypothesis(x4,c("L(dgas, c(1:4))1=0","L(dgas, c(1:4))2=0" ,"L(dgas, c(1:4))3=0","L(dgas, c(1:4))4=0"),vcov=vcovHAC(x4),test="F",data=x4)


