# Multivariate examples:
### VAR examples: 
#         oil & gas prices plus drilling index, 
#         Brent oil vs. Asia LNG prices


###### Read in packages and data #########
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
#install.packages("MTS")
require(MTS)
require(car)
# install.packages("strucchange")
require(strucchange)
# install.packages("vars")
require(vars)

# WTI
getSymbols("MCOILWTICO",src="FRED")
# Oil & gas drilling index
getSymbols("IPN213111N",src="FRED")
# Henry Hub
getSymbols("MHHNGSP",src="FRED")
# Industrial Production
getSymbols("INDPRO",src="FRED")
# Ocean Freight PPI
getSymbols("PCU483111483111",src="FRED")

# Brent
getSymbols("MCOILBRENTEU",src="FRED")
# LNG Asia
getSymbols("PNGASJPUSDM",src="FRED")


####### Prepare data #######
# merge data sets for each example and calculate returns

# merge LNG and Brent and calculate returns
LNGdata = merge.xts(MCOILBRENTEU,PNGASJPUSDM,join="inner")
plot(LNGdata)
# convert to ts object, 
# note that LNGdata starts in Jan 1992, so returns will start in Feb 1992
dbrent = ts(na.omit(diff(log(LNGdata$MCOILBRENTEU))),freq=12,start=1992+1/12)
dlng = ts(na.omit(diff(log(LNGdata$PNGASJPUSDM))),freq=12,start=1992+1/12)
brlng = cbind(dbrent,dlng)

# merge oil, gas, and drilling and calculate returns/changes
oilgas= merge.xts(MHHNGSP,MCOILWTICO,IPN213111N,INDPRO,PCU483111483111,join="inner")
plot(oilgas)
# calculate log differences as ts() objects, notice start dates
doil = ts(na.omit(diff(log(oilgas$MCOILWTICO))),freq=12,start=1997+1/12)
dgas = ts(na.omit(diff(log(oilgas$MHHNGSP))),freq=12,start=1997+1/12)
dwell = ts(na.omit(diff(oilgas$IPN213111N)),freq=12,start=1997+1/12)
dind = ts(na.omit(diff(oilgas$INDPRO)),freq=12,start=1997+1/12)
dfre = ts(na.omit(diff(oilgas$PCU483111483111)),freq=12,start=1997+1/12)
ogw = cbind(doil,dgas,dwell,dind)


######## Example 1: Var for oil & gas price returns ########

# use vars() package
og = cbind(doil,dgas)
# show series and summary statistics
plot(og,xlab="")
summary(og)

# select lag length
VARselect(og,lag.max=15,type="none")
# HQ and SC select 2 and 1 lags, FPE and AIC select 11 lags, same as below:
varog = VAR(og,lag.max=13,type="none",ic="FPE")
varog
# type = trend, const, both, none
# lag.max= or p=
# can enter exogen= for exogenous variables
# can enter season=12 for seasonal frequency
# from VAR() output, can do use the following commands:
# coef, fevd, fitted, irf, Phi, logLik, plot, 
# predict, print, Psi, resid, summary, Acoef, Bcoef, BQ, causality,
# restrict, roots, 
# diagnostics: arch.test, normality.test, serial.test, stability
plot(varog)
# basic results of entire VAR:
summary(varog)
# summarize and plot individual equations:
summary(varog,equation="dgas")
plot(varog,name="dgas")

# Should go with smaller model if it gets rid of residual autocorrelation
varog1 = VAR(og,type="none",p=1)
plot(varog1)
# mostly, residual autocorrelation is nill.
# looking at the VAR(1) vs. VAR(11), the later lags in the gas equation have
# a lot of explanatory power. 9 months to move rigs and begin production?

# lets go with the VAR(11) for the sake of discussion.
roots(varog)
causality(varog, cause="doil")
causality(varog, cause="dgas")
# appears that oil returns Granger-cause gas returns, but not vice versa.
# There is instantaneous/contemporaneous correlation between the two series.
# diagnostics:
normality.test(varog)
# rejects multivariate normality test!
# test for ARCH effects in the residuals:
arch.test(varog,lags.multi=9)
plot(arch.test(varog,lags.multi=9))
# significant ARCH effects in oil equation, could explain kurtosis/failure of normality.
serial.test(varog,lags.pt = 16)
# no residual serial correlation, which is good.
plot(serial.test(varog)) # same output as plot(arch.test())
# stability plots the cumulative sum of the errors of each equation
# because the errors are positive and negative, they should sum to zero.
# if the cumulative sum deviates, there is probably a structural break.
stability(varog)
plot(stability(varog))
# forecasting
varogp = predict(varog,n.ahead=48)
# 95% CI by default
plot(varogp)
# 90% CI by default, some better visualization tools: 
fanchart(varogp)
# impulse resonse functions. options ortho=T, cumulative=T or F
# can define impulse = "doil", response="dgas"
plot(irf(varog,n.ahead=24,ortho=FALSE))
plot(irf(varog,n.ahead=24,ortho=T))
plot(irf(varog,n.ahead=24,ortho=T,impulse = "doil",response="dgas"))
plot(fevd(varog,n.ahead=24))


# Calculate the covariance matrix of VAR(p) residuals:
Omega = summary(varog)$covres
# Factor it into upper/lower triangular. Recall R gives the upper triangular, when we want its transpose:
P = chol(Omega)
# check that products of lower triangular P and with its transpose return Omega:
t(P)%*%P
Omega

# could report in ADA' form
Dhalf = diag(P)
A = t(P/Dhalf)
# check that we get Omega back if we factored correctly:
A%*%diag(Dhalf)%*%diag(Dhalf)%*%t(A)
# report A:
A

# notice that the simple regression of residuals from gas equation on
# residuals from oil equation gives identical coefficient to the adjustment made in the A matrix;
# A matrix adjusts for their contemporaneous linear relationship
Areg = lm(residuals(varog$varresult$dgas) ~ residuals(varog$varresult$doil))
summary(Areg)


# Compare to structural VAR
# transform reduced form into structural
Avar = matrix(c(1,NA,0,1),nrow=2,ncol=2)
svarog = SVAR(varog,estmethod="direct",Amat=Avar,hessian=T)
# estimation method direct is MLE. hessian=T required for standard errors.
# coefficients of the structural matrix (contemporaneous coefficients)
svarog$A
# standard errors of the structural coefficients matrix
svarog$Ase
summary(svarog)
plot(irf(svarog,n.ahead=24,ortho=T))



####### Example 2: Oil prices, gas  prices, and drilling #######

god = cbind(dwell,doil,dgas)
# show series and summary statistics
plot(god,xlab="")
summary(god)

VARselect(god,lag.max=13,type="none")
# FPE and AIC select 4 lags, same as below:
vargod = VAR(god,lag.max=13,type="none",ic="FPE")
vargod
# type = trend, const, both, none
# lag.max= or p=
# can enter exogen= for exogenous variables
# can enter season=12 for seasonal frequency
# from VAR() output, can do coef, fevd, fitted, irf, Phi, logLik, plot, 
# predict, print, Psi, resid, summary, Acoef, Bcoef, BQ, causality,
# restrict, roots, 
# diagnostics: arch.test, normality.test, serial.test, stability

# basic results
summary(vargod)
summary(vargod,equation="dwell")
plot(vargod)
plot(vargod,name="dwell")
# 12 roots because n=3 variables, p=4 lags and companion form is n*p X n*p = 12
roots(vargod)
# joint Granger-causality test of oil on gas and oil on wells:
causality(vargod, cause="doil")
# diagnostics:
normality.test(vargod)
# rejects normality test for both skewness and kurtosis!
arch.test(vargod,lags.multi=4)
plot(arch.test(vargod,lags.multi=4))
# significant ARCH effects, could explain failure of normality.
serial.test(vargod,lags.pt = 16)
# no residual serial correlation, which is good.
plot(serial.test(vargod))
# stability plots the cumulative sum of the errors of each equation
# because the errors are positive and negative, they should sum to zero.
# if the cumulative sum deviates, there is probably a structural break.
stability(vargod)
plot(stability(vargod))
# forecasting
vargodp = predict(vargod,n.ahead=48)
# 95% CI by default
plot(vargodp)
# 90% CI by default, some better visualization tools: 
fanchart(vargodp)
# impulse resonse functions. options ortho=T, cumulative=T or F
# can define impulse = "doil", response="dwell"
plot(irf(vargod,n.ahead=24,ortho=F))
plot(irf(vargod,n.ahead=24,ortho=T))
plot(irf(vargod,n.ahead=24,ortho=T,impulse = "doil",response="dwell"))
plot(irf(vargod,n.ahead=24,ortho=T,impulse = "doil",response="dwell",cumulative = T))
# from the above, a one-standard deviation shock in the oil price return leads to a cumulative 
# 7 or 8 unit increase in the changes in the drilling index
plot(fevd(vargod,n.ahead=24))


# Calculate the covariance matrix of VAR(p) residuals:
Omega = summary(vargod)$covres
# Factor it into upper/lower triangular. Recall R gives the upper triangular, when we want its transpose:
P = chol(Omega)
# check that products of lower triangular P and with its transpose return Omega:
t(P)%*%P

# could report in ADA' form
Dhalf = diag(P)
A = t(P/Dhalf)
# check that we get Omega back if we factored correctly:
A%*%diag(Dhalf)%*%diag(Dhalf)%*%t(A)
# report A:
A

# transform reduced form into structural
Avar = matrix(c(1,NA,NA,0,1,NA,0,0,1),nrow=3,ncol=3)
svargod = SVAR(vargod,estmethod="direct",Amat=Avar,hessian=T)
# estimation method direct is MLE. hessian=T required for standard errors.
# coefficients of A matrix
svargod$A
Ao = solve(svargod$A)
# compare
Ao
A
# standard errors of A matrix
svargod$Ase
summary(svargod)
plot(irf(svargod,n.ahead=24,ortho=T))



####### Example 2a: Kilian-like example #######

kil = cbind(dwell,dfre,dind,doil)
# show series and summary statistics
plot(kil,xlab="")
summary(kil)

VARselect(kil,lag.max=13,type="none")
# FPE and AIC select 2 lags, same as below:
varkil = vars::VAR(kil,lag.max=13,type="none",ic="FPE")
varkil
# type = trend, const, both, none
# lag.max= or p=
# can enter exogen= for exogenous variables
# can enter season=12 for seasonal frequency
# from VAR() output, can do coef, fevd, fitted, irf, Phi, logLik, plot, 
# predict, print, Psi, resid, summary, Acoef, Bcoef, BQ, causality,
# restrict, roots, 
# diagnostics: arch.test, normality.test, serial.test, stability

# basic results
summary(varkil)
summary(varkil,equation="dwell")
plot(varkil)
plot(varkil,name="dwell")
# 6 roots because n=3 variables, p=2 lags and companion form is n*p X n*p = 6
roots(varkil)
# joint Granger-causality test of oil on gas and oil on wells:
causality(varkil, cause="doil")
# diagnostics:
normality.test(varkil)
# rejects normality test for both skewness and kurtosis!
arch.test(varkil,lags.multi=4)
plot(arch.test(varkil,lags.multi=4))
# significant ARCH effects, could explain failure of normality.
serial.test(varkil,lags.pt = 16)
# no residual serial correlation, which is good.
plot(serial.test(varkil))
# stability plots the cumulative sum of the errors of each equation
# because the errors are positive and negative, they should sum to zero.
# if the cumulative sum deviates, there is probably a structural break.
stability(varkil)
plot(stability(varkil))
# forecasting
varkilp = predict(varkil,n.ahead=48)
# 95% CI by default
plot(varkilp)
# 90% CI by default, some better visualization tools: 
fanchart(varkilp)
# impulse resonse functions. options ortho=T, cumulative=T or F
# can define impulse = "doil", response="dwell"
plot(irf(varkil,n.ahead=24,ortho=F))
plot(irf(varkil,n.ahead=24,ortho=T))
plot(irf(varkil,n.ahead=24,ortho=T,impulse = "dind",response="dwell"))
plot(irf(varkil,n.ahead=24,ortho=T,impulse = "doil",response="dwell",cumulative = T))
# from the above, a one-standard deviation shock in the oil price return leads to a cumulative 
# four unit increase in the changes in the drilling index
plot(fevd(varkil,n.ahead=24))


# Calculate the covariance matrix of VAR(p) residuals:
Omega = summary(varkil)$covres
# Factor it into upper/lower triangular. Recall R gives the upper triangular, when we want its transpose:
P = chol(Omega)
# check that products of lower triangular P and with its transpose return Omega:
t(P)%*%P

# could report in ADA' form
Dhalf = diag(P)
A = t(P/Dhalf)
# check that we get Omega back if we factored correctly:
A%*%diag(Dhalf)%*%diag(Dhalf)%*%t(A)
# report A:
A

# transform reduced form into structural
Avar = matrix(c(1,NA,NA,0,1,NA,0,0,1),nrow=3,ncol=3)
svarkil = SVAR(varkil,estmethod="direct",Amat=Avar,hessian=T)
# estimation method direct is MLE. hessian=T required for standard errors.
# coefficients of A matrix
svarkil$A
Ao = solve(svarkil$A)
# compare
Ao
A
# standard errors of A matrix
svarkil$Ase
summary(svarkil)
plot(irf(svarkil,n.ahead=24,ortho=T))



##### Example using Tsay's MTS package #####
#### Tsay also has a package called MTS (multivariate time series)
#### Many of the same command names as vars() package, need to be careful
require(MTS)
# Using MTS package commands
# investigate lag order
# criteria differ on optimal lag order, could go with 1, 2, 4
# just for sake of illustration let's do 2
VARorder(god,maxp=12,output=T)
# estimate the VAR
vardrill = MTS::VAR(god,p=2,output=T,include.mean = T)
# calculate forecasts
drpred = VARpred(vardrill,h=100)

# one way to illustrate impulse response functions
# need to feed in coefficients and error covariance from estimated model
VARMAirf(vardrill$Phi,Sigma=vardrill$Sigma)
# two graphs are shown: one time response and cumulative
# notice bottom row is response of wells to gas, oil, and wells in that order
# wells are really the only thing that responds to anything
# respond more to oil than gas, but both responses take a couple months.

# refVAR() drops insignificant coefficients
# default is t-stat below 1 gets dropped.
# take estimated model and "refine" it
# AIC improves, many zero coefficients in the AR matrices
vard2 = refVAR(vardrill)
drpred2 = VARpred(vard2,h=100)
VARMAirf(vard2$Phi,Sigma=vard2$Sigma)


# I can take the gas price return forecast only
predgas = drpred$pred[c(1:100),1]
# I can plot each return (oil, then wells)
plot(ts(drpred$pred[c(1:100),2]))
plot(ts(drpred$pred[c(1:100),3]))
ts.plot(dgas)
lines(ts(predgas,start=2017+10/12,freq=12),col=3)
plot(ts(predgas,start=2017+10/12,freq=12),col=3)


######## Example 3: Brent and LNG ######
# show series and summary statistics
plot(brlng,xlab="")
summary(brlng)

# select lag length
VARselect(brlng,lag.max=15,type="none")
# FPE and AIC select 14 lags
varbr = VAR(brlng,lag.max=15,type="none",ic="AIC")
varbr
# type = trend, const, both, none
# lag.max= or p=
# can enter exogen= for exogenous variables
# can enter season=12 for seasonal frequency
# from VAR() output, can do use the following commands:
# coef, fevd, fitted, irf, Phi, logLik, plot, 
# predict, print, Psi, resid, summary, Acoef, Bcoef, BQ, causality,
# restrict, roots, 
# diagnostics: arch.test, normality.test, serial.test, stability
plot(varbr)
# basic results of entire VAR:
summary(varbr)
# summarize and plot individual equations:
summary(varbr,equation="dlng")
plot(varbr,name="dlng")

# Should go with smaller model if it gets rid of residual autocorrelation
varbr1 = VAR(brlng,type="none",p=10)
plot(varbr1)
# mostly, residual autocorrelation is nill except 12th lags in the LNG equation.

# lets go with the VAR(14) for the sake of discussion.
roots(varbr)
causality(varbr, cause="dbrent")
causality(varbr, cause="dlng")
# appears that brent returns Granger-cause lng returns, and vice versa.
# There is no instantaneous/contemporaneous correlation between the two series.
# diagnostics:
normality.test(varbr)
# rejects multivariate normality test for kurtosis!
# test for ARCH effects in the residuals:
arch.test(varbr,lags.multi=13)
plot(arch.test(varbr,lags.multi=13))
# significant ARCH effects in oil equation, could explain kurtosis/failure of normality.
serial.test(varbr,lags.pt = 20)
# still some residual autocorrelation
plot(serial.test(varbr)) # same output as plot(arch.test())
# stability plots the cumulative sum of the errors of each equation
# because the errors are positive and negative, they should sum to zero.
# if the cumulative sum deviates, there is probably a structural break.
stability(varbr)
plot(stability(varbr))
# forecasting
varbrp = predict(varbr,n.ahead=48)
# 95% CI by default
plot(varbrp)
# 90% CI by default, some better visualization tools: 
fanchart(varbrp)
# impulse resonse functions. options ortho=T, cumulative=T or F
# can define impulse = "dbrent", response="dlng"
plot(irf(varbr,n.ahead=24,ortho=FALSE))
plot(irf(varbr,n.ahead=24,ortho=T))
plot(irf(varbr,n.ahead=24,ortho=T,impulse = "dbrent",response="dlng"))
plot(fevd(varbr,n.ahead=24))


# Calculate the covariance matrix of VAR(p) residuals:
Omega = summary(varbr)$covres
# Factor it into upper/lower triangular. Recall R gives the upper triangular, when we want its transpose:
P = chol(Omega)
# check that products of lower triangular P and with its transpose return Omega:
t(P)%*%P
Omega

# could report in ADA' form
Dhalf = diag(P)
A = t(P/Dhalf)
# check that we get Omega back if we factored correctly:
A%*%diag(Dhalf)%*%diag(Dhalf)%*%t(A)
# report A:
A

# notice that the simple regression of residuals from lng equation on
# residuals from brent equation gives identical coefficient to the adjustment made in the A matrix;
# A matrix adjusts for their contemporaneous linear relationship
Areg = lm(residuals(varbr$varresult$dlng) ~ residuals(varbr$varresult$dbrent))
summary(Areg)


# Compare to structural VAR
# transform reduced form into structural
Avar = matrix(c(1,NA,0,1),nrow=2,ncol=2)
svarbr = SVAR(varbr,estmethod="direct",Amat=Avar,hessian=T)
# estimation method direct is MLE. hessian=T required for standard errors.
# coefficients of the structural matrix (contemporaneous coefficients)
svarbr$A
# standard errors of the structural coefficients matrix
svarbr$Ase
summary(svarbr)
plot(irf(svarbr,n.ahead=24,ortho=T))

