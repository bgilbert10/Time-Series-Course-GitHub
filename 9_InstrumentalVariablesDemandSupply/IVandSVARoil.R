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

# install.packages("systemfit")
require(systemfit)  
# install.packages("AER")
require(AER)
# install.packages("sem)
require(sem)
# install.packages("lfe")
require(lfe)

require(fixest)

require(texreg)
require(lmtest)
require(sandwich)

# Read in data

# WTI
getSymbols("MCOILWTICO",src="FRED")
# Oil & gas drilling index
getSymbols("IPN213111N",src="FRED")
# Henry Hub
getSymbols("MHHNGSP",src="FRED")
# Industrial Production
getSymbols("INDPRO",src="FRED")
# Composite normalized GDP for US (already stationary transformation)
getSymbols("USALORSGPNOSTSAM",src="FRED")
# Brave-Butters-Kelley Real GDP Index (already stationary transformation)
getSymbols("BBKMGDP",src="FRED")

####### Prepare data #######
# merge data sets for each example and calculate returns

# merge oil, gas, and drilling and calculate returns/changes
oilgas= merge.xts(MHHNGSP,MCOILWTICO,IPN213111N,INDPRO,USALORSGPNOSTSAM,BBKMGDP,join="inner")
plot(oilgas)
colnames(oilgas) <- c("ngas","oil","drill","indpro","gdpind","bbk")
# calculate log differences as ts() objects, notice start dates
doil = ts(na.omit(100*diff(log(oilgas$MCOILWTICO))),freq=12,start=1997+1/12)
dgas = ts(na.omit(100*diff(log(oilgas$MHHNGSP))),freq=12,start=1997+1/12)
dwell = ts(na.omit(diff(oilgas$IPN213111N)),freq=12,start=1997+1/12)
dind = ts(na.omit(diff(oilgas$INDPRO)),freq=12,start=1997+1/12)
gdpi = ts(na.omit(oilgas$USALORSGPNOSTSAM[-1,]),freq=12,start=1997+1/12)
bbk = ts(na.omit(oilgas$BBKMGDP[-1,]),freq=12,start=1997+1/12)
ogw = cbind(doil,dgas,dwell,dind,gdpi,bbk)
rownames(ogw) <- 1:nrow(ogw)
ogw <- cbind(time = rownames(ogw), ogw)
colnames(ogw) <- c("time","doil","dgas","dwell","dind","gdpi","bbk")

oilp_iv <- felm(doil ~ 1 | 0 |
                   (dind ~ dwell) | 0 , 
                 data = ogw)
summary(oilp_iv)

########## Showing static SVAR: ##########
eq2 = lm(dind ~ dwell,data=ogw)
eq3 = lm(doil ~ dwell + dind, data=ogw)
summary(eq2)
summary(eq3)
# price response to the industrial production (demand shock): 
(eq3$coefficients[2] + eq3$coefficients[3]*eq2$coefficients[2])/(eq2$coefficients[2])


####### Example: Kilian-like example for oil #######

kil = cbind(dwell,gdpi,doil)
kil = cbind(dwell,bbk,doil)
kil = cbind(dwell,dind,doil)
colnames(kil) <- c("dwell","dind","doil")
kil <- na.omit(kil)
plot(kil,xlab="")
summary(kil)

VARselect(kil,lag.max=13,type="none")
# FPE and AIC select 2 lags, same as below:
varkil = VAR(kil,p=2,type="none")
varkil = VAR(kil,lag.max=13,type="none",ic="AIC")
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
plot(irf(varkil,n.ahead=24,ortho=F,impulse = "dind",response="doil"))
plot(irf(varkil,n.ahead=24,ortho=T))
plot(irf(varkil,n.ahead=24,ortho=T,impulse = "dind",response="doil"))
plot(irf(varkil,n.ahead=24,ortho=T,impulse = "dind",response="doil",cumulative = T))
# from the above, a one-standard deviation shock in the oil price return leads to a cumulative 
# four unit increase in the changes in the drilling index
plot(fevd(varkil,n.ahead=24))

plot(irf(varkil,n.ahead=24,ortho=T,impulse = "dwell",response="doil"))
plot(irf(varkil,n.ahead=24,ortho=T,impulse = "dind",response="doil"))

plot(irf(varkil,n.ahead=24,ortho=T,response="doil"))
plot(irf(varkil,n.ahead=24,ortho=T,response="dwell"))

irfs = irf(varkil,n.ahead=10,ortho=T)
irfs$irf$dwell
irfs$irf$dind
irfs$irf$doil

# Aggregate demand shock identifies supply elasticity:
# d q(t+1)/d p(t) = (d dwell(t+1)/d dind(t))/(d doil(t)/d dind(t)) = 
irfs$irf$dind[,"dwell"][2]/irfs$irf$dind[,"doil"][1]

# Long run elasticity: 
irfs$irf$dind[,"dwell"][2]/irfs$irf$dind[,"doil"][1] + 
  irfs$irf$dind[,"dwell"][3]/irfs$irf$dind[,"doil"][1] +
  irfs$irf$dind[,"dwell"][4]/irfs$irf$dind[,"doil"][1] +
  irfs$irf$dind[,"dwell"][5]/irfs$irf$dind[,"doil"][1] +
  irfs$irf$dind[,"dwell"][6]/irfs$irf$dind[,"doil"][1] +
  irfs$irf$dind[,"dwell"][7]/irfs$irf$dind[,"doil"][1] +
  irfs$irf$dind[,"dwell"][8]/irfs$irf$dind[,"doil"][1] +
  irfs$irf$dind[,"dwell"][9]/irfs$irf$dind[,"doil"][1] +
  irfs$irf$dind[,"dwell"][10]/irfs$irf$dind[,"doil"][1] 

# Precautionary demand shock identifies supply elasticity:
# d q(t+1)/d p(t) = (d dwell(t+1)/d doil(t))/(d doil(t)/d doil(t)) = 
irfs$irf$doil[,"dwell"][2]/irfs$irf$doil[,"doil"][1]

# Long run elasticity: 
irfs$irf$doil[,"dwell"][2]/irfs$irf$doil[,"doil"][1] + 
  irfs$irf$doil[,"dwell"][3]/irfs$irf$doil[,"doil"][1] +
  irfs$irf$doil[,"dwell"][4]/irfs$irf$doil[,"doil"][1] +
  irfs$irf$doil[,"dwell"][5]/irfs$irf$doil[,"doil"][1] +
  irfs$irf$doil[,"dwell"][6]/irfs$irf$doil[,"doil"][1] +
  irfs$irf$doil[,"dwell"][7]/irfs$irf$doil[,"doil"][1] +
  irfs$irf$doil[,"dwell"][8]/irfs$irf$doil[,"doil"][1] +
  irfs$irf$doil[,"dwell"][9]/irfs$irf$doil[,"doil"][1] +
  irfs$irf$doil[,"dwell"][10]/irfs$irf$doil[,"doil"][1] 

# Supply shock identifies demand elasticity:
# d q(t)/d p(t) = (d dwell(t)/d dwell(t))/(d doil(t)/d dwell(t)) = 
irfs$irf$dwell[,"dwell"][1]/irfs$irf$dwell[,"doil"][1]
irfs$irf$dwell[,"dwell"][2]/irfs$irf$dwell[,"doil"][2]
# Not well-identified, probably too big of an elasticity

# Long run elasticity: 
irfs$irf$dwell[,"dwell"][2]/irfs$irf$dwell[,"doil"][2] +
  irfs$irf$dwell[,"dwell"][3]/irfs$irf$dwell[,"doil"][2] +
  irfs$irf$dwell[,"dwell"][4]/irfs$irf$dwell[,"doil"][2] +
  irfs$irf$dwell[,"dwell"][5]/irfs$irf$dwell[,"doil"][2] +
  irfs$irf$dwell[,"dwell"][6]/irfs$irf$dwell[,"doil"][2] +
  irfs$irf$dwell[,"dwell"][7]/irfs$irf$dwell[,"doil"][2] +
  irfs$irf$dwell[,"dwell"][8]/irfs$irf$dwell[,"doil"][2] +
  irfs$irf$dwell[,"dwell"][9]/irfs$irf$dwell[,"doil"][2] +
  irfs$irf$dwell[,"dwell"][10]/irfs$irf$dwell[,"doil"][2]


####### Example: Kilian-like example for gas #######

kil = cbind(dwell,gdpi,dgas)
kil = cbind(dwell,bbk,dgas)
kil = cbind(dwell,dind,dgas)
colnames(kil) <- c("dwell","dind","dgas")
kil <- na.omit(kil)
plot(kil,xlab="")
summary(kil)

VARselect(kil,lag.max=13,type="none")
# FPE and AIC select 1 lags, same as below:
varkil = VAR(kil,p=1,type="none")
varkil = VAR(kil,lag.max=13,type="none",ic="AIC")
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
# 3 roots because n=3 variables, p=1 lags and companion form is n*p X n*p = 3
roots(varkil)
# joint Granger-causality test of oil on gas and oil on wells:
causality(varkil, cause="dgas")
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
# can define impulse = "dgas", response="dwell"
plot(irf(varkil,n.ahead=24,ortho=F))
plot(irf(varkil,n.ahead=24,ortho=F,impulse = "dind",response="dgas"))
plot(irf(varkil,n.ahead=24,ortho=T))
plot(irf(varkil,n.ahead=24,ortho=T,impulse = "dind",response="dgas"))
plot(irf(varkil,n.ahead=24,ortho=T,impulse = "dind",response="dgas",cumulative = T))
# from the above, a one-standard deviation shock in the oil price return leads to a cumulative 
# four unit increase in the changes in the drilling index
plot(fevd(varkil,n.ahead=24))

plot(irf(varkil,n.ahead=24,ortho=T,impulse = "dwell",response="dgas"))
plot(irf(varkil,n.ahead=24,ortho=T,impulse = "dind",response="dgas"))

plot(irf(varkil,n.ahead=24,ortho=T,response="dgas"))
plot(irf(varkil,n.ahead=24,ortho=T,response="dwell"))

irfs = irf(varkil,n.ahead=10,ortho=T)
irfs$irf$dwell
irfs$irf$dind
irfs$irf$dgas

# Aggregate demand shock identifies supply elasticity:
# d q(t+1)/d p(t) = (d dwell(t+1)/d dind(t))/(d dgas(t)/d dind(t)) = 
irfs$irf$dind[,"dwell"][2]/irfs$irf$dind[,"dgas"][1]
irfs$irf$dind[,"dwell"][3]/irfs$irf$dind[,"dgas"][2]

# Long run elasticity: 
  irfs$irf$dind[,"dwell"][3]/irfs$irf$dind[,"dgas"][2] +
  irfs$irf$dind[,"dwell"][4]/irfs$irf$dind[,"dgas"][2] +
  irfs$irf$dind[,"dwell"][5]/irfs$irf$dind[,"dgas"][2] +
  irfs$irf$dind[,"dwell"][6]/irfs$irf$dind[,"dgas"][2] +
  irfs$irf$dind[,"dwell"][7]/irfs$irf$dind[,"dgas"][2] +
  irfs$irf$dind[,"dwell"][8]/irfs$irf$dind[,"dgas"][2] +
  irfs$irf$dind[,"dwell"][9]/irfs$irf$dind[,"dgas"][2] +
  irfs$irf$dind[,"dwell"][10]/irfs$irf$dind[,"dgas"][2] 

# Precautionary demand shock identifies supply elasticity:
# d q(t+1)/d p(t) = (d dwell(t+1)/d dgas(t))/(d dgas(t)/d dgas(t)) = 
irfs$irf$dgas[,"dwell"][2]/irfs$irf$dgas[,"dgas"][1]

# Long run elasticity: 
irfs$irf$dgas[,"dwell"][2]/irfs$irf$dgas[,"dgas"][1] + 
  irfs$irf$dgas[,"dwell"][3]/irfs$irf$dgas[,"dgas"][1] +
  irfs$irf$dgas[,"dwell"][4]/irfs$irf$dgas[,"dgas"][1] +
  irfs$irf$dgas[,"dwell"][5]/irfs$irf$dgas[,"dgas"][1] +
  irfs$irf$dgas[,"dwell"][6]/irfs$irf$dgas[,"dgas"][1] +
  irfs$irf$dgas[,"dwell"][7]/irfs$irf$dgas[,"dgas"][1] +
  irfs$irf$dgas[,"dwell"][8]/irfs$irf$dgas[,"dgas"][1] +
  irfs$irf$dgas[,"dwell"][9]/irfs$irf$dgas[,"dgas"][1] +
  irfs$irf$dgas[,"dwell"][10]/irfs$irf$dgas[,"dgas"][1] 

# Supply shock identifies demand elasticity:
# d q(t)/d p(t) = (d dwell(t)/d dwell(t))/(d dgas(t)/d dwell(t)) = 
irfs$irf$dwell[,"dwell"][1]/irfs$irf$dwell[,"dgas"][1]
irfs$irf$dwell[,"dwell"][2]/irfs$irf$dwell[,"dgas"][2]
# Not well-identified, probably too big of an elasticity

# Long run elasticity: 
irfs$irf$dwell[,"dwell"][2]/irfs$irf$dwell[,"dgas"][2] +
  irfs$irf$dwell[,"dwell"][3]/irfs$irf$dwell[,"dgas"][2] +
  irfs$irf$dwell[,"dwell"][4]/irfs$irf$dwell[,"dgas"][2] +
  irfs$irf$dwell[,"dwell"][5]/irfs$irf$dwell[,"dgas"][2] +
  irfs$irf$dwell[,"dwell"][6]/irfs$irf$dwell[,"dgas"][2] +
  irfs$irf$dwell[,"dwell"][7]/irfs$irf$dwell[,"dgas"][2] +
  irfs$irf$dwell[,"dwell"][8]/irfs$irf$dwell[,"dgas"][2] +
  irfs$irf$dwell[,"dwell"][9]/irfs$irf$dwell[,"dgas"][2] +
  irfs$irf$dwell[,"dwell"][10]/irfs$irf$dwell[,"dgas"][2]

