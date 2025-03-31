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
flrdata <- read.csv("C:/Users/gilbe/Dropbox/Econometrics/MethFlare.csv", header=TRUE, sep=",")


flareIV <- felm(log_flare ~ day + m2 + m3 + m4 + m5 + m6 + m7 + m8 + m9 + m10 + m11 + m12 | 0 |
                   (diff_day ~ wshocknonzint) | 0 , 
                 data = flrdata)
summary(flareIV)
# can only use HC, not HAC standard erros with felm: 
coeftest(flareIV,vcov = vcovHC)
# Homoskedastic F stat
flareIV$stage1$iv1fstat
# Heteroskedastic F stat
flareIV$stage1$rob.iv1fstat
# First stage F-stat for IV. 

# Using fixest package:
flr_feols = feols(log_flare ~ day + m2 + m3 + m4 + m5 + m6 + m7 + m8 + m9 + m10 + m11 + m12 | 
                    diff_day ~ wshocknonzint, flrdata, panel.id = "day")
summary(flr_feols)
fitstat(flr_feols,show_types = TRUE)
# Homoskedastic F stats
fitstat(flr_feols, "ivf1")
fitstat(flr_feols, "ivwald1")
# Heteroskedastic F stat
fitstat(flr_feols, "ivwald1", se = "heter")
fitstat(flr_feols, "kpr", se = "heter")
# HAC F stat
fitstat(flr_feols, "kpr", vcov=NW(lag=39)~day)

#bw = bwNeweyWest(flr_feols, kernel = c("Bartlett"))


# Estimate with HC from the beginning
flr_feols = feols(log_flare ~ day + m2 + m3 + m4 + m5 + m6 + m7 + m8 + m9 + m10 + m11 + m12 | 
                    diff_day ~ wshocknonzint, flrdata, NW(lag=39)~day)
summary(flr_feols)
fitstat(flr_feols, "ivwald", se = "heter")
fitstat(flr_feols, "kpr")
fitstat(flr_feols, "kpr", vcov=NW(lag=39)~day)


# IV without HAC standard errors: 
flr_feols = feols(log_flare ~ day  | diff_day ~ wshocknonzint, flrdata)
summary(flr_feols)
fitstat(flr_feols, "kpr")
acf(residuals(flr_feols),lag.max=60)

# IV with HAC standard errors: 
flr_feols = feols(log_flare ~ day  | diff_day ~ wshocknonzint, flrdata, NW(lag=45) ~day)
summary(flr_feols)
fitstat(flr_feols, "kpr")

########## Showing static SVAR: ##########
eq2 = lm(diff_day ~ day + wshocknonzint,data=flrdata)
eq3 = lm(log_flare ~ day + wshocknonzint + diff_day, data=flrdata)
summary(eq2)
summary(eq3)
# Flaring response to the price differential: 
(eq3$coefficients[3] + eq3$coefficients[4]*eq2$coefficients[3])/(eq2$coefficients[3])

# Adding jobs: 
eq2 = lm(diff_day ~ day + wshocknonzint + tot_day_jobs,data=flrdata)
eq3 = lm(log_flare ~ day + wshocknonzint + diff_day + tot_day_jobs, data=flrdata)
summary(eq2)
summary(eq3)
# Flaring response to the price differential: 
(eq3$coefficients[3] + eq3$coefficients[4]*eq2$coefficients[3])/(eq2$coefficients[3])


####### Example: 3-equation Flaring #######

kil = as.data.frame(cbind(flrdata$wshocknonzint,flrdata$diff_day,flrdata$log_flare))
kil <- na.omit(kil)
colnames(kil) = c("weath","diff","flare")
kil <- ts(kil,start=2015,freq=365)
plot(kil,xlab="")
summary(kil)

VARselect(kil,lag.max=20,type="both")
# FPE and AIC select 2 lags, same as below:
varkil = VAR(kil,lag.max=20,type="both",ic="SC")
varkil = VAR(kil,p=1,type="both")
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
summary(varkil,equation="flare")
plot(varkil)
plot(varkil,name="flare")
# 6 roots because n=3 variables, p=2 lags and companion form is n*p X n*p = 6
roots(varkil)
# joint Granger-causality test of oil on gas and oil on wells:
causality(varkil, cause="weath")
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
plot(irf(varkil,n.ahead=24,ortho=F,impulse = "diff",response="flare"))
plot(irf(varkil,n.ahead=24,ortho=T))
plot(irf(varkil,n.ahead=24,ortho=T,impulse = "diff",response="flare"))
plot(irf(varkil,n.ahead=24,ortho=T,impulse = "diff",response="flare",cumulative = T))
# from the above, a one-standard deviation shock in the oil price return leads to a cumulative 
# four unit increase in the changes in the drilling index
plot(fevd(varkil,n.ahead=24))

plot(irf(varkil,n.ahead=24,ortho=T,impulse = "weath",response="flare"))
plot(irf(varkil,n.ahead=24,ortho=T,impulse = "diff",response="flare"))

plot(irf(varkil,n.ahead=24,ortho=T,response="flare"))
plot(irf(varkil,n.ahead=24,ortho=T,response="diff"))

irfs = irf(varkil,n.ahead=10,ortho=T)
irfs$irf$weath
irfs$irf$diff
irfs$irf$flare

# Demand shock for transportation out of the Permian identifies impact of congestion on flaring
# dflare/ddiff = (dflr/dwe)/(ddif/dwe) = 
irfs$irf$weath[,"flare"][1]/irfs$irf$weath[,"diff"][1]
irfs$irf$weath[,"flare"][2]/irfs$irf$weath[,"diff"][2]
irfs$irf$weath[,"flare"][5]/irfs$irf$weath[,"diff"][2]
irfs$irf$weath[,"flare"][7]/irfs$irf$weath[,"diff"][2]
irfs$irf$weath[,"flare"][10]/irfs$irf$weath[,"diff"][2]

# cumulative: 
irfs$irf$weath[,"flare"][2]/irfs$irf$weath[,"diff"][2] + 
  irfs$irf$weath[,"flare"][3]/irfs$irf$weath[,"diff"][2] +
  irfs$irf$weath[,"flare"][4]/irfs$irf$weath[,"diff"][2] +
  irfs$irf$weath[,"flare"][5]/irfs$irf$weath[,"diff"][2] +
  irfs$irf$weath[,"flare"][6]/irfs$irf$weath[,"diff"][2] +
  irfs$irf$weath[,"flare"][7]/irfs$irf$weath[,"diff"][2] 

# cumulative: 
irfs$irf$weath[,"flare"][2]/irfs$irf$weath[,"diff"][2] + 
  irfs$irf$weath[,"flare"][3]/irfs$irf$weath[,"diff"][3] +
  irfs$irf$weath[,"flare"][4]/irfs$irf$weath[,"diff"][4] +
  irfs$irf$weath[,"flare"][5]/irfs$irf$weath[,"diff"][5] +
  irfs$irf$weath[,"flare"][6]/irfs$irf$weath[,"diff"][6] +
  irfs$irf$weath[,"flare"][7]/irfs$irf$weath[,"diff"][7] 

# Transportation capacity supply shock identifies impact of congestion on flaring
# dflare/ddiff = (dflr/ddif)/(ddif/ddif) = 
irfs$irf$diff[,"flare"][1]/irfs$irf$diff[,"diff"][1]
irfs$irf$diff[,"flare"][2]/irfs$irf$diff[,"diff"][2]

# How much does flaring today alleviate the shadow price of pipelines in
# response to a weather demand shock?
irfs$irf$weath[,"diff"][1]/irfs$irf$weath[,"flare"][1]

# using otherwise unexpected flaring shocks
# ddiff/dflare = (ddif/dflr)/(dflr/dflr) = 
irfs$irf$flare[,"diff"][1]/irfs$irf$flare[,"flare"][1]
irfs$irf$flare[,"diff"][2]/irfs$irf$flare[,"flare"][2]
irfs$irf$flare[,"diff"][5]/irfs$irf$flare[,"flare"][1]


######## Example: 4-equation Flaring (include fracking jobs) ########

kil = as.data.frame(cbind(flrdata$wshocknonzint,flrdata$tot_day_jobs,
                          flrdata$diff_day,flrdata$log_flare,
            flrdata$m2,flrdata$m3,flrdata$m4,flrdata$m5,flrdata$m6,
            flrdata$m7,flrdata$m8,flrdata$m9,flrdata$m10,flrdata$m11,flrdata$m12))
kil <- na.omit(kil)
colnames(kil) <- c("wshocknonzint","tot_day_jobs","diff_day","log_flare",
                   "m2","m3","m4","m5","m6","m7","m8","m9","m10","m11","m12")

kilseas <- ts(cbind(kil$m2,kil$m3,kil$m4,kil$m5,kil$m6,kil$m7,kil$m8,kil$m9
                    ,kil$m10,kil$m11,kil$m12), start=2015, freq=365)
colnames(kilseas) = c("m2","m3","m4","m5","m6","m7","m8","m9","m10","m11","m12")

kil = ts(cbind(kil$wshocknonzint,kil$tot_day_jobs,kil$diff_day,kil$log_flare), start=2015, freq=365)
colnames(kil) = c("weath","jobs","diff","flare")



plot(kil,xlab="")
summary(kil)

VARselect(kil,lag.max=20,type="both",exogen=kilseas)

varkil = VAR(kil,lag.max=20,type="both",ic="SC", exogen=kilseas)
varkil = VAR(kil,lag.max=20,type="both",ic="SC", exogen=fourier(kil,K=c(4)))
varkil = VAR(kil,p=1,type="both", exogen=kilseas)
varkil = VAR(kil,p=1,type="both")

future_fourier <- fourier(kil, K = 4, h = 30)
colnames(future_fourier) <- c("S1.365","C1.365","S2.365","C2.365","S3.365","C3.365","S4.365","C4.365")
forecast_values <- predict(varkil, n.ahead = 30, dumvar = future_fourier)


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
summary(varkil,equation="flare")
plot(varkil)
plot(varkil,name="flare")
# 4 roots because n=4 variables, p=1 lags and companion form is n*p X n*p = 4
roots(varkil)
# joint Granger-causality test of weather shocks on others:
causality(varkil, cause="weath")
# diagnostics:
normality.test(varkil)
# rejects normality test for both skewness and kurtosis!
arch.test(varkil,lags.multi=4)
plot(arch.test(varkil,lags.multi=4))
# significant ARCH effects, could explain failure of normality.
serial.test(varkil,lags.pt = 16)
# there is residual serial correlation, which is not good.
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
# can define impulse = "diff", response="flare"
plot(irf(varkil,n.ahead=24,ortho=F))
plot(irf(varkil,n.ahead=24,ortho=F,impulse = "diff",response="flare"))
plot(irf(varkil,n.ahead=24,ortho=T))
plot(irf(varkil,n.ahead=24,ortho=T,impulse = "diff",response="flare"))
plot(irf(varkil,n.ahead=24,ortho=T,impulse = "diff",response="flare",cumulative = T))
# from the above, a one-standard deviation shock in the price differential leads to a cumulative 
# flaring 80 percent higher than it would have been over 24 days.
plot(fevd(varkil,n.ahead=24))


plot(irf(varkil,n.ahead=24,ortho=F))
plot(irf(varkil,n.ahead=24,ortho=F,impulse = "weath",response="flare"))
plot(irf(varkil,n.ahead=24,ortho=T))
plot(irf(varkil,n.ahead=24,ortho=T,impulse = "weath",response="flare"))

plot(irf(varkil,n.ahead=24,ortho=T,impulse = "jobs",response="flare"))
plot(irf(varkil,n.ahead=24,ortho=T,impulse = "event",response="flare"))


plot(irf(varkil,n.ahead=24,ortho=T,impulse = "weath",response="diff"))
plot(irf(varkil,n.ahead=24,ortho=T,impulse = "jobs",response="diff"))
plot(irf(varkil,n.ahead=24,ortho=T,impulse = "event",response="diff"))
plot(irf(varkil,n.ahead=24,ortho=T,impulse = "flare",response="diff"))

irfs = irf(varkil,n.ahead=10,ortho=T)
irfs$irf$weath
irfs$irf$jobs
irfs$irf$diff
irfs$irf$flare

# Demand shock for transportation out of the Permian identifies impact of congestion on flaring
# dflare/ddiff = (dflr/dwe)/(ddif/dwe) = 
irfs$irf$weath[,"flare"][1]/irfs$irf$weath[,"diff"][1]
irfs$irf$weath[,"flare"][2]/irfs$irf$weath[,"diff"][2]
irfs$irf$weath[,"flare"][5]/irfs$irf$weath[,"diff"][2]
irfs$irf$weath[,"flare"][7]/irfs$irf$weath[,"diff"][2]
irfs$irf$weath[,"flare"][10]/irfs$irf$weath[,"diff"][2]

# cumulative: 
irfs$irf$weath[,"flare"][2]/irfs$irf$weath[,"diff"][2] + 
  irfs$irf$weath[,"flare"][3]/irfs$irf$weath[,"diff"][2] +
  irfs$irf$weath[,"flare"][4]/irfs$irf$weath[,"diff"][2] +
  irfs$irf$weath[,"flare"][5]/irfs$irf$weath[,"diff"][2] +
  irfs$irf$weath[,"flare"][6]/irfs$irf$weath[,"diff"][2] +
  irfs$irf$weath[,"flare"][7]/irfs$irf$weath[,"diff"][2] 
  
# cumulative: 
irfs$irf$weath[,"flare"][2]/irfs$irf$weath[,"diff"][2] + 
  irfs$irf$weath[,"flare"][3]/irfs$irf$weath[,"diff"][3] +
  irfs$irf$weath[,"flare"][4]/irfs$irf$weath[,"diff"][4] +
  irfs$irf$weath[,"flare"][5]/irfs$irf$weath[,"diff"][5] +
  irfs$irf$weath[,"flare"][6]/irfs$irf$weath[,"diff"][6] +
  irfs$irf$weath[,"flare"][7]/irfs$irf$weath[,"diff"][7] 

# Demand shock from gas supply for transportation out of the Permian identifies impact of congestion on flaring
# dflare/ddiff = (dflr/djobs)/(ddif/djobs) = 
irfs$irf$jobs[,"flare"][1]/irfs$irf$jobs[,"diff"][1]
irfs$irf$jobs[,"flare"][2]/irfs$irf$jobs[,"diff"][2]
irfs$irf$jobs[,"flare"][5]/irfs$irf$jobs[,"diff"][2]
irfs$irf$jobs[,"flare"][7]/irfs$irf$jobs[,"diff"][2]
irfs$irf$jobs[,"flare"][10]/irfs$irf$jobs[,"diff"][2]

irfs$irf$jobs[,"flare"][3]/irfs$irf$jobs[,"diff"][3]

# Transportation capacity supply shock identifies impact of congestion on flaring
# dflare/ddiff = (dflr/ddif)/(ddif/ddif) = 
irfs$irf$diff[,"flare"][1]/irfs$irf$diff[,"diff"][1]
irfs$irf$diff[,"flare"][2]/irfs$irf$diff[,"diff"][2]

# How much does flaring today alleviate the shadow price of pipelines in
# response to a weather demand or jobs shock?
irfs$irf$weath[,"diff"][1]/irfs$irf$weath[,"flare"][1]
irfs$irf$jobs[,"diff"][1]/irfs$irf$jobs[,"flare"][1]

# using otherwise unexpected flaring shocks
# ddiff/dflare = (ddif/dflr)/(dflr/dflr) = 
irfs$irf$flare[,"diff"][1]/irfs$irf$flare[,"flare"][1]
irfs$irf$flare[,"diff"][2]/irfs$irf$flare[,"flare"][2]
irfs$irf$flare[,"diff"][5]/irfs$irf$flare[,"flare"][1]


########## Other VAR options #############

kil = as.data.frame(cbind(flrdata$wshocknonzint,flrdata$tot_day_jobs,flrdata$diff_day,flrdata$log_flare))
kil <- na.omit(kil)
colnames(kil) <- c("wshocknonzint","tot_day_jobs","diff_day","log_flare")
colnames(kil) = c("weath","jobs","diff","flare")
kil <- ts(kil,start=2015,freq=365)



kil = as.data.frame(cbind(flrdata$wshocknonzint,flrdata$tot_day_jobs,flrdata$unplanned_events_day2,flrdata$diff_day,flrdata$log_flare))
kil <- na.omit(kil)
colnames(kil) = c("weath","jobs","event","diff","flare")
kil <- ts(kil,start=2015,freq=365)
VARselect(kil,lag.max=20,type="both")
varkil = VAR(kil,p=1,type="both")


kil = as.data.frame(cbind(flrdata$wshocknonzint,flrdata$tot_day_jobs,flrdata$diff_day,flrdata$log_flare,
                          flrdata$log_meth, flrdata$m2,flrdata$m3,flrdata$m4,flrdata$m5,flrdata$m6,
                          flrdata$m7,flrdata$m8,flrdata$m9,flrdata$m10,flrdata$m11,flrdata$m12))
kil <- na.omit(kil)
colnames(kil) <- c("wshocknonzint","tot_day_jobs","diff_day","log_flare","lmethe",
                   "m2","m3","m4","m5","m6","m7","m8","m9","m10","m11","m12")

kilseas <- ts(cbind(kil$m2,kil$m3,kil$m4,kil$m5,kil$m6,kil$m7,kil$m8,kil$m9
                    ,kil$m10,kil$m11,kil$m12), start=2015, freq=365)
colnames(kilseas) = c("m2","m3","m4","m5","m6","m7","m8","m9","m10","m11","m12")

kil <- cbind(kil$wshocknonzint,kil$tot_day_jobs,kil$diff_day,kil$log_flare,kil$lmethe)
colnames(kil) = c("weath","jobs","diff","flare","methe")
kil <- ts(kil,start=2015,freq=365)
VARselect(kil,lag.max=20,type="both",exogen=kilseas)
varkil = VAR(kil,p=1,type="both",exogen=kilseas)

###### Decomposing structural part ##############

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
plot(irf(svarkil,n.ahead=24,ortho=T,impulse = "diff",response="flare"))

