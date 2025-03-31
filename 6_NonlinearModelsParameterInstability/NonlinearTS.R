##########################
# 1) download a few series of interest 
#     and merge them for a common time window of 01-03-2017 to 09-30-2021
# 2) investigate nonlinearity in the relationship

require(forecast)
require(quantmod)
require(caschrono)
require(texreg)
require(strucchange)
require(tsDyn)
require(CADFtest)


# 1) 
getSymbols("GC=F")   #gold futures price
getSymbols("DFII10",src="FRED")             #10y TIPS const maturity 2003-01-02
getSymbols("T10YIE",src="FRED")             #10y breakeven inflation 2003-01-02
getSymbols("MCOILWTICO",src="FRED")
getSymbols("IPN213111N",src="FRED") # Oil & gas drilling index
getSymbols("MHHNGSP",src="FRED") # Henry Hub
getSymbols("DFF",src="FRED") #Daily effective federal funds rate


data = merge.xts(MHHNGSP,MCOILWTICO,IPN213111N,join="inner")
plot(data)
CADFtest(data$MHHNGSP,criterion=c("AIC"),type="drift")
CADFtest(data$IPN213111N,criterion=c("AIC"),type="drift")
gas <- ts(na.omit(data$MHHNGSP),freq=12,start=1997)
oil <- ts(na.omit(data$MCOILWTICO),freq=12,start=1997)
drl <- ts(na.omit(data$IPN213111N),freq=12,start=1997)


grd <- merge.xts(`GC=F`$`GC=F.Adjusted`, DFII10$DFII10, 
                 T10YIE$T10YIE, DTWEXBGS, SP500$SP500, DFF$DFF,
                 all=TRUE, fill=NA, retclass="xts")

colnames(grd) <- c("gld","tip", "bei","twd","sp5","dff")

#SELECT START AND END DATE
startdate <- "2017-01-03"  
enddate   <- "2021-09-30"    

grd <- window(grd, start = startdate, end = enddate)
grd <- na.omit(grd)

chartSeries(grd$gld)
chartSeries(grd$tip)
chartSeries(grd$bei)
chartSeries(grd$dff)

bei <- ts(na.omit(grd$bei),freq=252,start=2017)
gld <- ts(na.omit(grd$gld),freq=252,start=2017)
tip <- ts(na.omit(grd$tip),freq=252,start=2017)
dff <- ts(na.omit(grd$dff),freq=252,start=2017)


dlgld  = ts(na.omit(diff(log(grd$gld))),freq=252,start = 2017)
dtip   = ts(na.omit(diff(grd$tip)),freq=252,start = 2017)
dbei   = ts(na.omit(diff(grd$bei)),freq=252,start = 2017)
ddff   = ts(na.omit(diff(grd$dff)),freq=252,start = 2017)


# 2) Estimate and investigate thresholds

selectSETAR(bei,m=3)
bei.setar <- setar(bei,m=1,thDelay=0,th=1.62)
summary(bei.setar)
setarTest(bei,m=1,thDelay = 0)
plot(bei.setar)

selectSETAR(bei,m=3,thDelay = 2)
bei.setar2 <- setar(bei,mL=3,mH=1,thDelay=2,th=1.62)
summary(bei.setar2)
setarTest(bei,m=3,thDelay = 2)
plot(bei.setar2)


selectSETAR(gld,m=3)
gld.setar <- setar(gld,mL=3,mH=1,thDelay=0,th=1481.7)
summary(gld.setar)
setarTest(gld,m=3,thDelay = 0)
plot(gld.setar)

#selectSETAR(gld,m=3,thVar = tip, thDelay = c(1,2))
gld.setar.tip <- setar(gld,mL=1,mH=3,thVar = tip)
summary(gld.setar.tip)
plot(gld.setar.tip)
gld.setar.dff <- setar(gld,mL=1,mH=3,thVar = dff)
summary(gld.setar.dff)
plot(gld.setar.dff)

gld.setar.tip2 <- setar(gld,mL=1,mH=3,mM=1,thVar = tip,nthresh = 2)
summary(gld.setar.tip2)
plot(gld.setar.tip2)
gld.setar.dff2 <- setar(gld,mL=1,mH=3,mM=1,thVar = dff,nthresh=2)
summary(gld.setar.dff2)
plot(gld.setar.dff2)

selectSETAR(gas,m=4,thDelay = 0:3)
gas.setar <- setar(gas,mL=2,mH=1,th=4.69)
gas.setar <- setar(gas,mL=3,mH=3,thDelay=0:2)
summary(gas.setar)
plot(gas.setar)
setarTest(gas,m=3,thDelay = 0:2)

# 3) Estimate smooth transition AR (STAR)

# how many regimes?
gas.star <- star(gas,m=3,thDelay = 2)
summary(gas.star)

# or add one at a time
gas.star2 <- star(gas,m=3,noRegimes = 2,thDelay = 2)
gas.star.3 <- addRegime(gas.star2)

# LSTAR two regimes
gas.lstar <- lstar(gas,m=3, thDelay = 2)
summary(gas.lstar)
plot(gas.lstar)
predict(gas.lstar,n.ahead=5)


## Compare models ------------------

modlist <- list()
modlist[["linear"]] <- linear(gas,m=3)
modlist[["setar"]] <- setar(gas,m=3,thDelay = 2)
modlist[["lstar"]] <- lstar(gas,m=3,thDelay=2,starting.control=list(gammaInt=c(1,100)))
sapply(modlist,AIC)
sapply(modlist,BIC)
sapply(modlist,MAPE)
summary(modlist[["lstar"]])

# Forecast over known data
set.seed(12345)
mod.test <- list()
gas.train <- window(gas,end = 2020+11/12)
gas.test <- window(gas,start=2021)
mod.test[["linear"]] <- linear(gas.train,m=3)
mod.test[["setar"]] <- setar(gas.train,m=3,thDelay = 2)
mod.test[["lstar"]] <- lstar(gas.train,m=3,thDelay=2,trace=FALSE,starting.control=list(gammaInt=c(1,100)))
frc.test <- lapply(mod.test, predict, n.ahead = 30)
plot(gas.test, ylim = range(gas))
for (i in 1:length(frc.test)) 
  lines(frc.test[[i]], lty = i + 1, col = i + 1)
  legend(2021, 12, lty = 1:(length(frc.test) + 1), 
       col = 1:(length(frc.test) + 1), 
       legend = c("observed", names(frc.test)))

set.seed(12345)
mod.test <- list()
gas.train <- window(gas,end = 2022+11/12)
gas.plot <- window(gas,start=2022)
mod.test[["linear"]] <- linear(gas.train,m=3)
mod.test[["setar"]] <- setar(gas.train,m=3,thDelay = 2)
mod.test[["lstar"]] <- lstar(gas.train,m=3,thDelay=2,trace=FALSE,starting.control=list(gammaInt=c(1,100)))
frc.test <- lapply(mod.test, predict, n.ahead = 15)
plot(gas.plot, ylim = range(gas))
for (i in 1:length(frc.test)) 
  lines(frc.test[[i]], lty = i + 1, col = i + 1)
  legend(2022, 12, lty = 1:(length(frc.test) + 1), 
       col = 1:(length(frc.test) + 1), 
       legend = c("observed", names(frc.test)))

# Forecast from different points in time
set.seed(12345)
mod.test <- list()
gas.train <- window(gas,end = 2019+11/12)
gas.plot <- window(gas,start=2020)
mod.test[["linear"]] <- linear(gas.train,m=3)
mod.test[["setar"]] <- setar(gas.train,m=3,thDelay = 2)
mod.test[["lstar"]] <- lstar(gas.train,m=3,thDelay=2,trace=FALSE,starting.control=list(gammaInt=c(1,100)))
frc.test <- lapply(mod.test, predict, n.ahead = 36)
plot(gas.plot, ylim = range(gas))
for (i in 1:length(frc.test)) 
  lines(frc.test[[i]], lty = i + 1, col = i + 1)
  legend(2022, 12, lty = 1:(length(frc.test) + 1), 
         col = 1:(length(frc.test) + 1), 
         legend = c("observed", names(frc.test)))

mod.test2 <- list()
gas.train2 <- window(gas,end = 2008+11/12)
#gas.test2 <- window(gas,start=2009+0/12)
mod.test2[["linear"]] <- linear(gas.train2,m=3)
mod.test2[["setar"]] <- setar(gas.train2,m=3,thDelay = 2)
mod.test2[["lstar"]] <- lstar(gas.train2,m=3,thDelay=2,trace=FALSE,starting.control=list(gammaInt=c(1,100)))
frc.test2 <- lapply(mod.test2, predict, n.ahead = 36)

plot(gas, ylim = range(gas))
for (i in 1:length(frc.test)) 
  lines(frc.test[[i]], lty = i + 1, col = i + 1)
  legend(2016, 13, lty = 1:(length(frc.test) + 1), 
       col = 1:(length(frc.test) + 1), 
       legend = c("observed", names(frc.test)))
for (i in 1:length(frc.test2)) 
  lines(frc.test2[[i]], lty = i + 1, col = i + 1)
  legend(1999, 13, lty = 1:(length(frc.test2) + 1), 
       col = 1:(length(frc.test2) + 1), 
       legend = c("observed", names(frc.test2)))

