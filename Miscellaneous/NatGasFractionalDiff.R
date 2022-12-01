require(quantmod)
require(forecast)
require(CADFtest)
require(fracdiff)
    getSymbols("MHHNGSP",src="FRED")
  gas <- MHHNGSP$MHHNGSP
  gas <- gas[paste("1997-01-01","2017-08-01",sep="/")]
  lgas <- log(gas)
  chartSeries(gas,theme="white")
  chartSeries(lgas,theme="white")
  
  gasgr <- diff(lgas) # price growth (returns)
  gasgr12 <- diff(lgas,lag=12,differences=1) # seasonal difference of prices
  gasseas <- diff(gasgr,lag=12,differences=1) # seasonal difference of returns
  par(mfcol=c(2,2))
  acf(lgas)
  acf(na.omit(gasgr))
  acf(na.omit(gasgr12))
  acf(na.omit(gasseas))
  par(mfcol=c(1,1))
  
  CADFtest(lgas,type="none")
x=  fracdiff(lgas,nar=1,nma=1)
summary(x)
tsdisplay(residuals(x),gof=25)
y=forecast(x)
plot(y)

fit_no_holds <- fracdiff(lgas[-c(201:250)],nar=1,nma=1)
fcast_no_holds <- forecast(fit_no_holds,h=80)
plot(fcast_no_holds,main=" ",include=150)
lines(ts(lgas))

fit_no_holds <- arima(lgas[-c(201:250)],order=c(1,0,1),include.mean=T)
fcast_no_holds <- forecast(fit_no_holds,h=80)
plot(fcast_no_holds,main=" ",include=150)
lines(ts(lgas))

fit_no_holds <- Arima(lgas[-c(201:250)],order=c(1,1,1),include.constant = T)
fcast_no_holds <- forecast(fit_no_holds,h=80)
plot(fcast_no_holds,main=" ",include=150)
lines(ts(lgas))
  