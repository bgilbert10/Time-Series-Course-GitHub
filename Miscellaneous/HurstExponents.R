# Hurst exponents vignette
# Estimate time-varying Hurst exponents for two assets, see how they correlate

# borrowing a bit from here: 
# https://stackoverflow.com/questions/26803471/rolling-hurst-exponent
# https://www.r-bloggers.com/2022/08/hurst-exponent-using-r-code/amp/ 

# use pracma package, hurstexp function to estimate hurst exponent
# use zoo package for rollapply() function to estimate on a rolling window. 

# install these if not already installed: 
#install.packages("pracma")
#install.packages("zoo)
library(pracma) # hurstexp
library(quantmod) # getSymbols
library(xts)
require(zoo)

# S&P500 ----------------

## Organize data --------------
# read S&P 500 index and calculate daily returns


sdate <- as.Date("1999-12-31")
edate <- as.Date("2022-08-01")
getSymbols("^GSPC", from=sdate, to=edate)

price <- GSPC[,6]
return <- dailyReturn(price)
df.stock <- cbind(time=time(price), 
                  as.data.frame(price), 
                  as.data.frame(return))
rownames(df.stock) <- NULL
colnames(df.stock) <- c("time", "SPC.price", "SPC.return")


## Estimate Hurst exponents ---------------
# Using R package simply
# see Empirical Hurst exponent


# notice there are multiple methods of calculating hurst exponent
# I have no opinion about which is best. 
# Using "corrected empirical hurst exponent" in the rolling loop below
h = hurstexp(return)
h

# estimate hurst 100 days at a time
z <- rollapply(return,100, function(y){
    h = hurstexp(y)$Hal
    h
  } )

# the first 100 days are blank, so omit them
z<- na.omit(z)
# S&P mostly a random walk with hurst about 0.54 most of the time, btw 0.5 and 0.6. 
chartSeries(z)
mean(z)

# Gold futures --------------

## Organize data -------------------

# try with gold futures for "fun"
getSymbols("GC=F", from=sdate, to=edate)
gold = na.omit(`GC=F`)

g.price <- gold[,6]
g.return <- dailyReturn(g.price)
df.gold <- cbind(time=time(g.price), 
                  as.data.frame(g.price), 
                  as.data.frame(g.return))
rownames(df.gold) <- NULL
colnames(df.gold) <- c("time", "gold.price", "gold.return")

## Estimate Hurst exponents ----------------

g = hurstexp(g.return)
g

# estimate hurst 100 days at a time
zg <- rollapply(g.return,100, function(y){
  g = hurstexp(y)$Hal
  g
} )

# the first 100 days are blank, so omit them
zg<- na.omit(zg)
# gold seems also mostly a random walk with hurst around 0.55 most of the time. 
chartSeries(zg)
mean(zg)

# Compare Hurst exponents from two assets -----------------

# compare them. There appears to be no relationship
z.all = merge.xts(z,zg,join="inner")
z.all = na.omit(z.all)
colnames(z.all) <- c("h.spy","h.gld")

## Visualize comparison ------------------

# no obvious comovement in hurst exponents
plot(z.all)

# scatter plot doesn't seem to show much
df.z <- as.data.frame(z.all)
plot(df.z$h.spy,df.z$h.gld)

# the hurst exponents themselves don't appear stationary
acf(z.all$h.spy)
acf(z.all$h.gld)

## Quantify comparison --------------

# the residuals from a regression of gold H on s&P h also not stationary
# (tell-tale sign of spurious regression - regression is meaningless)
summary(lm(h.gld ~ h.spy,data=z.all))
acf(residuals(lm(h.gld ~ h.spy,data=z.all)))

# looking at first differences of H, also no relationship. 
acf(residuals(lm(diff(h.gld) ~ diff(h.spy),data=z.all)))
summary(lm(diff(h.gld) ~ diff(h.spy),data=z.all))


