# Preamble --------------

# A program to plot heat maps and histograms
# Packages we will use

# install.packages("quantmod")
library(quantmod)
# install.packages("ggplot2")
library(ggplot2)
# install.packages("plyr")
library(plyr)
#install.packages("epitools")
library(epitools)
#install.packages("gplots")
library(gplots)

#install.packages("reshape2")
library(reshape2)
#install.packages("MASS")
library(MASS)      
#install.packages("RColorBrewer")
library(RColorBrewer)

# Alternatively, use pacman:
#if(!require(pacman)) install.packages("pacman")
#pacman::p_load(quantmod,ggplot2,plyr,epitools,gplots,reshape2,MASS,RColorBrewer)

# store plotting margin defaults
.pardefault <- par()

# Get price data and calculate returns --------------

# download Market Indices, calculate returns, and chart basic graphs

getSymbols("XLE",from="2000-01-03") # default pulls from yahoo finance
                                    # could ad src="google" or src="FRED". 
                                    # default time span is 1/3/07 to present
# XLE is symbol for the SPDR Energy ETF
chartSeries(XLE,type=c("line"),theme="white",TA=NULL)
XLE.dlrtn = 100*dailyReturn(XLE,leading=FALSE,type='log')   # daily log return
XLE.mlrtn = 100*monthlyReturn(XLE,leading=FALSE,type='log') # monthly log return

getSymbols("SPY",from="2000-01-03")   
# SPY is the symbol for the S&P 500 Index
chartSeries(SPY,type=c("line"),theme="white",TA=NULL)
SPY.dlrtn = 100*dailyReturn(SPY,leading=FALSE,type='log')
SPY.mlrtn = 100*monthlyReturn(SPY,leading=FALSE,type='log')

qplot(SPY.dlrtn,XLE.dlrtn)

# Change formatting to "ts" object -----------------

# getSymbols/quantmod stores the data as a "zoo" object. 
# A "ts" object is sometimes easier to work with, 
# so transform returns data to "ts"
spy=na.omit(ts(SPY.dlrtn))
xle=na.omit(ts(XLE.dlrtn))
df=cbind(spy,xle)

# Construct the heat map and histograms -----------------

# make the histogram objects that will go along the axes
hspy <- hist(spy, breaks=25, plot=F)
hxle <- hist(xle, breaks=25, plot=F)

# now create the combination heat map/histogram graph
top = max(hspy$counts,hxle$counts)
hd=hist2d(df,nbins=50,FUN=function(x)
  log(length(x)))

par(mar=c(3,3,1,1))
layout(matrix(c(2,0,1,3),2,2,byrow=T),c(3,1), c(1,3))
hd=hist2d(df,nbins=50,FUN=function(x)
  log(length(x)))
par(mar=c(0,2,1,0))
barplot(hspy$counts, axes=F, ylim=c(0, top), space=0, col='red')
par(mar=c(2,0,0.5,1))
barplot(hxle$counts, axes=F, xlim=c(0, top), space=0, col='red', horiz=T)

par(.pardefault) # restore the previous graphical layout


# Alternative graphing approach ----------------

# Define extent of X and Y graph axes
tab=dcast(df,cut(spy,breaks=c(-15,-10,-5,0,5,10,15))~cut(xle,breaks=c(-20,-10,0,10,20)))

# choose color pallette
rf <- colorRampPalette(rev(brewer.pal(11,'Spectral')))
r <- rf(32)

# make heat map image
hd2=kde2d(spy,xle,n=500)
image(hd2,col=r)

# combine heat map with histograms
top = max(hspy$counts,hxle$counts)
hd3=kde2d(spy,xle,n=200)
par(mar=c(3,3,1,1))
layout(matrix(c(2,0,1,3),2,2,byrow=T),c(3,1), c(1,3))
image(hd3, col=r) #plot the image
par(mar=c(0,2,1,0))
barplot(hspy$counts, axes=F, ylim=c(0, top), space=0, col='red')
par(mar=c(2,0,0.5,1))
barplot(hxle$counts, axes=F, xlim=c(0, top), space=0, col='red', horiz=T)

par(.pardefault)






