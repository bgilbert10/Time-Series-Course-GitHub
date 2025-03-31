## Program to demonstrate instrumental variables regression using 4 packages
## Example with natural gas supply and demand in California
## The AER, sem, lfe packages all support IV/2SLS regression
## The systemfit package supports IV/2SLS in a system of equations, sometimes
## known as "three stage least squares" or 3SLS. 

## We will use insight from Roberts & Schlenker 2013 and Hausman & Kellogg 2015
## Weather shocks in California are demand shocks and can be IVs for supply
## Weather shocks in states outside California are supply shocks to California. 
## This is because those outside Californnia shocks shift demand in those states, 
## reducing the supply available to California. 
## Because these are California supply shifters, they are IVs for CA demand. 
## The cumulative non-California weather shocks over a year also affect 
## inventories, which is an exogenous price instrument as well. 

## For weather shocks, we use Heating Degree Days and Cooling Degree Days
## The distance of temperature from a neutral 65F, multiplied by the number of
## days in the month and weighted by population. 
## This tells us how much either heating or cooling demand in a region we might
## expect over that month. 

# install.packages("systemfit")
require(systemfit)  
# install.packages("AER")
require(AER)
# install.packages("sem)
require(sem)
# install.packages("lfe")
require(lfe)

require(texreg)
require(lmtest)
require(sandwich)

# Import regional heating and cooling degree days. 
# Taken from American Gas Association
# https://www.aga.org/knowledgecenter/facts-and-data/annual-statistics/weekly-and-monthly-statistics
# Natural gas price and quantity data.
# citygate price from https://www.eia.gov/dnav/ng/hist/n3050ca3m.htm 
# consumption from https://www.eia.gov/dnav/ng/ng_cons_sum_dcu_SCA_m.htm 
# also use US NonFarm Payroll for income

calgasdata <- read.csv('C:/Users/gilbe/Dropbox/Econometrics/gasdemand.csv',sep=",")
calgas <- ts(calgasdata, freq=12, start = 1989)
calgas <- window(calgas,c(2001,2),c(2016,12))

# lntot = log of total natural gas delivered in CA. This is our quantity q. 
# lntotlag = one-month lag of quantity.
# lnprice =log of the CA citygate price
# pacCDDm and pacHDDm are the monthly CDD and HDD in the pacific region (CA)
# lpacCDDm and lpacHDDm are the one-month lags
# mtnCDDm and mtnHDDm are for the Rocky Mountain West (CA neighbors)
# l2smtnCDD is the 12 month cumulative sum, so one year of cooling demand
#     similar for l2s for HDD and for pacific region. Inventory instrument. 

# OLS results for demand
gasdemOLS <- lm(lntot ~ lnprice + lntotlag + 
                  pacCDDm + pacHDDm + lpacCDDm + lpacHDDm + TIME + 
                  mo_1 + mo_2 + mo_3 + mo_4 + mo_5 + mo_6 + 
                  mo_7 + mo_8 + mo_9 + mo_10 + mo_11, data=calgas)
summary(gasdemOLS)
demct <- coeftest(gasdemOLS,vcov = vcovHC(gasdemOLS))

# OLS results for supply
gassupOLS <- lm(lntot ~ lnprice + lntotlag + 
                  mtnCDDm + mtnHDDm + lmtnCDDm + lmtnHDDm + 
                  l2smtnCDD + l2smtnHDD + TIME + 
                  mo_1 + mo_2 + mo_3 + mo_4 + mo_5 + mo_6 + 
                  mo_7 + mo_8 + mo_9 + mo_10 + mo_11, data=calgas)
summary(gassupOLS)
supct <- coeftest(gassupOLS,vcov = vcovHC(gassupOLS))


## IV results:

# Using felm() function from lfe package
# syntax: 
# felm(regression | fixed effects | (IV first stage) | clustering variables)
# we will focus on regression and IV first stage

# demand:
gasdemIV <- felm(lntot ~ pacCDDm + pacHDDm + lpacCDDm + lpacHDDm + TIME + 
                     mo_1 + mo_2 + mo_3 + mo_4 + mo_5 + mo_6 + 
                     mo_7 + mo_8 + mo_9 + mo_10 + mo_11 | 0 |
                     (lnprice | lntotlag ~ mtnCDDm + mtnHDDm + lmtnCDDm + 
                      lmtnHDDm + l2smtnCDD + l2smtnHDD) | 0 , 
                   data = calgas)
summary(gasdemIV,robust=TRUE)
# Joint F-test for instruments for both endogneous variables is too low:
condfstat(gasdemIV,type="robust", quantiles = c(0.05,0.95))

# supply:
gassupIV <- felm(lntot ~  + TIME + mtnCDDm + mtnHDDm + lmtnCDDm + 
                   lmtnHDDm + l2smtnCDD + l2smtnHDD +
                   mo_1 + mo_2 + mo_3 + mo_4 + mo_5 + mo_6 + 
                   mo_7 + mo_8 + mo_9 + mo_10 + mo_11 | 0 |
                (lnprice | lntotlag ~ pacCDDm + pacHDDm + lpacCDDm + lpacHDDm) |
                   0 , data = calgas)
summary(gassupIV,robust=TRUE)
# Joint F-test for instruments for both endogneous variables is too low:
condfstat(gassupIV,type="robust", quantiles = c(0.05,0.95))


# Compare models in a nicer table

# may need to substitute HC robust standard errors
demOLSse <- demct[,2]
demOLSp <- demct[,4]
supOLSse <- supct[,2]
supOLSp <- supct[,4]
# get first-stage F stats
demF = condfstat(gasdemIV)
supF = condfstat(gassupIV)

screenreg(list(gasdemOLS,gasdemIV),
          override.se = list(demOLSse, gasdemIV$rse),
          override.pvalues = list(demOLSp,gasdemIV$rpval),
          digits = 4,
          stars = c(0.1,0.05,0.01),
          include.adjrs = FALSE,
          include.rs = FALSE,
          custom.gof.rows = 
            list("Log Price F-stat"=c(NA,demF[,1]),
                 "Log Q(t-1) F-stat" = c(NA,demF[,2])),
          omit.coef = "mo_",
          custom.model.names = c("OLS Demand","2SLS Demand"),
          custom.coef.names = c("Intercept","Log Price","Log Q(t-1)",
                                "Pacific CDD","Pacific HDD",
                                "Pacific CDD(t-1)","Pacific HDD(t-1)","Trend",
                                "Log Price","Log Q(t-1)"),
          custom.note = "Monthly dummies are included",
          caption.above = "California Natural Gas Demand Estimation")

screenreg(list(gassupOLS,gassupIV),
          override.se = list(supOLSse, gassupIV$rse),
          override.pvalues = list(supOLSp,gassupIV$rpval),
          digits = 4,
          stars = c(0.1,0.05,0.01),
          include.adjrs = FALSE,
          include.rs = FALSE,
          custom.gof.rows = 
              list("Log Price F-stat"=c(NA,supF[,1]),
                   "Log Q(t-1) F-stat" = c(NA,supF[,2])),
          omit.coef = "mo_",
          custom.model.names = c("OLS Supply","2SLS Supply"),
          custom.coef.names = c("Intercept","Log Price","Log Q(t-1)",
                                "Mtn CDD","Mtn HDD",
                                "Mtn CDD(t-1)","Mtn HDD(t-1)",
                                "Cumulative Mtn CDD","Cumulative Mtn HDD",
                                "Trend","Log Price","Log Q(t-1)"),
          custom.note = "Monthly dummies are included",
          caption.above = "California Natural Gas Supply Estimation")



# Using single equation ivreg() from the AER package
gasdemIV2 <- ivreg(lntot ~ lntotlag + lnprice + 
                          pacCDDm + pacHDDm + lpacCDDm + lpacHDDm + TIME + 
                          mo_1 + mo_2 + mo_3 + mo_4 + mo_5 + mo_6 + 
                          mo_7 + mo_8 + mo_9 + mo_10 + mo_11
                         | mtnCDDm + mtnHDDm + lmtnCDDm + lmtnHDDm + l2smtnCDD + 
                          l2smtnHDD + 
                          pacCDDm + pacHDDm + lpacCDDm + lpacHDDm + TIME + 
                          mo_1 + mo_2 + mo_3 + mo_4 + mo_5 + mo_6 + 
                          mo_7 + mo_8 + mo_9 + mo_10 + mo_11, data = calgas)
summary(gasdemIV2,diagnostics=TRUE,vcov = vcovHC)
coeftest(gasdemIV2, vcov = vcovHC, type = "HC1")

# Using tsls() from sem package
gasdemIV3 <- tsls(lntot ~ lntotlag + lnprice + pacCDDm + pacHDDm + lpacCDDm + 
                          lpacHDDm + TIME + 
                          mo_1 + mo_2 + mo_3 + mo_4 + mo_5 + mo_6 + 
                          mo_7 + mo_8 + mo_9 + mo_10 + mo_11,
                         ~ mtnCDDm + mtnHDDm + lmtnCDDm + lmtnHDDm + l2smtnCDD + 
                          l2smtnHDD + pacCDDm + pacHDDm + lpacCDDm + lpacHDDm + 
                          TIME + mo_1 + mo_2 + mo_3 + mo_4 + mo_5 + mo_6 + 
                          mo_7 + mo_8 + mo_9 + mo_10 + mo_11, data = calgas)
summary(gasdemIV3)
coeftest(gasdemIV3, vcov = vcovHC, type = "HC1")


# Estimate supply and demand simultaneously using systemfit package

# Define equation objects 
eqdem <- lntot ~ lntotlag + lnprice + 
                  pacCDDm + pacHDDm + lpacCDDm + lpacHDDm + TIME + 
                  mo_1 + mo_2 + mo_3 + mo_4 + mo_5 + mo_6 + 
                  mo_7 + mo_8 + mo_9 + mo_10 + mo_11

eqsup <- lntot ~ lntotlag + lnprice + 
                  mtnCDDm + mtnHDDm + lmtnCDDm + lmtnHDDm + 
                  l2smtnCDD + l2smtnHDD + TIME + 
                  mo_1 + mo_2 + mo_3 + mo_4 + mo_5 + mo_6 + 
                  mo_7 + mo_8 + mo_9 + mo_10 + mo_11

eqsys <- list(dem=eqdem, supply=eqsup)

fit2slsELAST <- systemfit(eqsys,method="2SLS",
                          inst =~ pacCDDm + pacHDDm + lpacCDDm + lpacHDDm + 
                            lmtnCDDm + lmtnHDDm + 
                            l2smtnCDD + l2smtnHDD + mtnCDDm + mtnHDDm + TIME + 
                            mo_1 + mo_2 + mo_3 + mo_4 + mo_5 + mo_6 + 
                            mo_7 + mo_8 + mo_9 + mo_10 + mo_11,data=calgas)
summary(fit2slsELAST)

fit3slsELAST <- systemfit(eqsys,method="3SLS",
                          inst =~ pacCDDm + pacHDDm + lpacCDDm + lpacHDDm + 
                            lmtnCDDm + lmtnHDDm + 
                            l2smtnCDD + l2smtnHDD + mtnCDDm + mtnHDDm + TIME + 
                            mo_1 + mo_2 + mo_3 + mo_4 + mo_5 + mo_6 + 
                            mo_7 + mo_8 + mo_9 + mo_10 + mo_11,data=calgas)
summary(fit3slsELAST)

# 2SLS should be correct but 3SLS is more efficient (lower standard errors)
# we can test if coefficients are close enough, use the 3SLS ones:
hausman.systemfit(fit2slsELAST,fit3slsELAST)


# We could do this in levels instead of natural logs, and then these are 
# interpreted as slopes instead of elasticities:

# define equation objects
eqdem <- totalgas ~ ltotalgas + natgasprice + pacCDDm + pacHDDm +
                lpacCDDm + lpacHDDm + TIME + 
                  mo_1 + mo_2 + mo_3 + mo_4 + mo_5 + mo_6 +
                mo_7 + mo_8 + mo_9 + mo_10 + mo_11

eqsup <- totalgas ~ ltotalgas + natgasprice + mtnCDDm + mtnHDDm +
                lmtnCDDm + lmtnHDDm + l2smtnCDD + l2smtnHDD + TIME + 
                  mo_1 + mo_2 + mo_3 + mo_4 + mo_5 + mo_6 +
                mo_7 + mo_8 + mo_9 + mo_10 + mo_11

eqsys <- list(dem=eqdem, supply=eqsup)

# estimate supply and demand separately within one command
fit2slsSLOPE <- systemfit(eqsys,method="2SLS",
                          inst = ~pacCDDm + pacHDDm + lpacCDDm + lpacHDDm + 
                            lmtnCDDm + lmtnHDDm + l2smtnCDD + l2smtnHDD + 
                            mtnCDDm + mtnHDDm + TIME + 
                            mo_1 + mo_2 + mo_3 + mo_4 + mo_5 + mo_6 + 
                            mo_7 + mo_8 + mo_9 + mo_10 + mo_11,
                          data=calgas)

# estimate supply and demand simultaneously as a system of equations
fit3slsSLOPE <- systemfit(eqsys,method="3SLS",
                          inst=~pacCDDm + pacHDDm + lpacCDDm + lpacHDDm + 
                            lmtnCDDm + lmtnHDDm + 
                            l2smtnCDD + l2smtnHDD + mtnCDDm + mtnHDDm + TIME + 
                            mo_1 + mo_2 + mo_3 + mo_4 + mo_5 + mo_6 + 
                            mo_7 + mo_8 + mo_9 + mo_10 + mo_11,
                          data=calgas)
summary(fit3slsSLOPE)
hausman.systemfit(fit2slsSLOPE,fit3slsSLOPE)
