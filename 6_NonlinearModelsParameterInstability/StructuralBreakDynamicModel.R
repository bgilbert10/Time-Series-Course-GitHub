require(forecast)
require(quantmod)
require(strucchange)
require(dynlm)

getSymbols("GC=F")   #gold futures price
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


static.model = lm(dlgld~(breakfactor(bp.gold,breaks=2)/(dtip + dbei+ dtwd+dsp5)))
summary(static.model)
dynamic.model = dynlm(dlgld~(breakfactor(bp.gold,breaks=2)/(L(dlgld,1) + L(dtip,1) + L(dbei,1)+ L(dtwd,1)+L(dsp5,1))))
summary(dynamic.model)

# Rolling regressions
library(dplyr); library(tidyr); library(purrr) # Data wrangling
library(ggplot2); library(stringr) # Plotting
library(lubridate)   # Date calculations
library(tidyfit)     # Model fitting

data <- as.data.frame(cbind(dlgld, dtip, dbei, dtwd, dsp5, index(dlgld)))
colnames(data) <- c("dlgld", "dtip", "dbei", "dtwd", "dsp5", "date")

mod_frame <- data %>% 
  regress(dlgld ~ dtip + dbei + dtwd + dsp5, 
          m("lm", vcov. = "HAC"))

# Using a sliding window - agnostic about time index, just over last 200 rows. 
mod_frame_rolling <- data %>% 
  regress(dlgld ~ dtip + dbei + dtwd + dsp5, 
          m("lm", vcov. = "HAC"),
          .cv = "sliding_window", .cv_args = list(lookback = 200, step = 10),
          .force_cv = TRUE, .return_slices = TRUE)

df_beta <- coef(mod_frame_rolling)

df_beta <- df_beta %>% 
  unnest(model_info) %>% 
  mutate(upper = estimate + 2 * std.error, lower = estimate - 2 * std.error)

df_beta %>% 
  mutate(slice_id = as.numeric(substr(slice_id,6,nchar(slice_id)))) %>% 
  filter(term == "dtip") %>% 
  ggplot(aes(slice_id)) +
  geom_hline(yintercept = filter(coef(mod_frame),term=="dtip")$estimate) +
  geom_ribbon(aes(ymax = upper, ymin = lower), alpha = 0.25) +
  geom_line(aes(y = estimate)) +
  theme_bw(8)

df_beta %>% 
  mutate(slice_id = as.numeric(substr(slice_id,6,nchar(slice_id)))) %>% 
  filter(term == "dbei") %>% 
  ggplot(aes(slice_id)) +
  geom_hline(yintercept = filter(coef(mod_frame),term=="dbei")$estimate) +
  geom_ribbon(aes(ymax = upper, ymin = lower), alpha = 0.25) +
  geom_line(aes(y = estimate)) +
  theme_bw(8)

df_beta %>% 
  mutate(slice_id = as.numeric(substr(slice_id,6,nchar(slice_id)))) %>% 
  filter(term == "dtwd") %>% 
  ggplot(aes(slice_id)) +
  geom_hline(yintercept = filter(coef(mod_frame),term=="dtwd")$estimate) +
  geom_ribbon(aes(ymax = upper, ymin = lower), alpha = 0.25) +
  geom_line(aes(y = estimate)) +
  theme_bw(8)

df_beta %>% 
  mutate(slice_id = as.numeric(substr(slice_id,6,nchar(slice_id)))) %>% 
  filter(term == "dsp5") %>% 
  ggplot(aes(slice_id)) +
  geom_hline(yintercept = filter(coef(mod_frame),term=="dsp5")$estimate) +
  geom_ribbon(aes(ymax = upper, ymin = lower), alpha = 0.25) +
  geom_line(aes(y = estimate)) +
  theme_bw(8)

# Using a sliding index - need to get the indexing right!
mod_frame_rolling <- data %>% 
  regress(dlgld ~ dtip + dbei + dtwd + dsp5, 
          m("lm", vcov. = "HAC"),
          .cv = "sliding_index", .cv_args = list(lookback = 1, step = 10, index = "date"),
          .force_cv = TRUE, .return_slices = TRUE)

df_beta <- coef(mod_frame_rolling)

df_beta <- df_beta %>% 
  unnest(model_info) %>% 
  mutate(upper = estimate + 2 * std.error, lower = estimate - 2 * std.error)

df_beta %>% 
  mutate(slice_id = as.numeric(slice_id)) %>% 
  filter(term == "dtip") %>% 
  ggplot(aes(slice_id)) +
  geom_hline(yintercept = filter(coef(mod_frame),term=="dtip")$estimate) +
  geom_ribbon(aes(ymax = upper, ymin = lower), alpha = 0.25) +
  geom_line(aes(y = estimate)) +
  theme_bw(8)
