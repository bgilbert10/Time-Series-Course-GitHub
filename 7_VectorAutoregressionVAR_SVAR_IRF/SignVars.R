install.packages("minqa")
install.packages("C:/Users/gilbe/Downloads/HI_0.5.tar.gz", repos = NULL)
install.packages("mvnfast")
install.packages("C:/Users/gilbe/Downloads/VARsignR_0.1.2.tar.gz", repos = NULL)
install.packages("bsvarSIGNs")

# http://cran.nexr.com/web/packages/VARsignR/vignettes/VARsignR-vignette.html
set.seed(12345)
library(VARsignR)
data(uhligdata)
# Uhlig: what is the effect of unanticipated shock to "i" on "y"?
# what is the effect of "i" on "p"?
# y = real gdp, yd = gdp deflator, p = commodity price index, i = fed funds rate
# rnb = nonborrowed reserves, rt = total reserves. 
# Uhlig assumptions: unanticipated shock to "i":
# 1. nonnegative impact on subsequent "i" for x periods: >= 0.
# 2. non positive impact on "p" and "yd" for x periods: =< 0.
# 3. non positive impact on "rnb" for x periods: =< 0.

# order of variables: y, yd, p, i, rnb, rt
# sign restrictions on 2nd (-), 3rd (-), 4th (+), and 5th (-) variables.
constr = c(+4,-3,-2,-5)
# first restriction MUST be on the impulse variable, "i", the 4th. 
# others are optional, can be in any order (e.g., 3, 2, 5, or 2, 3, 5 or whatever)

model1 <- uhlig.reject(Y=uhligdata, nlags=12, draws=200, subdraws=200, nkeep=1000,
                       KMIN=1, KMAX=6, constrained=constr, constant=FALSE, steps=60)
birfs <- model1$IRFS
v1 = c("GDP","GDP Deflator","Comm.Pr.Index","Fed Funds Rate","NB Reserves","Total Reserves")
irfplot(irfdraws = birfs, type="median", labels=v1, save=FALSE, bands=c(0.16, 0.84), grid=TRUE, bw=FALSE)



# order of variables: weath, diff, flr
# sign restrictions on 2nd (+) and 3rd (+) variables (weather increases diff and flaring).
constr = c(+1,+2,+3)
# sign restrictions on 2nd (+) - weather increases diff, but anything can happen to flaring.
constr = c(+1,+2)
# first restriction MUST be on the impulse variable, "weath", the 1st. 
# others are optional, can be in any order 

model1 <- uhlig.reject(Y=kil, nlags=2, draws=200, subdraws=200, nkeep=1000,
                       KMIN=1, KMAX=2, constrained=constr, constant=TRUE, steps=25)
model3 <- uhlig.penalty(Y=kil, nlags=2, draws=2000, subdraws=1000,
                        nkeep=1000, KMIN=1, KMAX=2, constrained=constr,
                        constant=TRUE, steps=25, penalty=100, crit=0.001)
birfs <- model1$IRFS
v1 = c("Weather Shock","Differential","Log Flaring")
irfplot(irfdraws = birfs, type="median", labels=v1, save=FALSE, bands=c(0.16, 0.84), grid=TRUE, bw=FALSE)
irfs3 <- model3$IRFS
irfplot(irfdraws=irfs3, type="median", labels=v1, save=FALSE, bands=c(0.16, 0.84),
        grid=TRUE, bw=FALSE)

fp.target(Y=kil, irfdraws=birfs,  nlags=2,  constant=TRUE, labels=v1, target=TRUE,
          type="median", bands=c(0.16, 0.84), save=FALSE,  grid=TRUE, bw=FALSE, 
          legend=TRUE, maxit=1000)
fp.target(Y=kil, irfdraws=irfs3,  nlags=2,  constant=TRUE, labels=v1, target=TRUE,
          type="median", bands=c(0.16, 0.84), save=FALSE,  grid=TRUE, bw=FALSE, 
          legend=TRUE, maxit=1000)

model0 <- rfbvar(Y=kil, nlags=2, draws=1000, constant=TRUE,
                 steps=25, shock=1)

irfs0 <- model0$IRFS

irfplot(irfdraws=irfs0, type="median", labels=v1, save=FALSE, bands=c(0.16, 0.84),
        grid=TRUE, bw=FALSE)


plot(irf(varkil,n.ahead=24,ortho=T,impulse = "weath",response="weath"))
plot(irf(varkil,n.ahead=24,ortho=T,impulse = "weath",response="diff"))
plot(irf(varkil,n.ahead=24,ortho=T,impulse = "weath",response="flare"))

plot(irf(varkil,n.ahead=24,ortho=T,impulse = "weath"))
plot(irf(varkil,n.ahead=24,ortho=T,impulse = "diff",response="weath"))
plot(irf(varkil,n.ahead=24,ortho=T,impulse = "diff",response="diff"))
plot(irf(varkil,n.ahead=24,ortho=T,impulse = "diff",response="flare"))
plot(irf(varkil,n.ahead=24,ortho=T,impulse = "flare",response="flare"))

# now investigate price differential shock

# order of variables: weath, diff, flr
# sign restrictions on 2nd (+) and 3rd (+) variables (diff increases diff and flaring).
constr = c(+2,+3)
# sign restrictions on 2nd (+) - diff increases diff, but anything can happen to flaring.
constr = c(+2)
# first restriction MUST be on the impulse variable, "diff", the 2nd. 
# others are optional, can be in any order 

model1 <- uhlig.reject(Y=kil, nlags=2, draws=200, subdraws=200, nkeep=1000,
                       KMIN=1, KMAX=2, constrained=constr, constant=TRUE, steps=25)
model3 <- uhlig.penalty(Y=kil, nlags=2, draws=2000, subdraws=1000,
                        nkeep=1000, KMIN=1, KMAX=2, constrained=constr,
                        constant=TRUE, steps=25, penalty=100, crit=0.001)
birfs <- model1$IRFS
v1 = c("Weather Shock","Differential","Log Flaring")
irfplot(irfdraws = birfs, type="median", labels=v1, save=FALSE, bands=c(0.16, 0.84), grid=TRUE, bw=FALSE)
irfs3 <- model3$IRFS
irfplot(irfdraws=irfs3, type="median", labels=v1, save=FALSE, bands=c(0.16, 0.84),
        grid=TRUE, bw=FALSE)

fp.target(Y=kil, irfdraws=birfs,  nlags=2,  constant=TRUE, labels=v1, target=TRUE,
          type="median", bands=c(0.16, 0.84), save=FALSE,  grid=TRUE, bw=FALSE, 
          legend=TRUE, maxit=1000)
fp.target(Y=kil, irfdraws=irfs3,  nlags=2,  constant=TRUE, labels=v1, target=TRUE,
          type="median", bands=c(0.16, 0.84), save=FALSE,  grid=TRUE, bw=FALSE, 
          legend=TRUE, maxit=1000)

model0 <- rfbvar(Y=kil, nlags=2, draws=1000, constant=TRUE,
                 steps=25, shock=2)

irfs0 <- model0$IRFS

irfplot(irfdraws=irfs0, type="median", labels=v1, save=FALSE, bands=c(0.16, 0.84),
        grid=TRUE, bw=FALSE)


# https://bsvars.org/bsvarSIGNs/
install.packages("bsvarSIGNs")
library(bsvarSIGNs)

# specify identifying restrictions:
# + positive effect on differential (positive sign restriction)
sign_irf       = matrix(c(1, 1, NA, NA, 1, 1, NA, NA, 1), 3, 3)
sign_irf       = matrix(c(1, 1, NA, NA, 1, NA, NA, NA, 1), 3, 3)
sign_irf       = matrix(c(1, 1, NA, NA, 1, 1, NA, -1, 1), 3, 3)

# specify the model
specification  = specify_bsvarSIGN$new(kil,
                                       p        = 2,
                                       sign_irf = sign_irf)

# estimate the model
posterior      = estimate(specification, S = 100)

# compute and plot impulse responses
irf            = compute_impulse_responses(posterior, horizon = 25)
plot(irf, probability = 0.68)


# investigate the effects of the optimism shock
data(optimism)

# specify identifying restrictions:
# + no effect on productivity (zero restriction)
# + positive effect on stock prices (positive sign restriction)
sign_irf       = matrix(c(0, 1, rep(NA, 23)), 5, 5)

# specify the model
specification  = specify_bsvarSIGN$new(optimism * 100,
                                       p        = 4,
                                       sign_irf = sign_irf)

# estimate the model
posterior      = estimate(specification, S = 100)

# compute and plot impulse responses
irf            = compute_impulse_responses(posterior, horizon = 40)
plot(irf, probability = 0.68)


# investigate the effects of the contractionary monetary policy shock
data(monetary)

# specify identifying restrictions:
# + sign restrictions on the impulse responses at horizons from 0 to 5
sign_irf       = matrix(NA, 6, 6)
sign_irf[, 1]  = c(NA, -1, -1, NA, -1, 1)
sign_irf       = array(sign_irf, dim = c(6, 6, 6))

# + narrative sign restriction: the shock is positive in October 1979
sign_narrative = list(
  specify_narrative(start = 166, periods = 1, type = "S", sign = 1, shock = 1),
  specify_narrative(start = 166, periods = 1, type = "B", sign = 1, shock = 1, var = 6)
)

# specify the model
specification  = specify_bsvarSIGN$new(monetary       * 100,
                                       p              = 12,
                                       sign_irf       = sign_irf,
                                       sign_narrative = sign_narrative)

# estimate the model
posterior      = estimate(specification, S = 100)

# compute and plot impulse responses
irf            = compute_impulse_responses(posterior, horizon = 60)
plot(irf, probability = 0.68)
