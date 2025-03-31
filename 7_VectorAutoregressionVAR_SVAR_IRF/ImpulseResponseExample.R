# Delving into Impulse Response Functions

# Some packages we might use:
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
# install.packages("MTS")
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

# merge oil, gas, and drilling and calculate returns/changes
oilgas= merge.xts(MHHNGSP,MCOILWTICO,IPN213111N,join="inner")
plot(oilgas)
# calculate log differences as ts() objects, notice start dates
doil = ts(na.omit(diff(log(oilgas$MCOILWTICO))),freq=12,start=1997+1/12)
dgas = ts(na.omit(diff(log(oilgas$MHHNGSP))),freq=12,start=1997+1/12)
dwell = ts(na.omit(diff(oilgas$IPN213111N)),freq=12,start=1997+1/12)
ogw = cbind(doil,dgas,dwell)

# To look at a 2-equation oil & gas price return VAR
# create the vector of returns - remember order matters!
og <- cbind(doil,dgas)
MTSplot(og)

# Estimate the VAR(p)
varog = VAR(og,p=2)
summary(varog)

# Estimate and visualize the Impulse Response Function for comparison 
irfog = irf(varog,n.ahead=6,ortho=FALSE)
plot(irfog)

# Use the VAR coefficients to calculate "F", the companion form matrix
F = t(matrix(c(varog$varresult$doil$coefficients[c(1:4)],varog$varresult$dgas$coefficients[c(1:4)],1,0,0,0,0,1,0,0),nrow=4,ncol=4))
F

# 2-period-ahead IRF (non-orthogonalized) is the upper quadrant of F*F
F2 = F%*%F          
irfF = F2[-c(3:4),-c(3:4)]

# compare my direct calculations (irfF matrix) to R's automated irf output - "irfog"
irfF
irfog$irf
# 3 period ahead, etc. 
F3 = F%*%F2
irfF3 = F3[-c(3:4),-c(3:4)]
irfF3
irfog$irf



# orthogonolized IRF from R
irfog.orth = irf(varog,n.ahead=6,ortho=T)
plot(irfog.orth)

# Show how to calculate the Orthogonalized IRF step by step
# Calculate the covariance matrix of VAR(p) residuals:
Omega = summary(varog)$covres
# Factor it into upper/lower triangular. Recall R gives the upper triangular, when we want its transpose:
P = chol(Omega)
# check that products of lower triangular P and with its transpose return Omega:
t(P)%*%P

# could report in ADA' form
Dhalf = diag(P)
A = t(P/Dhalf)
# check that we get Omega back if we factored correctly:
A%*%diag(Dhalf)%*%diag(Dhalf)%*%t(A)

# Effect of one standard deviation increase in y, adjust irf by lower triangular P:
irfForth = irfF%*%t(P)
# compare our calculations to R's output
irfForth
irfog.orth

# Effect of one unit increase in y, adjust by A instead:
irfForthA = irfF%*%A
# notice that this is different than R's automated irf:
# R computes effect of one standard devation change for orthogonalized IRFs.
irfForthA

# Compare to structural VAR
# transform reduced form into structural
Avar = matrix(c(1,NA,0,1),nrow=2,ncol=2)
svarog = SVAR(varog,estmethod="direct",Amat=Avar,hessian=T)
# estimation method direct is MLE. hessian=T required for standard errors.
# coefficients of the structural matrix (contemporaneous coefficients)
svarog$A
# transform to the "A" matrix from the orthogonalized IRFs above and compare: 
A.struct = solve(svarog$A)
A.struct
# standard errors of the structural coefficients matrix
svarog$Ase
summary(svarog)
plot(irf(svarog,n.ahead=6,ortho=T))

# notice that the simple regression of gas returns on oil returns gives
# similar coefficient to the adjustment made in the A matrix;
# A matrix adjusts for their contemporaneous linear relationship
Areg = lm(dgas ~ doil)
summary(Areg)


# Repeat this for a VAR(2) in oil returns and changes in the drilling index
ow <- cbind(doil,dwell)
MTSplot(ow)
varow = VAR(ow,p=2)
summary(varow)

irfow = irf(varow,n.ahead=12,ortho=FALSE)
plot(irfow)

F = t(matrix(c(varow$varresult$doil$coefficients[c(1:4)],varow$varresult$dwell$coefficients[c(1:4)],1,0,0,0,0,1,0,0),nrow=4,ncol=4))
F2 = F%*%F          
irfF = F2[-c(3:4),-c(3:4)]
irfF
irfow

# orthogonolized
irfow.orth = irf(varow,n.ahead=2,ortho=TRUE)
plot(irfow.orth)
Omega = summary(varow)$covres
P = chol(Omega)
# check
t(P)%*%P

# could report in ADA' form
Dhalf = diag(P)
A = t(P/Dhalf)
# check
A%*%diag(Dhalf)%*%diag(Dhalf)%*%t(A)

# Effect of one standard deviation increase in y
irfForth = irfF%*%t(P)
irfForth
irfow.orth

# Effect of one unit increase in y
irfForthA = irfF%*%A
irfForthA

# Compare to structural VAR
# transform reduced form into structural
Avar = matrix(c(1,NA,0,1),nrow=2,ncol=2)
svarow = SVAR(varow,estmethod="direct",Amat=Avar,hessian=T)
# estimation method direct is MLE. hessian=T required for standard errors.
# coefficients of A matrix
svarow$A
A = solve(svarow$A)
# standard errors of A matrix
svarow$Ase
summary(svarow)
plot(irf(svarow,n.ahead=6,ortho=T))

Ax = lm(dwell ~ doil)
summary(Ax)
