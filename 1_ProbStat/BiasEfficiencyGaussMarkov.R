# Program to illustrate bias and efficiency when Gauss Markov assumptions fail

# Bias: deviation of expected value of sample parameter estimate from "true" 
#       population parameter

# Efficiency: the variance of the sample parameter estimate should be as small 
#       as possible

# Consistency: distribution of sample parameter estimate should converge to 
#       population value as sample size grows

# Gauss-Markov: OLS (Ordinary Least Squares) is Best (lowest variance) Linear 
#       Unbiased Estimator (BLUE) IF:
#     1. true model linear in parameters and residuals: 
#             y = b0 + b1*x1 + b2*x2 + e
#     2. X variables (right hand side) are not constants or perfectly correlated 
#             with each other
#     3. Residuals "e" have constant variance 
#           - (not more noisy for some X's than others, homoskedasticity vs.
#             heteroskedasticy)
#     4. Residuals "e" are uncorrelated with each other 
#           - (no peer effects, no serial correlation)
#     5. All X variables are uncorrelated with the residual e 
#           - (observed X is not picking up some unobserved or uncontrolled 
#             factor)
#
# In each case we will run the linear regression y = b0 + b1*x1 + b2*x2 + e on 
#     the data, but the "true" model or "data generating process" is different

# packages
# install.packages("MASS")
require(MASS)
# install.packages("car")
require(car)

# 1. True model not linear in parameters --------------
# Set the seed and generate the true model
set.seed(826)
# Covariance matrix of x1, x2, and e 
Sig <- matrix(c(4,1,0,1,2,0,0,0,1),3,3)
Sig       # Notice e does not covary with x1 or x2 (assumption 5) 
          # also x1 and x2 can covary, but not perfectly (assumption 2)
# Mean of x1, x2, and e
moo <- c(10,3,0)
# generate data
Xe <- mvrnorm(n=1000,mu=moo,Sigma=Sig)
# give the variables names
colnames(Xe)<-c("x1","x2","e")
# store as a data frame
Xe <- as.data.frame(Xe)
head(Xe)

# plot empirical distribution of each: 
hist(Xe$x1, breaks = 100, cex.main = 0.9)
hist(Xe$x2, breaks = 100, cex.main = 0.9)
hist(Xe$e, breaks = 100, cex.main = 0.9)
# Sample correlations and covariances (notice difference from "truth")
cov(Xe)
cor(Xe)
scatterplotMatrix(Xe)

# generate the true y (outcome)
y <- exp(10 - 6*Xe$x1 + 5*Xe$x2 + Xe$e) # log(y) is linear in parameters and 
                                        # residual, but y is not.
plot(Xe$x1,y)
plot(Xe$x1,log(y))
plot(Xe$x2,y)
plot(Xe$x2,log(y))

# run the linear regression
summary(lm(y~x1+x2,data=Xe))
summary(lm(log(y)~x1+x2,data=Xe))


# 2. Perfect colinearity -------------------
# Right hand side variables are constants or are perfectly correlated
# NOTE: The intercept is a constant, but no other variables can be constant 
#     - (otherwise perfectly correlated with the intercept)

set.seed(826)
# Covariance matrix of x1 and e 
Sig <- matrix(c(4,0,0,1),2,2)
Sig       # Notice e does not covary with x1 (assumption 5) 
# We will make x2 an exact multiple of x1 (violate assumption 2)
# Mean of x1 and e
moo <- c(10,0)
# generate data
Xe <- mvrnorm(n=1000,mu=moo,Sigma=Sig)
# generate x2 as arbitrary multiple of x1
x2 <- 7*Xe[,1]
Xe <- as.data.frame(cbind(Xe[,1],x2,Xe[,2]))
# give the variables names
colnames(Xe)<-c("x1","x2","e")
head(Xe)

# Sample correlations and covariances (notice difference from "truth")
cov(Xe)
cor(Xe)
scatterplotMatrix(Xe)

# generate the true y (outcome)
y <- 10 - 6*Xe$x1 + 5*Xe$x2 + Xe$e # y is linear in parameters (assumption 1)
plot(Xe$x1,y)
plot(Xe$x2,y) # Should be a tight fit!

# run the linear regression
summary(lm(y~x1+x2,data=Xe)) # Why an intercept of 10 and coefficient of 29? 
                             # -6 + 5*7 = 29
                             # some stats packages will not produce output

# NOTE: regressing on an intercept ONLY estimates the mean of y - intercept can 
# be constant. 
mean(y)
summary(lm(y~1))

# But what if another regressor is also constant?
set.seed(826)
# Covariance matrix of x1, x2, and e 
Sig <- matrix(c(4,0,0,0,0,0,0,0,1),3,3)
Sig       # Notice e does not covary with x1 or x2 (assumption 5) 
          # But x2 also has no variance (violate assumption 2)
# Mean of x1, x2, and e
moo <- c(10,3,0)
# generate data
Xe <- mvrnorm(n=1000,mu=moo,Sigma=Sig)
# give the variables names
colnames(Xe)<-c("x1","x2","e")
Xe <- as.data.frame(Xe)
head(Xe)

# Sample correlations and covariances (notice difference from "truth")
cov(Xe)
cor(Xe)
scatterplotMatrix(Xe)

# generate the true y (outcome)
y <- 10 - 6*Xe$x1 + 5*Xe$x2 + Xe$e # y is linear in parameters (assumption 1)
plot(Xe$x1,y)
plot(Xe$x2,y) # Should be a tight fit!

# run the linear regression
summary(lm(y~x1+x2,data=Xe)) # Why an intercept of 25 and coefficient of -6? 
                             # 10 + 5*3
                             # some stats packages will not even produce output


# 3. Heteroskedastic residuals -----------------
# Residuals "e" do not have constant variance (are "heteroskedastic")
# Set the seed and generate the true model
set.seed(826)
# Covariance matrix of x1, x2
Sig <- matrix(c(4,1,1,2),2,2)
Sig       
# Mean of x1, x2
moo <- c(10,3)
# generate data
Xe <- mvrnorm(n=1000,mu=moo,Sigma=Sig)
# generate homoskedastic residuals
eps = rnorm(n=1000,mean=0,sd=sqrt(1))
# generate heteroskedastic residuals
sigma2 = (eps^2)*(Xe[,1]^2+Xe[,2]^2)
eps2 = rnorm(n=1000,mean=0,sd=sqrt(sigma2))

Xe1 = as.data.frame(cbind(Xe,eps))
colnames(Xe1)<-c("x1","x2","e1")
Xe2 = as.data.frame(cbind(Xe,eps2))
colnames(Xe2)<-c("x1","x2","e2")

# Sample correlations and covariances (notice difference from "truth")
scatterplotMatrix(Xe1)
scatterplotMatrix(Xe2)

# generate the true y (outcome)
y1 <- 10 - 6*Xe1$x1 + 5*Xe1$x2 + Xe1$e1
y2 <- 10 - 6*Xe2$x1 + 5*Xe2$x2 + Xe2$e2

# not necessarily obvious differences in the plot
plot(Xe1$x1,y1)
plot(Xe2$x1,y2) 

# run the linear regression - notice the difference in residual standard error, 
# coefficient std. error, Rsquared. 
summary(lm(y1~x1+x2,data=Xe1)) 
summary(lm(y2~x1+x2,data=Xe2))



# 4. Serially correlated residuals ---------------
# Residuals "e" are serially correlated (autocorrelated)
# Set the seed and generate the true model
set.seed(826)
# Covariance matrix of x1, x2
Sig <- matrix(c(4,1,1,2),2,2)
Sig       
# Mean of x1, x2
moo <- c(10,3)
# generate data
Xe <- mvrnorm(n=1000,mu=moo,Sigma=Sig)
# generate independent residuals
eps = rnorm(n=1000,mean=0,sd=sqrt(1))
# generate serially correlated residuals
eps2 <- arima.sim(model=list(ar=c(0.8)),n=1000,sd=1)

Xe1 = as.data.frame(cbind(Xe,eps))
colnames(Xe1)<-c("x1","x2","e1")
Xe2 = as.data.frame(cbind(Xe,eps2))
colnames(Xe2)<-c("x1","x2","e2")

# Sample correlations and covariances (notice difference from "truth")
scatterplotMatrix(Xe1)
scatterplotMatrix(Xe2)

# generate the true y (outcome)
y1 <- 10 - 6*Xe1$x1 + 5*Xe1$x2 + Xe1$e1
y2 <- 10 - 6*Xe2$x1 + 5*Xe2$x2 + Xe2$e2

# not necessarily obvious differences in the plot
plot(Xe1$x1,y1)
plot(Xe2$x1,y2) 

# run the linear regression - notice the difference in residual standard error, 
# coefficient std. error, Rsquared. 
summary(lm(y1~x1+x2,data=Xe1)) 
summary(lm(y2~x1+x2,data=Xe2))



# 5. Residuals correlated with X's -----------------
# Residuals "e" are correlated with X variables (endogenous)

## 5a. Residuals (unobservable) covary with X's (observable) --------------
set.seed(826)
# Covariance matrix of x1, x2, and e 
n <- 3
A <- matrix(runif(n^2)*2-1, ncol=n) 
Sig <- t(A) %*% A
Sig       # Notice e covaries with x1 and x2
# also x1 and x2 can covary, but not perfectly (assumption 2)
# Mean of x1, x2, and e
moo <- c(10,3,0)
# generate data
Xe1 <- mvrnorm(n=1000,mu=moo,Sigma=Sig)
# give the variables names
colnames(Xe1)<-c("x1","x2","e1")
# store as a data frame
Xe1 <- as.data.frame(Xe1)
head(Xe1)

# plot empirical distribution of each: 
hist(Xe1$x1, breaks = 100, cex.main = 0.9)
hist(Xe1$x2, breaks = 100, cex.main = 0.9)
hist(Xe1$e1, breaks = 100, cex.main = 0.9)
# Sample correlations and covariances (notice difference from "truth")
cov(Xe1)
cor(Xe1)
scatterplotMatrix(Xe1)

# generate the true y (outcome)
y1 <- 10 - 6*Xe1$x1 + 5*Xe1$x2 + Xe1$e1

# not necessarily obvious differences in the plot
plot(Xe1$x1,y1)
plot(Xe1$x2,y1) 

# run the linear regression - notice the difference in residual standard error,
# coefficient std. error, Rsquared. 
summary(lm(y1~x1+x2,data=Xe1)) 


## 5b. Omitted variable -----------------
# An observable variable (that is correlated with included variables) was omitted 
set.seed(826)
# Covariance matrix of x1, x2, and e 
Sig <- matrix(c(4,1,0,1,2,0,0,0,1),3,3)
Sig       # Notice e does not covary with x1 or x2 (assumption 5) 
# also x1 and x2 can covary, but not perfectly (assumption 2)
# Mean of x1, x2, and e
moo <- c(10,3,0)
# generate data
Xe <- mvrnorm(n=1000,mu=moo,Sigma=Sig)
# give the variables names
colnames(Xe)<-c("x1","x2","e")
# store as a data frame
Xe <- as.data.frame(Xe)

# generate the true y (outcome)
y2 <- 10 - 6*Xe$x1 + 5*Xe$x2 + Xe$e 

# run the linear regression with an omitted variable
summary(lm(y2~x1,data=Xe))
summary(lm(y2~x2,data=Xe))


## 5c. autocorrelated Y and autocorrelated residual --------------
# y is autocorrelated, may or may not have serial correlation/autocorrelation 
#   in the residual.  
set.seed(826)
eps <- arima.sim(model=list(ar=c(0.8)),n=999,sd=1)
y1 <- list()
y10 <- rnorm(n=1,mean=10,sd=1)
y1[[1]] <- y10
for(i in 2:1000) {
  y1[[i]] <- 10 + 0.4*y1[[i-1]] + eps[i]
}
y1 <- unlist(y1)
arima(y1,order=c(1,0,0))


