# Preamble -------------

# Teaching program to show LLN and CLT applied to 
  # mean, variance, skewness, kurtosis, regression coefficients
  # individual and joint normality tests on simulated and financial data

# Skills covered:
  # simulating data and calculating moments
  # plotting histogram and fitting density
  # adding lines to plots
  # writing a loop to fill an empty list
  # run and summarize linear regression coefficients
  # use regression coefficients in later analysis

if(!require(pacman)) install.packages("pacman")
pacman::p_load(fBasics,quantmod,MASS)

# Show LLN and CLT of the first four moments of simulated data 
# do CLT for both Normal and non-Normal (Bernoulli) samples

# Define the population ---------------

# Set the seed
set.seed(12345)
popMean = 2.5
popSD = 0.5
popSkew = 0.0
popKurt = 0.0 # Subtract 3 to get the kurtosis associated with the sample data
X <- rnorm(1000000,mean = popMean, sd=popSD)
hist(X, breaks=100)
abline(v = popMean, lwd = 3, lty = 2)

# Law of Large Numbers (LLN) -----------------

# Calculate moments for increasingly large samples
mean(sample(X,10, replace = TRUE))
mean(sample(X,100, replace = TRUE))
mean(sample(X,1000, replace = TRUE))
mean(sample(X,10000, replace = TRUE))
sd(sample(X,10, replace = TRUE))
sd(sample(X,100, replace = TRUE))
sd(sample(X,1000, replace = TRUE))
sd(sample(X,10000, replace = TRUE))
skewness(sample(X,10, replace = TRUE))
skewness(sample(X,100, replace = TRUE))
skewness(sample(X,1000, replace = TRUE))
skewness(sample(X,10000, replace = TRUE))
kurtosis(sample(X,10, replace = TRUE),method = "moment")
kurtosis(sample(X,100, replace = TRUE),method = "moment")
kurtosis(sample(X,1000, replace = TRUE),method = "moment")
kurtosis(sample(X,10000, replace = TRUE),method = "moment")


# Central Limit Theorem -------------------

# Now for CLT. Take repeated samples from X all of the same size.
# 10,000 samples of size 10, compare to samples of size 100, 1000, 10000 etc.
# With many samples of increasingly large size... 
# ...the distribution of the sample statistics will converge to Normal.

## Take Samples ------------

realz = 10000 # number of samples or bootstrap realizations
N = 1000      # set to 10, 100, 1000, 10000 
mean_list <- list()
sd_list  <- list()
skew_list <- list()
kurt_list <- list()
for(i in 1:realz) {
  mean_list[[i]] <- mean(sample(X,N, replace = TRUE))
  sd_list[[i]] <- sd(sample(X,N, replace = TRUE))
  skew_list[[i]] <- skewness(sample(X,N, replace = TRUE))
  kurt_list[[i]] <- kurtosis(sample(X,N, replace = TRUE))
}

## "True" Distributions of Sample Statistics -----------

# compute population distribution for sample mean 
# standard error of the mean is (Std. Dev of sample/square root N)
# x = seq(0,5,0.001) says make x-axis from 0 to 5 in increments of 0.001. 
mu = dnorm(x=seq(0,5,0.001), mean=popMean, sd=popSD/sqrt(N))

# compute population distribution for stdev
# trick for inverse chi-squared since R doesn't have one by default
# integrate to get the CDF
s = seq(0,1,0.001)
sigmaCDF = sqrt(((N-1)*popSD*popSD)/qchisq(p=s, df=N-1))
# then differentiate to get the PDF
sigma = numeric(length(s)-1)
for (i in 2:length(s)){
  sigma[i] = (s[i]-s[i-1])/(sigmaCDF[i-1]-sigmaCDF[i])
}

# Skew and kurtosis are Normally distributed with sd = sqrt(6/N) and sd = sqrt(24/N), respectively.
g = dnorm(x=seq(-3,3,0.001), mean=popSkew, sd=sqrt(6/N)) # Skewness 
g2 = dnorm(x=seq(-3,3,0.001), mean=popKurt, sd=sqrt(24/N)) # Kurtosis 

## Distribution of Mean ----------------
hist(unlist(mean_list), breaks = 500, 
     xlab = "Mean of 10,000 samples from X of size 10", 
     main = "CLT of sample mean",
     cex.main = 0.8)
abline(v = popMean, lwd = 3, col = "white", lty = 2)
# Distribution of the population vs sample mean
plot(density(unlist(mean_list)),ylim=c(0,max(mu)), xlab="Mean", ylab="Density", main="Distribution of the Mean")
points(seq(0,5,0.001), mu, type='l', lwd = 2, col = "red")
abline(v = mean(unlist(mean_list)), lwd = 1, col = "black", lty = 2)
abline(v = popMean, lwd = 3, col = "red", lty = 2)
legend("topright", legend=c("bootstrap density", "theoretical density", "bootstrap mean value", "population value"), 
       col=c("black", "red", "black", "red"), lty=c(1,1,2,2), cex=0.8)

## Distribution of Std. Dev. --------------
hist(unlist(var_list), breaks = 500, 
     xlab = "Variance of 10,000 samples from X of size 10", 
     main = "CLT of sample variance",
     cex.main = 0.8)
abline(v = popSD^2, lwd = 3, col = "white", lty = 2)
# Distribution of the population vs. sample standard deviation
plot(density(unlist(sd_list)), ylim=c(0,max(sigma)), xlab="Standard Deviation", ylab="Density", main="Distribution of the Std. Dev")
points(sigmaCDF, sigma, type='l', lwd = 2, col = "red")
abline(v = mean(unlist(sd_list)), lwd = 1, col = "black", lty = 2)
abline(v = popSD, lwd = 3, col = "red", lty = 2)
legend("topright", legend=c("bootstrap density", "theoretical density", "bootstrap mean value", "population value"), 
       col=c("black", "red", "black", "red"), lty=c(1,1,2,2), cex=0.8)

## Distribution of Skewness ----------------
hist(unlist(skew_list), breaks = 500, 
     xlab = "Skewness of 10,000 samples from X of size 10", 
     main = "CLT of sample skewness",
     cex.main = 0.8)
abline(v = popSkew, lwd = 3, col = "white", lty = 2)
# Distribution of the population vs sample skew
plot(density(unlist(skew_list)), ylim=c(0,max(g)), xlab="Skewness", ylab="Density", main="Distribution of the Skewness")
points(seq(-3,3,0.001), g, type='l', lwd = 2, col = "red")
abline(v = mean(unlist(skew_list)), lwd = 1, col = "black", lty = 2)
abline(v = popSkew, lwd = 3, col = "red", lty = 2)
legend("topright", legend=c("bootstrap density", "theoretical density", "bootstrap mean value", "population value"), 
       col=c("black", "red", "black", "red"), lty=c(1,1,2,2), cex=0.8)

## Distribution of Kurtosis ---------------
hist(unlist(kurt_list), breaks = 500, 
     xlab = "Kurtosis of 10,000 samples from X of size 10", 
     main = "CLT of sample kurtosis",
     cex.main = 0.8)
abline(v = popKurt, lwd = 3, col = "white", lty = 2)
# Distribution of the population vs sample kurtosis
plot(density(unlist(kurt_list)), ylim=c(0,max(g2)), xlab="Kurtosis", ylab="Density", main="Distribution of the Kurtosis")
points(seq(-3,3,0.001), g2, type='l', lwd = 2, col = "red")
abline(v = mean(unlist(kurt_list)), lwd = 1, col = "black", lty = 2)
abline(v = popKurt, lwd = 3, col = "red", lty = 2)
legend("topright", legend=c("bootstrap density", "theoretical density", "bootstrap mean value", "population value"), 
       col=c("black", "red", "black", "red"), lty=c(1,1,2,2), cex=0.8)

## CLT for Non-Normal Distribution --------------

# Show that this works even if we weren't sampling from a Normal distribution
# sample from coin flips
population <- sample(c(0,1), 1000000, replace = TRUE)
hist(population, main = "Non-normal", cex.main = 0.8)
abline(v = mean(population), lwd = 3, lty = 3)

mean_list <- list()
for(i in 1:realz) {
  mean_list[[i]] <- mean(sample(population, N, replace = TRUE))
}
hist(unlist(mean_list), main = "Distribution of averages", cex.main = 0.8, xlab = "Average of samples with N=100")
abline(v = 0.5, lwd = 3)

# LLN and CLT for Regression Coefficients -----------------

## Create the population ---------------

eps <- rnorm(1000000, mean=0, sd=2)
beta0 = 0.5
beta1 = 0.8
Y <- beta0 + beta1*X + eps

summary(lm(Y~X))

## LLN for coefficients ------------------
Z = cbind(Y,X)
s10 <- Z[sample(nrow(Z),10,replace=TRUE),]
summary(lm(s10[,1]~s10[,2]))
s100 <- Z[sample(nrow(Z),100,replace=TRUE),]
summary(lm(s100[,1]~s100[,2]))
s1000 <- Z[sample(nrow(Z),1000,replace=TRUE),]
summary(lm(s1000[,1]~s1000[,2]))
s10000 <- Z[sample(nrow(Z),10000,replace=TRUE),]
summary(lm(s10000[,1]~s10000[,2]))

## CLT for coefficients ------------------
b0_list <- list()
b1_list <- list()
for(i in 1:realz) {
  sNx <- Z[sample(nrow(Z),N,replace=TRUE),]
  b0_list[[i]] <- lm(sNx[,1]~sNx[,2])$coefficients[1]
  b1_list[[i]] <- lm(sNx[,1]~sNx[,2])$coefficients[2]
}
hist(unlist(b0_list), main = "Distribution of intercepts", cex.main = 0.8, xlab = "Intercepts with N=100")
abline(v = beta0, lwd = 3)
hist(unlist(b1_list), main = "Distribution of slopes", cex.main = 0.8, xlab = "Slopes with N=100")
abline(v = beta1, lwd = 3)
