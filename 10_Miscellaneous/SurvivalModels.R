# Duration or Survival Modeling from ISLR V2
# Introduction to Statistical Learning Version 2.0

install.packages("ISLR2")
install.packages("survival")
install.packages("coxed")
require(ISLR2)
require(survival)
require(coxed)

# Brain Cancer example ------------
names(BrainCancer)

# Examine the data
attach(BrainCancer)
table(sex)
table(diagnosis)
table(status)

BrainCancer <- BrainCancer

sapply(BrainCancer, class)
sapply(BrainCancer, typeof)

# Create Kaplan-Meier Survival Curve
fit.surv <- survfit(Surv(time, status) ~ 1, data=BrainCancer)
plot(fit.surv, xlab = "Months",
     ylab = "Estimated Probability of Survival")

fit.sex <- survfit(Surv(time, status) ~ sex, data=BrainCancer)
plot(fit.sex, xlab = "Months",
     ylab = "Estimated Probability of Survival", col = c(2,4))
legend("bottomleft", levels(BrainCancer$sex), col = c(2,4), lty = 1)

# log rank survival comparison of males to females
logrank.test <- survdiff(Surv(time, status) ~ sex)
logrank.test

# fit Cox Proportional Hazards Model
fit.cox <- coxph(Surv(time, status) ~ sex, data=BrainCancer)
# Hazard is 1.5 times greater for Men. 
summary(fit.cox)

# fit Cox Proportional Hazards Model with more covariates
fit.all <- coxph(
  Surv(time, status) ~ sex + diagnosis + loc + ki + gtv + stereo, data=BrainCancer)
summary(fit.all)

# Survival curves for each diagnosis category
# hold constant other variables at their means or a specific level
modaldata <- data.frame(
  diagnosis = levels(BrainCancer$diagnosis),
  sex = rep("Female", 4),
  loc = rep("Supratentorial", 4),
  ki = rep(mean(BrainCancer$ki), 4),
  gtv = rep(mean(BrainCancer$gtv), 4),
  stereo = rep("SRT", 4)
)

survplots <- survfit(fit.all, newdata = modaldata)
plot(survplots, xlab = "Months",
     ylab = "Survival Probability", col=c(2:5))
legend("bottomleft", levels(BrainCancer$diagnosis), col = c(2:5), lty = 1)

# Publication times example -----------------
fit.posres <- survfit(
  Surv(time, status) ~ posres , data = Publication
)
# Waiting time doesn't vary with ultimate outcome
plot(fit.posres , xlab = "Months",
       ylab = "Probability of Not Yet Published", col = 3:4)
legend("topright", c("Negative Result", "Positive Result"),
         col = 3:4, lty = 1)

fit.pub <- coxph(Surv(time, status) ~ posres ,
                 data = Publication)
summary(fit.pub)

logrank.test <- survdiff(Surv(time, status) ~ posres ,
                         data = Publication)
logrank.test

# results change with additional predictors
fit.pub2 <- coxph(Surv(time, status) ~ . - mech,
                  data = Publication)
fit.pub2 <- coxph(Surv(time, status) ~ posres + multi + clinend + sampsize + budget + impact,
                  data = Publication)
summary(fit.pub2)

# Survival curves for whether or not study has positive result
# hold constant other variables at their means or a specific level
modaldata <- data.frame(
  posres = levels(as.factor(Publication$posres)),
  multi = rep(0, 2),
  clinend = rep(0, 2),
  sampsize = rep(mean(Publication$sampsize), 2),
  budget = rep(mean(Publication$budget), 2),
  impact = rep(mean(Publication$impact), 2)
)

# Studies with positive results get published sooner
survplots <- survfit(fit.pub2, newdata = modaldata)
plot(survplots, xlab = "Months",
     ylab = "Survival Probability", col=c(2:3))
legend("bottomleft", levels(as.factor(Publication$posres)), col = c(2:3), lty = 1)


# Call Center Wait Times Example -------------

# This example uses simulated data

set.seed (4)
N <- 2000
Operators <- sample (5:15 , N, replace = T)   # Number of operators could be 5 to 15
Center <- sample(c("A", "B", "C"), N, replace = T) # 3 call centers available
Time <- sample(c("Morn.", "After.", "Even."), N, replace = T) # 3 times of day
X <- model.matrix( ~ Operators + Center + Time)[, -1]
X[1:5,]

# Coefficients and hazard function
true.beta <- c(0.04 , -0.3, 0, 0.2, -0.2)
h.fn <- function(x) return (0.00001 * x)

# set maximum possible wait time to 1000 seconds
queuing <- sim.survdata(N = N, T = 1000, X = X,
                          beta = true.beta , hazard.fun = h.fn)
names(queuing)
head(queuing$data)
#90% of calls are answered in 1000 seconds
mean(queuing$data$failed)

# kaplan meier survival curve, stratified by Center and then by Time of day
# Call Center B is the worst, while Evening is the best time to call
par(mfrow = c(1, 2))
fit.Center <- survfit(Surv(y, failed) ~ Center ,
                        data = queuing$data)
plot(fit.Center , xlab = "Seconds",
     ylab = "Probability of Still Being on Hold",
     col = c(2, 4, 5))
legend("topright",
         c("Call Center A", "Call Center B", "Call Center C"),
         col = c(2, 4, 5), lty = 1)
fit.Time <- survfit(Surv(y, failed) ~ Time ,
                    data = queuing$data)
plot(fit.Time , xlab = "Seconds",
       ylab = "Probability of Still Being on Hold",
       col = c(2, 4, 5))
legend("topright", c("Morning", "Afternoon", "Evening"),
         col = c(5, 2, 4), lty = 1)
par(mfrow = c(1, 1))

# Differences are statistically significant
survdiff(Surv(y, failed) ~ Center , data = queuing$data)
survdiff(Surv(y, failed) ~ Time , data = queuing$data)

# Fit model on all predictors
fit.queuing <- coxph(Surv(y, failed) ~ .,
                     data = queuing$data)
summary(fit.queuing)
