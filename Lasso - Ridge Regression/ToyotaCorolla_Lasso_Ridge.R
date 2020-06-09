library(readr)
corolla <- read_csv(file.choose())
View(corolla)

corolla <- corolla[c("Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight")]
View(corolla)

# Data Preparation & EDA

# checking for missing values

sum(is.na(corolla)) # No missing values

# 6 - point Summary

summary(corolla)
colnames(corolla)

attach(corolla)

library(dplyr)
summarise_if(corolla , is.numeric, sd) # Standard Deviation
summarise_if(corolla , is.numeric, max) - summarise_if(corolla , is.numeric, min) # Range

# Skewness
#install.packages("moments")
library(moments)
skewness(corolla)

#Kurtosis
kurtosis(corolla)

#Graphical Representation

#Distribution
dens <- apply(corolla , 2 , density)
plot(NA, xlim=range(sapply(dens, "[", "x")), ylim=range(sapply(dens, "[", "y")))
mapply(lines, dens, col=1:length(dens))
legend("topright", legend=names(dens), fill=1:length(dens))

#Box Plot
boxplot(corolla) # Outliers exists for Price, KM, CC

# Pair plot
pairs(corolla)

# Correlation Matrix
panel.cor <- function(x, y, digits=2, prefix="", cex.cor)
{
  #usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  r = (cor(x, y))
  txt <- format(c(r, 0.123456789), digits=digits)[1]
  txt <- paste(prefix, txt, sep="")
  if(missing(cex.cor)) cex <- 0.4/strwidth(txt)
  text(0.5, 0.5, txt, cex = cex)
}
pairs(corolla, upper.panel=panel.cor,main="Scatter Plot Matrix with Correlation Coefficients")


#Linear Model
model1 <- lm(Price ~ . , data = corolla)
summary(model1)

# Output Summary:
# 1. p-values of most of the coeffs are less than 0.05 but cc and Doors are insignificant
# R squared adj and R squared are closeby
# R squared value is 0.864 - Determination is strong 


# Diagnostic Plots
#install.packages("car")
library(car)
plot(model1)


# Deletion Diagnostics for identifying influential variable
influence.measures(model1)
influenceIndexPlot(model1, id.n=3) 
influencePlot(model1, id.n=3) # Points 81, 222 & 961 seems to be outliers


# Converting the data into format in which model accepts 
x <- model.matrix(Price~.-1,data=corolla)
y <- corolla$Price

library(glmnet)

# setting lamda as 10^10 till 10^-2
lambda <- 10^seq(10, -2, length = 50)

# Ridge Regression
ridge_reg <- glmnet(x,y,alpha=0,lambda=lambda)
summary(ridge_reg)

# Coefficients vs Lambda
plot(ridge_reg,xvar="lambda",label=T)

# ridge regression coefficients, stored in a matrix 
dim(coef(ridge_reg))
plot(ridge_reg)

# Cross Validation in Ridge Regression
cv_ridge = cv.glmnet(x, y, alpha = 0) 

# Select lamda that minimizes training MSE
bestlam = cv_ridge$lambda.min  
bestlam

# Draw plot of training MSE as a function of lambda
plot(cv_ridge) 

ridge_pred1 = predict(ridge_reg, s = bestlam, newx = x)
sqrt(mean((ridge_pred1 - y)^2)) # RMSE is 1384.5

ridge_pred1 = predict(ridge_reg, s = 0, newx = x)
sqrt(mean((ridge_pred1 - y)^2)) #RMSE is 1338.2 indicates basic regression is better than lasso regression


# LASSO Regression
lasso_mod = glmnet(x,y, alpha = 1, lambda = lambda)

plot(lasso_mod)    # Draw plot of coefficients

cv_lasso = cv.glmnet(x, y, alpha = 1) # Fit lasso model on training data

plot(cv_lasso) # Draw plot of training MSE as a function of lambda

bestlam_lasso = cv_lasso$lambda.min # Select lamda that minimizes training MSE
bestlam_lasso
# Use best lambda to predict test data
lasso_pred = predict(lasso_mod, s = bestlam_lasso, newx = x)

sqrt(mean((lasso_pred - y)^2)) # RMSE is 8981.8 which is is higher than basic multiple linear regression

# Display coefficients using lambda chosen by CV
lasso_coef = predict(lasso_mod, type = "coefficients", s = bestlam)[1:5,] 
lasso_coef


#######
# Partitioning Data into training set and testing set
# Spliting data into train and test
n=nrow(corolla)
n1=n*0.7
n2=n-n1
train=sample(1:n,n1)
test = corolla[-train,]

pred=predict(model1,newdat=test)
actual=test$Price
error=actual-pred

test.rmse=sqrt(mean(error**2))
test.rmse

train.rmse = sqrt(mean(model1$residuals**2))
train.rmse 

# Train error and Test error are similar So the model is having a good fit

