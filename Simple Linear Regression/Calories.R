# Opening file and assigning to an object
cal <- read.csv(choose.files())
View(cal)

# Business Problem: Predict Weight with Max accuracy

#EDA/Descriptive Analytics:
# First Moment Business Decision - Measure of Central Tendancy

summary(cal)

# Second Moment Business Decision - Measure of Dispersion

colnames(cal) <- c("weight" , "calories")
attach(cal)

#Std Deviation
print(paste("Std Deviation of Weight: " , sd(weight)))
print(paste("Std Deviation of Calories: " , sd(calories)))

#Variance
print(paste("Variance of Weight: " , sd(weight)^2))
print(paste("Variance of Calories: " , sd(calories)^2))

#Range
print(paste("Range of Weight: " , max(weight) - min(weight)))
print(paste("Range of Calories: " , max(calories) - min(calories)))

# Third Moment Business Decision - Skewness
library(moments)
print(paste("Skewness of Weight : " , skewness(weight)))
print(paste("Skewness of Calories : " , skewness(calories)))

# Fourth Moment Business Decision - Kurtosis
print(paste("Kurtosis of Weight : " , kurtosis(weight)))
print(paste("Kurtosis of Calories : " , kurtosis(calories)))

# Graphical Representation - Distribution, Box Plot, Scatter Plot
plot(density(weight))
plot(density(calories))

boxplot(weight)
boxplot(calories)

plot(calories,weight)

#Modeling - Linear Regression

model1 <- lm(weight ~ calories)
summary(model1) # output Evaluation

# Model Evaluation & Assumptions

cor(weight,calories) # Correlation Coefficient
error <- predict(model1 , level = "confidence") - weight

rmse <- sqrt(mean(error^2)) # Root Mean Squared Value
rmse

shapiro.test(model1$residuals) # Shapiro test for normality
qqnorm(model1$residuals) # QQplot
qqline(model1$residuals) # QQLine

# Coeffs p-value is less than 0.05 - Coeffs values are stastically significant
# F Statistic p-value < 0.05 - Regression Eqn is statistically significant
# RMSE is very low
# Shapiro test on Residuals - p-val > 0.05, Residuals are normal

confint(model1) # CI of Coeffs at 95% limit
predict(model1 , interval = "confidence") # Predicting weight based on confidence intervals

#Log Transform
model2 <- lm(weight ~ log(calories))
summary(model2) # output Evaluation

# Model Evaluation & Assumptions

cor(weight,log(calories)) # Correlation Coefficient
error <- predict(model2 , level = "confidence") - weight

rmse <- sqrt(mean(error^2)) # Root Mean Squared Value
rmse

