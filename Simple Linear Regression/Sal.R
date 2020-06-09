# Opening file and assigning to an object
sal <- read.csv(choose.files())
View(sal)

# Business Problem: Predict salary out with Max accuracy

#EDA/Descriptive Analytics:
# First Moment Business Decision - Measure of Central Tendancy

summary(sal)

# Second Moment Business Decision - Measure of Dispersion

colnames(sal) <- c("exp" , "salary")
attach(sal)

#Std Deviation
print(paste("Std Deviation of exp: " , sd(exp)))
print(paste("Std Deviation of salary: " , sd(salary)))

#Variance
print(paste("Variance of exp: " , sd(exp)^2))
print(paste("Variance of salary: " , sd(salary)^2))

#Range
print(paste("Range of exp: " , max(exp) - min(exp)))
print(paste("Range of salary: " , max(salary) - min(salary)))

# Third Moment Business Decision - Skewness
library(moments)
print(paste("Skewness of exp : " , skewness(exp)))
print(paste("Skewness of salary : " , skewness(salary)))

# Fourth Moment Business Decision - Kurtosis
print(paste("Kurtosis of exp : " , kurtosis(exp)))
print(paste("Kurtosis of salary : " , kurtosis(salary)))

# Graphical Representation - Distribution, Box Plot, Scatter Plot
plot(density(exp))
plot(density(salary))

boxplot(exp)
boxplot(salary)

plot(salary,exp)

#Modeling - Linear Regression

model1 <- lm(salary ~ exp)
summary(model1) # output Evaluation

# Model Evaluation & Assumptions

cor(exp,salary) # Correlation Coefficient

error <- predict(model1) - salary
rmse <- sqrt(mean(error^2)) # Root Mean Squared Value
rmse

shapiro.test(model1$residuals) # Shapiro test for normality
qqnorm(model1$residuals) # QQplot
qqline(model1$residuals) # QQLine

# Coeffs p-value is less than 0.05 - Coeffs values are stastically significant
# F Statistic p-value < 0.05 - Regression Eqn is statistically significant
# RMSE is very low
# Shapiro test on Residuals - p-val > 0.05, Residuals are normal
# R Squared value is only 0.957
# Correlation Coefficient is 0.978 indicates strong positive relationship


#Modeling - Linear Regression - Log


cor(log(exp),salary) # Correlation Coefficient
plot(log(exp),salary)

model2 <- lm(salary ~ log(exp))
summary(model2) # output Evaluation

# Model Evaluation & Assumptions

error <- predict(model2) - salary
rmse <- sqrt(mean(error^2)) # Root Mean Squared Value
rmse

shapiro.test(model2$residuals) # Shapiro test for normality
qqnorm(model2$residuals) # QQplot
qqline(model2$residuals) # QQLine


# Coeffs p-value is less than 0.05 - Coeffs values are stastically significant
# F Statistic p-value < 0.05 - Regression Eqn is statistically significant
# RMSE is very low
# Shapiro test on Residuals - p-val > 0.05, Residuals are normal
# R Squared value is 0.8539
# Correlation Coefficient is - 0.924 indicates strong positive relationship



#Modeling - Linear Regression - Exponential


cor(log(salary),exp) # Correlation Coefficient
plot(log(salary), exp)

model3 <- lm(log(salary) ~ exp)
summary(model3) # output Evaluation

# Model Evaluation & Assumptions

error <- exp(predict(model3 , level = "confidence")) - salary
rmse <- sqrt(mean(error^2)) # Root Mean Squared Value
rmse

shapiro.test(model3$residuals) # Shapiro test for normality
qqnorm(model3$residuals) # QQplot
qqline(model3$residuals) # QQLine

# Coeffs p-value is less than 0.05 - Coeffs values are stastically significant
# F Statistic p-value < 0.05 - Regression Eqn is statistically significant
# RMSE is very low
# Shapiro test on Residuals - p-val > 0.05, Residuals are normal
# R Squared value is 0.93
# Correlation Coefficient is - 0.96 indicates strong positive relationship


#Modeling - Polynomial Regression


cor(exp,salary) # Correlation Coefficient
plot(exp, salary)

model4 <- lm(salary ~ exp + I(exp * exp))
summary(model4) # output Evaluation

# Model Evaluation & Assumptions

error <- predict(model4 , level = "confidence") - salary
rmse <- sqrt(mean(error^2)) # Root Mean Squared Value
rmse

shapiro.test(model4$residuals) # Shapiro test for normality
qqnorm(model4$residuals) # QQplot
qqline(model4$residuals) # QQLine

# Coeffs p-value is less than 0.05 - Coeffs values are stastically significant
# F Statistic p-value < 0.05 - Regression Eqn is statistically significant
# RMSE is very low
# Shapiro test on Residuals - p-val > 0.05, Residuals are normal
# R Squared value is 0.957

final_model <- lm(salary ~ exp + I(exp * exp))
summary(final_model)

confint(final_model) # CI of Coeffs at 95% limit
predict(final_model , interval = "confidence") # Predicting exp based on confidence intervals
