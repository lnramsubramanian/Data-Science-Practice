# Opening file and assigning to an object
delivery <- read.csv(choose.files())
View(delivery)

# Business Problem: Predict delivery_time with Max accuracy

#EDA/Descriptive Analytics:
# First Moment Business Decision - Measure of Central Tendancy

summary(delivery)

# Second Moment Business Decision - Measure of Dispersion

colnames(delivery) <- c("delivery_time" , "sort_time")
attach(delivery)

#Std Deviation
print(paste("Std Deviation of delivery_time: " , sd(delivery_time)))
print(paste("Std Deviation of sort_time: " , sd(sort_time)))

#Variance
print(paste("Variance of delivery_time: " , sd(delivery_time)^2))
print(paste("Variance of sort_time: " , sd(sort_time)^2))

#Range
print(paste("Range of delivery_time: " , max(delivery_time) - min(delivery_time)))
print(paste("Range of sort_time: " , max(sort_time) - min(sort_time)))

# Third Moment Business Decision - Skewness
library(moments)
print(paste("Skewness of delivery_time : " , skewness(delivery_time)))
print(paste("Skewness of sort_time : " , skewness(sort_time)))

# Fourth Moment Business Decision - Kurtosis
print(paste("Kurtosis of delivery_time : " , kurtosis(delivery_time)))
print(paste("Kurtosis of sort_time : " , kurtosis(sort_time)))

# Graphical Representation - Distribution, Box Plot, Scatter Plot
plot(density(delivery_time))
plot(density(sort_time))

boxplot(delivery_time)
boxplot(sort_time)

plot(sort_time,delivery_time)

#Modeling - Linear Regression

model1 <- lm(delivery_time ~ sort_time)
summary(model1) # output Evaluation

# Model Evaluation & Assumptions

cor(delivery_time,sort_time) # Correlation Coefficient

error <- predict(model1) - delivery_time
rmse <- sqrt(mean(error^2)) # Root Mean Squared Value
rmse

shapiro.test(model1$residuals) # Shapiro test for normality
qqnorm(model1$residuals) # QQplot
qqline(model1$residuals) # QQLine

# Coeffs p-value is less than 0.05 - Coeffs values are stastically significant
# F Statistic p-value < 0.05 - Regression Eqn is statistically significant
# RMSE is very low
# Shapiro test on Residuals - p-val > 0.05, Residuals are normal
# But the R Squared value is only 0.68


#Modeling - Linear Regression - Log


cor(delivery_time,log(sort_time)) # Correlation Coefficient
plot(delivery_time,log(sort_time))

model2 <- lm(delivery_time ~ log(sort_time))
summary(model2) # output Evaluation

# Model Evaluation & Assumptions

error <- predict(model2) - delivery_time
rmse <- sqrt(mean(error^2)) # Root Mean Squared Value
rmse

shapiro.test(model2$residuals) # Shapiro test for normality
qqnorm(model2$residuals) # QQplot
qqline(model2$residuals) # QQLine

# Coeffs p-value is less than 0.05 - Coeffs values are stastically significant
# F Statistic p-value < 0.05 - Regression Eqn is statistically significant
# RMSE is very low
# Shapiro test on Residuals - p-val > 0.05, Residuals are normal
# But the R Squared value is only 0.69



#Modeling - Linear Regression - Exponential


cor(log(delivery_time),sort_time) # Correlation Coefficient
plot(log(delivery_time), sort_time)

model3 <- lm(log(delivery_time) ~ sort_time)
summary(model3) # output Evaluation

# Model Evaluation & Assumptions

error <- exp(predict(model3 , level = "confidence")) - delivery_time
rmse <- sqrt(mean(error^2)) # Root Mean Squared Value
rmse

shapiro.test(model3$residuals) # Shapiro test for normality
qqnorm(model3$residuals) # QQplot
qqline(model3$residuals) # QQLine

# Coeffs p-value is less than 0.05 - Coeffs values are stastically significant
# F Statistic p-value < 0.05 - Regression Eqn is statistically significant
# RMSE is very low
# Shapiro test on Residuals - p-val > 0.05, Residuals are normal
# R Squared value is 0.71



#Modeling - Polynomial Regression


cor(delivery_time,sort_time) # Correlation Coefficient
plot(delivery_time, sort_time)

model4 <- lm(delivery_time ~ sort_time + I(sort_time * sort_time))
summary(model4) # output Evaluation

# Model Evaluation & Assumptions

error <- predict(model4 , level = "confidence") - delivery_time
rmse <- sqrt(mean(error^2)) # Root Mean Squared Value
rmse

shapiro.test(model4$residuals) # Shapiro test for normality
qqnorm(model4$residuals) # QQplot
qqline(model4$residuals) # QQLine

# Coeffs p-value > 0.05 - Coeffs values are stastically insignificant
# F Statistic p-value < 0.05 - Regression Eqn is statistically significant
# RMSE is very low
# Shapiro test on Residuals - p-val > 0.05, Residuals are normal
# R Squared value is 0.69




#Modeling - Linear Regression - Log Transform of X & Y 


cor(log(delivery_time),log(sort_time)) # Correlation Coefficient
plot(log(delivery_time), log(sort_time))

model5 <- lm(log(delivery_time) ~ log(sort_time))
summary(model5) # output Evaluation

# Model Evaluation & Assumptions

error <- exp(predict(model3 , level = "confidence")) - delivery_time
rmse <- sqrt(mean(error^2)) # Root Mean Squared Value
rmse

shapiro.test(model3$residuals) # Shapiro test for normality
qqnorm(model3$residuals) # QQplot
qqline(model3$residuals) # QQLine

# Coeffs p-value is less than 0.05 - Coeffs values are stastically significant
# F Statistic p-value < 0.05 - Regression Eqn is statistically significant
# RMSE is very low
# Shapiro test on Residuals - p-val > 0.05, Residuals are normal
# R Squared value is 0.77

final_model <- lm(log(delivery_time) ~ log(sort_time))
summary(final_model)

confint(final_model) # CI of Coeffs at 95% limit
predict(final_model , interval = "confidence") # Predicting delivery_time based on confidence intervals
