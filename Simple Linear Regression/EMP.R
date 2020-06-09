# Opening file and assigning to an object
emp <- read.csv(choose.files())
View(emp)

# Business Problem: Predict Churn out with Max accuracy

#EDA/Descriptive Analytics:
# First Moment Business Decision - Measure of Central Tendancy

summary(emp)

# Second Moment Business Decision - Measure of Dispersion

colnames(emp) <- c("hike" , "churn")
attach(emp)

#Std Deviation
print(paste("Std Deviation of hike: " , sd(hike)))
print(paste("Std Deviation of churn: " , sd(churn)))

#Variance
print(paste("Variance of hike: " , sd(hike)^2))
print(paste("Variance of churn: " , sd(churn)^2))

#Range
print(paste("Range of hike: " , max(hike) - min(hike)))
print(paste("Range of churn: " , max(churn) - min(churn)))

# Third Moment Business Decision - Skewness
library(moments)
print(paste("Skewness of hike : " , skewness(hike)))
print(paste("Skewness of churn : " , skewness(churn)))

# Fourth Moment Business Decision - Kurtosis
print(paste("Kurtosis of hike : " , kurtosis(hike)))
print(paste("Kurtosis of churn : " , kurtosis(churn)))

# Graphical Representation - Distribution, Box Plot, Scatter Plot
plot(density(hike))
plot(density(churn))

boxplot(hike)
boxplot(churn)

plot(churn,hike)

#Modeling - Linear Regression

model1 <- lm(churn ~ hike)
summary(model1) # output Evaluation

# Model Evaluation & Assumptions

cor(hike,churn) # Correlation Coefficient

error <- predict(model1) - churn
rmse <- sqrt(mean(error^2)) # Root Mean Squared Value
rmse

shapiro.test(model1$residuals) # Shapiro test for normality
qqnorm(model1$residuals) # QQplot
qqline(model1$residuals) # QQLine

# Coeffs p-value is less than 0.05 - Coeffs values are stastically significant
# F Statistic p-value < 0.05 - Regression Eqn is statistically significant
# RMSE is very low
# Shapiro test on Residuals - p-val > 0.05, Residuals are normal
# R Squared value is only 0.83
# Correlation Coefficient is - 0.91 indicates strong inversely relationship


#Modeling - Linear Regression - Log


cor(log(hike),churn) # Correlation Coefficient
plot(log(hike),churn)

model2 <- lm(churn ~ log(hike))
summary(model2) # output Evaluation

# Model Evaluation & Assumptions

error <- predict(model2) - churn
rmse <- sqrt(mean(error^2)) # Root Mean Squared Value
rmse

shapiro.test(model2$residuals) # Shapiro test for normality
qqnorm(model2$residuals) # QQplot
qqline(model2$residuals) # QQLine


# Coeffs p-value is less than 0.05 - Coeffs values are stastically significant
# F Statistic p-value < 0.05 - Regression Eqn is statistically significant
# RMSE is very low
# Shapiro test on Residuals - p-val > 0.05, Residuals are normal
# R Squared value is 0.848
# Correlation Coefficient is - 0.92 indicates strong inversely relationship



#Modeling - Linear Regression - Exponential


cor(log(churn),hike) # Correlation Coefficient
plot(log(churn), hike)

model3 <- lm(log(churn) ~ hike)
summary(model3) # output Evaluation

# Model Evaluation & Assumptions

error <- exp(predict(model3 , level = "confidence")) - churn
rmse <- sqrt(mean(error^2)) # Root Mean Squared Value
rmse

shapiro.test(model3$residuals) # Shapiro test for normality
qqnorm(model3$residuals) # QQplot
qqline(model3$residuals) # QQLine

# Coeffs p-value is less than 0.05 - Coeffs values are stastically significant
# F Statistic p-value < 0.05 - Regression Eqn is statistically significant
# RMSE is very low
# Shapiro test on Residuals - p-val > 0.05, Residuals are normal
# R Squared value is 0.873
# Correlation Coefficient is - 0.934 indicates strong inversely relationship



#Modeling - Polynomial Regression


cor(hike,churn) # Correlation Coefficient
plot(hike, churn)

model4 <- lm(churn ~ hike + I(hike * hike))
summary(model4) # output Evaluation

# Model Evaluation & Assumptions

error <- predict(model4 , level = "confidence") - churn
rmse <- sqrt(mean(error^2)) # Root Mean Squared Value
rmse

shapiro.test(model4$residuals) # Shapiro test for normality
qqnorm(model4$residuals) # QQplot
qqline(model4$residuals) # QQLine

# Coeffs p-value is less than 0.05 - Coeffs values are stastically significant
# F Statistic p-value < 0.05 - Regression Eqn is statistically significant
# RMSE is very low
# Shapiro test on Residuals - p-val > 0.05, Residuals are normal
# R Squared value is 9737


final_model <- lm(churn ~ hike + I(hike * hike))
summary(final_model)

confint(final_model) # CI of Coeffs at 95% limit
predict(final_model , interval = "confidence") # Predicting hike based on confidence intervals
