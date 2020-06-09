import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import pylab


# Opening file and assigning to an object
delivery = pd.read_csv('C:/Users/lnram/Desktop/ExcelR/Module 6 - Simple Linear Regression/Assignments/delivery_time.csv')

# Business Problem: Predict delivery_time with Max accuracy

#EDA/Descriptive Analytics:
# First Moment Business Decision - Measure of Central Tendancy

delivery.describe()

# Second Moment Business Decision - Measure of Dispersion

delivery.columns = "delivery_time" , "sort_time"

#Std Deviation
print("Std Deviation of delivery_time: " , delivery.delivery_time.std())
print("Std Deviation of sort_time: " , delivery.sort_time.std())


#Range
print("Range of delivery_time: " , max(delivery.delivery_time) - min(delivery.delivery_time))
print("Range of sort_time: " , max(delivery.sort_time) - min(delivery.sort_time))


# Third Moment Business Decision - Skewness
print("Skewness of delivery_time : " , delivery.delivery_time.skew())
print("Skewness of sort_time : " , delivery.sort_time.skew())


# Fourth Moment Business Decision - Kurtosis
print("Kurtosis of delivery_time : " , delivery.delivery_time.kurt())
print("Kurtosis of sort_time : " , delivery.sort_time.kurt())

# Graphical Representation - Distribution, Box Plot, Scatter Plot
sns.distplot(delivery.delivery_time)
sns.distplot(delivery.sort_time)

plt.boxplot(delivery.delivery_time)
plt.boxplot(delivery.sort_time)

plt.scatter(delivery.sort_time,delivery.delivery_time)

#Modeling - Linear Regression

import statsmodels.formula.api as smf

model1 = smf.ols('delivery_time ~ sort_time' , data = delivery).fit()
model1.summary()  # output Evaluation

# Model Evaluation & Assumptions

np.corrcoef(delivery.delivery_time,delivery.sort_time) # Correlation Coefficient

error = model1.predict(delivery.sort_time) - delivery.delivery_time
rmse = np.sqrt(np.mean(error * error)) # Root Mean Squared Value
rmse


stats.shapiro(error) # Shapiro test for normality
stats.probplot(error , dist = "norm" , plot = pylab) # QQplot

# Coeffs p-value is less than 0.05 - Coeffs values are stastically significant
# F Statistic p-value < 0.05 - Regression Eqn is statistically significant
# RMSE is very low
# Shapiro test on Residuals - p-val > 0.05, Residuals are normal
# But the R Squared value is only 0.68


#Modeling - Linear Regression - Log


np.corrcoef(delivery.delivery_time,np.log(delivery.sort_time)) # Correlation Coefficient
plt.scatter(delivery.delivery_time,np.log(delivery.sort_time))


model2 = smf.ols('delivery_time ~ np.log(sort_time)' , data = delivery).fit()
model2.summary()  # output Evaluation

# Model Evaluation & Assumptions
error2 = model2.predict(delivery.sort_time) - delivery.delivery_time
rmse2 = np.sqrt(np.mean(error2 * error2)) # Root Mean Squared Value
rmse2


stats.shapiro(error2) # Shapiro test for normality
stats.probplot(error2 , dist = "norm" , plot = pylab) # QQplot

# Coeffs p-value is less than 0.05 - Coeffs values are stastically significant
# F Statistic p-value < 0.05 - Regression Eqn is statistically significant
# RMSE is very low
# Shapiro test on Residuals - p-val > 0.05, Residuals are normal
# But the R Squared value is only 0.69



#Modeling - Linear Regression - Exponential


np.corrcoef(np.log(delivery.delivery_time),delivery.sort_time) # Correlation Coefficient
plt.scatter(np.log(delivery.delivery_time), delivery.sort_time)

model3 = smf.ols('np.log(delivery_time) ~ sort_time' , data = delivery).fit()
model3.summary()  # output Evaluation

# Model Evaluation & Assumptions
error3 = np.exp(model3.predict(delivery.sort_time)) - delivery.delivery_time
rmse3 = np.sqrt(np.mean(error3 * error3)) # Root Mean Squared Value
rmse3


stats.shapiro(error3) # Shapiro test for normality
stats.probplot(error3 , dist = "norm" , plot = pylab) # QQplot


# Coeffs p-value is less than 0.05 - Coeffs values are stastically significant
# F Statistic p-value < 0.05 - Regression Eqn is statistically significant
# RMSE is very low
# Shapiro test on Residuals - p-val > 0.05, Residuals are normal
# R Squared value is 0.71



#Modeling - Polynomial Regression


np.corrcoef(delivery.delivery_time,delivery.sort_time) # Correlation Coefficient
plt.scatter(delivery.delivery_time, delivery.sort_time)

model4 = smf.ols('delivery_time ~ sort_time + I(sort_time * sort_time)' , data = delivery).fit()
model4.summary()  # output Evaluation

# Model Evaluation & Assumptions
error4 = model4.predict(delivery.sort_time) - delivery.delivery_time
rmse4 = np.sqrt(np.mean(error4 * error4)) # Root Mean Squared Value
rmse4


stats.shapiro(error4) # Shapiro test for normality
stats.probplot(error4 , dist = "norm" , plot = pylab) # QQplot




# Coeffs p-value > 0.05 - Coeffs values are stastically insignificant
# F Statistic p-value < 0.05 - Regression Eqn is statistically significant
# RMSE is very low
# Shapiro test on Residuals - p-val > 0.05, Residuals are normal
# R Squared value is 0.69



model5 = smf.ols('np.log(delivery_time) ~ np.log(sort_time)' , data = delivery).fit()
model5.summary()  # output Evaluation

# Model Evaluation & Assumptions
error5 = model5.predict(delivery.sort_time) - delivery.delivery_time
rmse5 = np.sqrt(np.mean(error5 * error5)) # Root Mean Squared Value
rmse5


stats.shapiro(error5) # Shapiro test for normality
stats.probplot(error5 , dist = "norm" , plot = pylab) # QQplot

# Coeffs p-value is less than 0.05 - Coeffs values are stastically significant
# F Statistic p-value < 0.05 - Regression Eqn is statistically significant
# RMSE is very low
# Shapiro test on Residuals - p-val > 0.05, Residuals are normal
# R Squared value is 0.77

final_model = smf.ols('np.log(delivery_time) ~ np.log(sort_time)' , data = delivery).fit()
final_model.summary()  # output Evaluation


final_model.conf_int(0.05) # CI of Coeffs at 95% limit
final_model.predict(pd.DataFrame(delivery.sort_time)) # Predicting delivery_time based on confidence intervals
