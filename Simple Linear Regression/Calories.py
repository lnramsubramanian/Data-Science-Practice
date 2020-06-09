import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pylab

# Business Problem: Predict Weight with Max accuracy

#EDA/Descriptive Analytics:
# First Moment Business Decision - Measure of Central Tendancy

cal.describe()

# Second Moment Business Decision - Measure of Dispersion

cal.columns = "weight" , "calories"

#Std Deviation
print("Std Deviation of Weight: " , cal.weight.std())
print("Std Deviation of Calories: " , cal.calories.std())

#Range
print("Range of Weight: " , max(cal.weight) - min(cal.weight))
print("Range of Calories: " , max(cal.calories) - min(cal.calories))

# Third Moment Business Decision - Skewness
print("Skewness of Weight : " , cal.weight.skew())
print("Skewness of Calories : " , cal.calories.skew())

# Fourth Moment Business Decision - Kurtosis
print("Kurtosis of Weight : " , cal.weight.kurt())
print("Kurtosis of Calories : " , cal.calories.kurt())

# Graphical Representation - Distribution, Box Plot, Scatter Plot
sns.distplot(cal.weight)
sns.distplot(cal.calories)

plt.boxplot(cal.weight)
plt.boxplot(cal.calories)

plt.scatter(cal.calories,cal.weight)

#Modeling - Linear Regression
import statsmodels.formula.api as smf

model1 = smf.ols('weight ~ calories' , data = cal).fit()
model1.summary()  # output Evaluation

# Model Evaluation & Assumptions


np.corrcoef(cal.weight,cal.calories) # Correlation Coefficient
error = model1.predict(cal.calories) - cal.weight

rmse = np.sqrt(np.mean(error * error)) # Root Mean Squared Value
rmse


stats.shapiro(error) # Shapiro test for normality
stats.probplot(error , dist = "norm" , plot = pylab) # QQplot


# Coeffs p-value is less than 0.05 - Coeffs values are stastically significant
# F Statistic p-value < 0.05 - Regression Eqn is statistically significant
# RMSE is very low
# Shapiro test on Residuals - p-val > 0.05, Residuals are normal

model1.conf_int(0.05) # CI of Coeffs at 95% limit
model1.predict(cal.calories) # Predicting cal.weight based on confidence intervals

#Log Transform
model2 = smf.ols('weight ~ np.log(calories)' , data = cal).fit()
model2.summary()  # output Evaluation

# Model Evaluation & Assumptions


np.corrcoef(cal.weight,np.log(cal.calories)) # Correlation Coefficient

error = model2.predict(cal.calories) - cal.weight

rmse = np.sqrt(np.mean(error * error)) # Root Mean Squared Value
rmse


