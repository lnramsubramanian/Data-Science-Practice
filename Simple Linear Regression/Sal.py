import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import pylab


# Opening file and assigning to an object
sal = pd.read_csv('C:/Users/lnram/Desktop/ExcelR/Module 6 - Simple Linear Regression/Assignments/Salary_Data.csv')

# Business Problem: Predict salary with Max accuracy

#EDA/Descriptive Analytics:
# First Moment Business Decision - Measure of Central Tendancy

sal.describe()

# Second Moment Business Decision - Measure of Dispersion

sal.columns = "salary" , "exp"

#Std Deviation
print("Std Deviation of salary: " , sal.salary.std())
print("Std Deviation of exp: " , sal.exp.std())


#Range
print("Range of salary: " , max(sal.salary) - min(sal.salary))
print("Range of exp: " , max(sal.exp) - min(sal.exp))


# Third Moment Business Decision - Skewness
print("Skewness of salary : " , sal.salary.skew())
print("Skewness of exp : " , sal.exp.skew())


# Fourth Moment Business Decision - Kurtosis
print("Kurtosis of salary : " , sal.salary.kurt())
print("Kurtosis of exp : " , sal.exp.kurt())

# Graphical Representation - Distribution, Box Plot, Scatter Plot
sns.distplot(sal.salary)
sns.distplot(sal.exp)

plt.boxplot(sal.salary)
plt.boxplot(sal.exp)

plt.scatter(sal.exp,sal.salary)

#Modeling - Linear Regression

import statsmodels.formula.api as smf

model1 = smf.ols('salary ~ exp' , data = sal).fit()
model1.summary()  # output Evaluation

# Model Evaluation & Assumptions

np.corrcoef(sal.salary,sal.exp) # Correlation Coefficient

error = model1.predict(sal.exp) - sal.salary
rmse = np.sqrt(np.mean(error * error)) # Root Mean Squared Value
rmse


stats.shapiro(error) # Shapiro test for normality
stats.probplot(error , dist = "norm" , plot = pylab) # QQplot

# Coeffs p-value is less than 0.05 - Coeffs values are stastically significant
# F Statistic p-value < 0.05 - Regression Eqn is statistically significant
# RMSE is very low
# Shapiro test on Residuals - p-val > 0.05, Residuals are normal
# R Squared value is only 0.957
# Correlation Coefficient is 0.978 indicates strong positive relationship


#Modeling - Linear Regression - Log


np.corrcoef(sal.salary,np.log(sal.exp)) # Correlation Coefficient
plt.scatter(sal.salary,np.log(sal.exp))


model2 = smf.ols('salary ~ np.log(exp)' , data = sal).fit()
model2.summary()  # output Evaluation

# Model Evaluation & Assumptions
error2 = model2.predict(sal.exp) - sal.salary
rmse2 = np.sqrt(np.mean(error2 * error2)) # Root Mean Squared Value
rmse2


stats.shapiro(error2) # Shapiro test for normality
stats.probplot(error2 , dist = "norm" , plot = pylab) # QQplot

# Coeffs p-value is less than 0.05 - Coeffs values are stastically significant
# F Statistic p-value < 0.05 - Regression Eqn is statistically significant
# RMSE is very low
# Shapiro test on Residuals - p-val > 0.05, Residuals are normal
# R Squared value is 0.8539
# Correlation Coefficient is - 0.924 indicates strong positive relationship



#Modeling - Linear Regression - Exponential


np.corrcoef(np.log(sal.salary),sal.exp) # Correlation Coefficient
plt.scatter(np.log(sal.salary), sal.exp)

model3 = smf.ols('np.log(salary) ~ exp' , data = sal).fit()
model3.summary()  # output Evaluation

# Model Evaluation & Assumptions
error3 = np.exp(model3.predict(sal.exp)) - sal.salary
rmse3 = np.sqrt(np.mean(error3 * error3)) # Root Mean Squared Value
rmse3


stats.shapiro(error3) # Shapiro test for normality
stats.probplot(error3 , dist = "norm" , plot = pylab) # QQplot


# Coeffs p-value is less than 0.05 - Coeffs values are stastically significant
# F Statistic p-value < 0.05 - Regression Eqn is statistically significant
# RMSE is very low
# Shapiro test on Residuals - p-val > 0.05, Residuals are normal
# R Squared value is 0.93
# Correlation Coefficient is - 0.96 indicates strong positive relationship



#Modeling - Polynomial Regression


np.corrcoef(sal.salary,sal.exp) # Correlation Coefficient
plt.scatter(sal.salary, sal.exp)

model4 = smf.ols('salary ~ exp + I(exp * exp)' , data = sal).fit()
model4.summary()  # output Evaluation

# Model Evaluation & Assumptions
error4 = model4.predict(sal.exp) - sal.salary
rmse4 = np.sqrt(np.mean(error4 * error4)) # Root Mean Squared Value
rmse4


stats.shapiro(error4) # Shapiro test for normality
stats.probplot(error4 , dist = "norm" , plot = pylab) # QQplot





# Coeffs p-value is less than 0.05 - Coeffs values are stastically significant
# F Statistic p-value < 0.05 - Regression Eqn is statistically significant
# RMSE is very low
# Shapiro test on Residuals - p-val > 0.05, Residuals are normal
# R Squared value is 0.957

final_model = smf.ols('salary ~ exp + I(exp * exp)' , data = sal).fit()
final_model.summary()  # output Evaluation


final_model.conf_int(0.05) # CI of Coeffs at 95% limit
final_model.predict(pd.DataFrame(sal.exp)) # Predicting salary based on confidence intervals
