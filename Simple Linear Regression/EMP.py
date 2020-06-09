import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import pylab


# Opening file and assigning to an object
emp = pd.read_csv('C:/Users/lnram/Desktop/ExcelR/Module 6 - Simple Linear Regression/Assignments/emp_data.csv')

# Business Problem: Predict churn with Max accuracy

#EDA/Descriptive Analytics:
# First Moment Business Decision - Measure of Central Tendancy

emp.describe()

# Second Moment Business Decision - Measure of Dispersion

emp.columns = "hike" , "churn"

#Std Deviation
print("Std Deviation of churn: " , emp.churn.std())
print("Std Deviation of hike: " , emp.hike.std())


#Range
print("Range of churn: " , max(emp.churn) - min(emp.churn))
print("Range of hike: " , max(emp.hike) - min(emp.hike))


# Third Moment Business Decision - Skewness
print("Skewness of churn : " , emp.churn.skew())
print("Skewness of hike : " , emp.hike.skew())


# Fourth Moment Business Decision - Kurtosis
print("Kurtosis of churn : " , emp.churn.kurt())
print("Kurtosis of hike : " , emp.hike.kurt())

# Graphical Representation - Distribution, Box Plot, Scatter Plot
sns.distplot(emp.churn)
sns.distplot(emp.hike)

plt.boxplot(emp.churn)
plt.boxplot(emp.hike)

plt.scatter(emp.hike,emp.churn)

#Modeling - Linear Regression

import statsmodels.formula.api as smf

model1 = smf.ols('churn ~ hike' , data = emp).fit()
model1.summary()  # output Evaluation

# Model Evaluation & Assumptions

np.corrcoef(emp.churn,emp.hike) # Correlation Coefficient

error = model1.predict(emp.hike) - emp.churn
rmse = np.sqrt(np.mean(error * error)) # Root Mean Squared Value
rmse


stats.shapiro(error) # Shapiro test for normality
stats.probplot(error , dist = "norm" , plot = pylab) # QQplot


# Coeffs p-value is less than 0.05 - Coeffs values are stastically significant
# F Statistic p-value < 0.05 - Regression Eqn is statistically significant
# RMSE is very low
# Shapiro test on Residuals - p-val > 0.05, Residuals are normal
# R Squared value is only 0.83
# Correlation Coefficient is - 0.91 indicates strong inversely relationship


#Modeling - Linear Regression - Log

np.corrcoef(emp.churn,np.log(emp.hike)) # Correlation Coefficient
plt.scatter(emp.churn,np.log(emp.hike))


model2 = smf.ols('churn ~ np.log(hike)' , data = emp).fit()
model2.summary()  # output Evaluation

# Model Evaluation & Assumptions
error2 = model2.predict(emp.hike) - emp.churn
rmse2 = np.sqrt(np.mean(error2 * error2)) # Root Mean Squared Value
rmse2


stats.shapiro(error2) # Shapiro test for normality
stats.probplot(error2 , dist = "norm" , plot = pylab) # QQplot


# Coeffs p-value is less than 0.05 - Coeffs values are stastically significant
# F Statistic p-value < 0.05 - Regression Eqn is statistically significant
# RMSE is very low
# Shapiro test on Residuals - p-val > 0.05, Residuals are normal
# R Squared value is 0.848
# Correlation Coefficient is - 0.92 indicates strong inversely relationship


#Modeling - Linear Regression - Exponential

np.corrcoef(np.log(emp.churn),emp.hike) # Correlation Coefficient
plt.scatter(np.log(emp.churn), emp.hike)

model3 = smf.ols('np.log(churn) ~ hike' , data = emp).fit()
model3.summary()  # output Evaluation

# Model Evaluation & Assumptions
error3 = np.exp(model3.predict(emp.hike)) - emp.churn
rmse3 = np.sqrt(np.mean(error3 * error3)) # Root Mean Squared Value
rmse3


stats.shapiro(error3) # Shapiro test for normality
stats.probplot(error3 , dist = "norm" , plot = pylab) # QQplot



# Coeffs p-value is less than 0.05 - Coeffs values are stastically significant
# F Statistic p-value < 0.05 - Regression Eqn is statistically significant
# RMSE is very low
# Shapiro test on Residuals - p-val > 0.05, Residuals are normal
# R Squared value is 0.873
# Correlation Coefficient is - 0.934 indicates strong inversely relationship



#Modeling - Polynomial Regression


np.corrcoef(emp.churn,emp.hike) # Correlation Coefficient
plt.scatter(emp.churn, emp.hike)

model4 = smf.ols('churn ~ hike + I(hike * hike)' , data = emp).fit()
model4.summary()  # output Evaluation

# Model Evaluation & Assumptions
error4 = model4.predict(emp.hike) - emp.churn
rmse4 = np.sqrt(np.mean(error4 * error4)) # Root Mean Squared Value
rmse4


stats.shapiro(error4) # Shapiro test for normality
stats.probplot(error4 , dist = "norm" , plot = pylab) # QQplot


# Coeffs p-value is less than 0.05 - Coeffs values are stastically significant
# F Statistic p-value < 0.05 - Regression Eqn is statistically significant
# RMSE is very low
# Shapiro test on Residuals - p-val > 0.05, Residuals are normal
# R Squared value is 9737


final_model = smf.ols('churn ~ hike + I(hike * hike)' , data = emp).fit()
final_model.summary()  # output Evaluation


final_model.conf_int(0.05) # CI of Coeffs at 95% limit
final_model.predict(pd.DataFrame(emp.hike)) # Predicting churn based on confidence intervals
