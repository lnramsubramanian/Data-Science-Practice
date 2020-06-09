import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

corolla = pd.read_csv('C:/Users/lnram/Desktop/ExcelR/Module 7 - Multiple Linear Regresion/Assignments/ToyotaCorolla.csv' , encoding= 'unicode_escape')

corolla = corolla[["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]

# Check for missing values
corolla.isna().sum()

# Basic Statistics
corolla.describe()

corolla.dtypes

#Skeweness
corolla.skew() # CC value is skewed positively

#Kurtosis
corolla.kurt() # CC value is leptokurtic

#Graphical Representation

len(corolla.columns)

i = 1
for cols in corolla.columns:
    plt.subplot(3,3,i)
    sns.distplot(corolla[cols] , hist = False , label = cols)    
    i = i+1

# Positively Skewed - Price, KM, HP, CC, Gears, Weight
# Negatively Skewed - Age
# ultiple Disributions - Doors, Quarterly_Tax


corolla.boxplot(grid = False)

# Outliers exist is all features except for Doors

# Model Building - Multiple Linear Regression
# Steps - Build model, Score, Predicting, RMSE, Model Assumptions

from sklearn.linear_model import LinearRegression
LR1 = LinearRegression()

x = corolla.iloc[:,1:]
y = corolla.iloc[:,0]


LR1.fit(x , y)
# Getting coefficients of variables               
LR1.coef_
LR1.intercept_

# Getting Adjusted R squared value
LR1.score(x,y)

pred1 = LR1.predict(x)

# Rmse value
np.sqrt(np.mean((pred1-y)**2))

# Validating Model Assumptions
# Residuals Vs Fitted Values
plt.scatter(x = pred1 , y = y - pred1);plt.xlabel("Fitted");plt.ylabel("Residuals")
# Checking normal distribution 
plt.hist(pred1-y) # Normally distributed but skewed

# Predicted Vs Actual
plt.scatter(x = pred1 , y = y);plt.xlabel("Predicted");plt.ylabel("Actual")
plt.bar(height = pd.Series(LR1.coef_),x = x.columns) #clearly the coeff magnitudes vary by large extent

# Running Ridge Regression
from sklearn.linear_model import Ridge
RM1 = Ridge(alpha = 0.4,normalize=True)
RM1.fit(x,y)
# Coefficient values for all the independent variables
RM1.coef_
RM1.intercept_
plt.bar(height = pd.Series(RM1.coef_),x=pd.Series(x.columns))

# Getting Adjusted R squared value
RM1.score(x,y)

pred1 = RM1.predict(x)

# Rmse value
np.sqrt(np.mean((pred1-y)**2))

# Validating Model Assumptions
# Residuals Vs Fitted Values
plt.scatter(x = pred1 , y = y - pred1);plt.xlabel("Fitted");plt.ylabel("Residuals")
# Checking normal distribution 
plt.hist(pred1-y) # Normally distributed but skewed

# Predicted Vs Actual
plt.scatter(x = pred1 , y = y);plt.xlabel("Predicted");plt.ylabel("Actual")
plt.bar(height = pd.Series(LR1.coef_),x = x.columns) #clearly the coeff magnitudes vary by large extent

# Running a Ridge Regressor of set of alpha values and observing how the R-Squared
rmse = []
R_sqrd = []
alphas = np.arange(0,100,0.05)
for i in alphas:
    RM = Ridge(alpha = i,normalize=True)
    RM.fit(x , y)
    R_sqrd.append(RM.score(x , y))
    rmse.append(np.sqrt(np.mean((RM.predict(x) - y)**2)))
    
# Plotting RMSE,R_Squared values with respect to alpha values

# Alpha vs R_Squared values
plt.scatter(x=alphas,y=R_sqrd);plt.xlabel("alpha");plt.ylabel("R_Squared")

# Alpha vs RMSE
plt.scatter(x=alphas,y=rmse);plt.xlabel("alpha");plt.ylabel("rmse")

# We got minimum R_Squared value at small alpha values 

# Running Lasso Regression
from sklearn.linear_model import Lasso
LM1 = Lasso(alpha = 0.01,normalize=True)
LM1.fit(x , y)

# Coefficient values for all the independent variables
LM1.coef_
LM1.intercept_

# Getting Adjusted R squared value
LM1.score(x,y)

pred1 = LM1.predict(x)

# Rmse value
np.sqrt(np.mean((pred1-y)**2))

# Validating Model Assumptions
# Residuals Vs Fitted Values
plt.scatter(x = pred1 , y = y - pred1);plt.xlabel("Fitted");plt.ylabel("Residuals")
# Checking normal distribution 
plt.hist(pred1-y) # Normally distributed but skewed

# Predicted Vs Actual
plt.scatter(x = pred1 , y = y);plt.xlabel("Predicted");plt.ylabel("Actual")
plt.bar(height = pd.Series(LM1.coef_),x = x.columns) #clearly the coeff magnitudes vary by large extent

# Running a Ridge Regressor of set of alpha values and observing how the R-Squared
rmse = []
R_sqrd = []
alphas = np.arange(0,30,0.05)
for i in alphas:
    LM = Lasso(alpha = i,normalize=True)
    LM.fit(x , y)
    R_sqrd.append(LM.score(x , y))
    rmse.append(np.sqrt(np.mean((LM.predict(x) - y)**2)))
    
    
# Plotting RMSE,R_Squared values with respect to alpha values

# Alpha vs R_Squared values
plt.scatter(x=alphas,y=R_sqrd);plt.xlabel("alpha");plt.ylabel("R_Squared")

# Alpha vs RMSE
plt.scatter(x=alphas,y=rmse);plt.xlabel("alpha");plt.ylabel("rmse")

# R Squared value is constant for different alpha values
# The RMSE Value is lower for small alpha values - So we would use simple multiple regression


import statsmodels.formula.api as smf
formula = "Price" + "~" + "+".join(x.columns)
model = smf.ols(formula ,data=corolla).fit() # regression model

# Summary
model.summary()


# influence index plots
import statsmodels.api as sm
fig, ax = plt.subplots(figsize=(12,8))
fig = sm.graphics.influence_plot(model, alpha = 0.05 , ax = ax , criterion = "cooks") # Influential Points - 80

new_corolla = corolla.drop(corolla.index[[80]] , axis=0)
model2 = smf.ols(formula ,data=new_corolla).fit() # regression model

# Summary
model2.summary()
# Influce index plots
fig, ax = plt.subplots(figsize=(12,8))
fig = sm.graphics.influence_plot(model2, alpha = 0.05 , ax = ax , criterion = "cooks") # Points 960,221 seems to be outliers

#RMSE
pred2 = model2.predict(new_corolla)
res = pred2 - new_corolla.Price
rmse = np.sqrt(np.mean(res * res))
rmse

new_corolla = corolla.drop(corolla.index[[80 , 221 , 960]] , axis=0)
model3 = smf.ols(formula ,data=new_corolla).fit() # regression model

# Summary
model3.summary()
# Influce index plots
fig, ax = plt.subplots(figsize=(12,8))
fig = sm.graphics.influence_plot(model3, alpha = 0.05 , ax = ax , criterion = "cooks") # Points 960,221 seems to be outliers

#RMSE
pred3 = model3.predict(new_corolla)
res = pred3 - new_corolla.Price
rmse = np.sqrt(np.mean(res * res))
rmse

# Validating Model Assumptions
# Residuals Vs Fitted Values
plt.scatter(x = pred3 , y = new_corolla.Price - pred3);plt.xlabel("Fitted");plt.ylabel("Residuals")
# Checking normal distribution 
plt.hist(pred3 - new_corolla.Price) # Normally distributed but skewed

# Predicted Vs Actual
plt.scatter(x = pred3 , y = new_corolla.Price);plt.xlabel("Predicted");plt.ylabel("Actual")


# QQ plot for residuals 
import pylab          
import scipy.stats as st

# Checking Residuals are normally distributed
st.probplot(model3.resid_pearson, dist="norm", plot=pylab)


############ Homoscedasticity #######

# Residuals VS Fitted Values 
plt.scatter(pred3,model3.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")

# Spliting the data into train and test 
from sklearn.model_selection import train_test_split
train,test = train_test_split(new_corolla,test_size=0.2)


# preparing the model on train data 

model_train = smf.ols(formula,data=train).fit()

# train_data prediction
train_pred = model_train.predict(train)

# train residual values 
train_resid  = train_pred - train.Price

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid))

# prediction on test data set 
test_pred = model_train.predict(test)

# test residual values 
test_resid  = test_pred - test.Price

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))

# Both test and train RMSE are close by
