import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

comp = pd.read_csv('C:/Users/lnram/Desktop/ExcelR/Module 7 - Multiple Linear Regresion/Assignments/Computer_Data.csv')
comp.head()
comp = comp.iloc[:,1:] # Removing index column
# Business Problem - Predicting Price of Computer


# Checking for missing values
comp.isna().sum()  # No Missing values

# Summary Statistics
comp.describe()

# Different data Types
comp.dtypes # three categorical variables - cd, multi, premium


# Continous variables

# Skeweness
comp.skew()

#Kurtosis
comp.kurt()

# Graphical Analysis

comp.select_dtypes(exclude = "object").columns  # to get numerical columns

# Distribution Plots
sns.distplot(comp.price , hist = False , label = "price");
sns.distplot(comp.speed , hist = False , label = "speed"); # Multi modal
sns.distplot(comp.hd , hist = False , label = "hd"); # Multi modal and right skewed
sns.distplot(comp.ram , hist = False , label = "ram");  # Multi modal and right skewed
sns.distplot(comp.screen , hist = False , label = "screen"); # multi modal and right skewed
sns.distplot(comp.ads , hist = False , label = "ads"); # left skewed and bi modal
sns.distplot(comp.trend , hist = False , label = "trend"); # peakedness

comp.boxplot() # Outliers exists for Price, hd, ram, screen

# Discrete Analysis
comp.select_dtypes("object").columns

comp['cd'].value_counts(normalize=True) # Balanced data
comp['multi'].value_counts(normalize=True) # Imbalanced data
comp['premium'].value_counts(normalize=True) #Imbalanced data

# Relationship with Price
sns.pairplot(comp)
sns.heatmap(comp.corr() , annot = True) # Price and ram are moderately correlated, ram and hd are moderately correlated

# Model Building - Multiple Linear Regression
# Steps - Build model, Score, Predicting, RMSE, Model Assumptions

from sklearn.linear_model import LinearRegression
LR1 = LinearRegression()

# Converting categorical variables to dummies and dropping the first column to N-1 dummies
new_comp = pd.get_dummies(comp , columns=comp.select_dtypes("object").columns , drop_first = True)

x = new_comp.iloc[:,1:]
y = new_comp.iloc[:,0]


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

# We got minimum R_Squared value at small alpha values 

# So Simple Regression Model yields better results than Lasso and Ridge

import statsmodels.formula.api as smf
formula = "price" + "~" + "+".join(x.columns)
model = smf.ols(formula ,data=new_comp).fit() # regression model

# Summary
model.summary()


# influence index plots
import statsmodels.api as sm
fig, ax = plt.subplots(figsize=(12,8))
fig = sm.graphics.influence_plot(model, alpha = 0.05 , ax = ax , criterion = "cooks") # Points 1440 , 1700 seems to be outliers

new_comp = new_comp.drop(new_comp.index[[1440,1700]] , axis=0)
model2 = smf.ols(formula ,data=new_comp).fit() # regression model

# Summary
model2.summary()
# Influce index plots
fig, ax = plt.subplots(figsize=(12,8))
fig = sm.graphics.influence_plot(model2, alpha = 0.05 , ax = ax , criterion = "cooks") # Points 19, 24 ,27 seems to be outliers

#RMSE
pred2 = model2.predict(new_comp)
res = pred2 - new_comp.price
rmse = np.sqrt(np.mean(res * res))
rmse

new_comp = new_comp.drop(new_comp.index[[19 , 24, 27]] , axis=0)
model3 = smf.ols(formula ,data=new_comp).fit() # regression model
model3.summary()

#RMSE
pred3 = model3.predict(new_comp)
res = pred3 - new_comp.price
rmse = np.sqrt(np.mean(res * res))
rmse

# Validating Model Assumptions
# Residuals Vs Fitted Values
plt.scatter(x = pred3 , y = new_comp.price - pred3);plt.xlabel("Fitted");plt.ylabel("Residuals")
# Checking normal distribution 
plt.hist(pred3 - new_comp.price) # Normally distributed but skewed

# Predicted Vs Actual
plt.scatter(x = pred3 , y = new_comp.price);plt.xlabel("Predicted");plt.ylabel("Actual")


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
train,test = train_test_split(new_comp,test_size=0.2)


# preparing the model on train data 

model_train = smf.ols(formula,data=train).fit()

# train_data prediction
train_pred = model_train.predict(train)

# train residual values 
train_resid  = train_pred - train.price

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid))

# prediction on test data set 
test_pred = model_train.predict(test)

# test residual values 
test_resid  = test_pred - test.price

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))

# Both test and train RMSE are close by

