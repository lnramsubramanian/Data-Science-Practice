import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

start.columns = ['R_D', 'Administration', 'Marketing', 'State', 'Profit']

# Check for missing values
start.isna().sum() # No missing values

# Basic Statistics
start.describe()

start.dtypes # State - Categorical variable

#Skeweness
start.skew() # Admin value is skewed negatively

#Kurtosis
start.kurt() 

#Graphical Representation
start.columns
sns.distplot(start['R_D'])
sns.distplot(start['Administration'])
sns.distplot(start['Marketing'])
sns.distplot(start['Profit'])


start.boxplot(grid = False)

# Profit has outliers

# Correlation Matrix
sns.heatmap(start.corr(),annot=True)  # Administration is poorly correlated and other two features are having strong correlation

# Discrete Variable - EDA

start['State'].value_counts(normalize=True)*100 # Almost Equally distributed

# Multiple Regression Models
import statsmodels.formula.api as smf # for regression model

ml1 = smf.ols('Profit ~ R_D + Administration + Marketing + State' , data=start).fit() # regression model
ml1.summary()


# Output Summary:
# 1. p-values of most of the coeffs are less than 0.05 but State Florida , State New York, Administration and Marketing are insignificant
# R squared adj and R squared are closeby
# R squared value is 0.951 - Determination is strong 
# AIC value is 1063
# Durbin-Watson is in the normal range

#Variance Inflation Factors
rsq_R_D = smf.ols('R_D ~ Administration + Marketing' , data = start).fit().rsquared
vif_R_D = 1 / (1-rsq_R_D)

rsq_admin = smf.ols('Administration ~ R_D + Marketing' , data = start).fit().rsquared
vif_admin = 1 / (1-rsq_admin)

rsq_marketing = smf.ols('Marketing ~ Administration + R_D' , data = start).fit().rsquared
vif_marketing = 1 / (1-rsq_marketing)

# All the VIF values are below 10 - No multi-collinearity



# Removing Administration due to poor correlation
ml2 = smf.ols('Profit ~ R_D + Marketing + State' , data=start).fit() # regression model
ml2.summary()


# Output Summary:
# Not much change in the output when compared to Model 1


from sklearn.model_selection import train_test_split
start_train,start_test  = train_test_split(start,test_size = 0.3) # 30% test data

ml_final = smf.ols('Profit ~ R_D + Administration + Marketing + State' , data=start_train).fit() # regression model
ml_final.summary()


train_pred = ml_final.predict(start_train)

train_res = train_pred - start_train.Profit

train_rmse = np.sqrt(np.mean(train_res * train_res))

test_pred = ml_final.predict(start_test)
test_res = test_pred - start_test.Profit
test_rmse = np.sqrt(np.mean(test_res * test_res))

train_rmse
test_rmse

# Train RMSE and Test RMSE are close by - Good Fitting