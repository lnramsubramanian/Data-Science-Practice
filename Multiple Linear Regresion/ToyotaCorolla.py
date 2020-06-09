import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

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

# Multiple Regression Models
import statsmodels.formula.api as smf # for regression model

ml1 = smf.ols('Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight',data=corolla).fit() # regression model
ml1.summary()


# Output Summary:
# 1. p-values of most of the coeffs are less than 0.05 but cc and Doors are insignificant
# R squared adj and R squared are closeby
# R squared value is 0.864 - Determination is strong 
# AIC value is 2477
# Durbin-Watson is in the normal range

corolla.columns

# Variance Inflation Factors (VIF) for Continuous variables
rsq_age = smf.ols('Age_08_04 ~ KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight' , data = corolla).fit().rsquared
vif_age = 1 / (1-rsq_age)

rsq_km = smf.ols('KM ~ Age_08_04 + HP + cc + Doors + Gears + Quarterly_Tax + Weight' , data = corolla).fit().rsquared
vif_km = 1 / (1-rsq_km)

rsq_HP = smf.ols('HP ~ Age_08_04 + KM + cc + Doors + Gears + Quarterly_Tax + Weight' , data = corolla).fit().rsquared
vif_HP = 1 / (1-rsq_HP)

rsq_cc = smf.ols('cc ~ Age_08_04 + HP + KM + Doors + Gears + Quarterly_Tax + Weight' , data = corolla).fit().rsquared
vif_cc = 1 / (1-rsq_cc)

rsq_Doors = smf.ols('Doors ~ Age_08_04 + HP + cc + KM + Gears + Quarterly_Tax + Weight' , data = corolla).fit().rsquared
vif_Doors = 1 / (1-rsq_Doors)

rsq_Gears = smf.ols('Gears ~ Age_08_04 + HP + cc + Doors + KM + Quarterly_Tax + Weight' , data = corolla).fit().rsquared
vif_Gears = 1 / (1-rsq_Gears)

rsq_tax = smf.ols('Quarterly_Tax ~ Age_08_04 + HP + cc + Doors + Gears + KM + Weight' , data = corolla).fit().rsquared
vif_tax = 1 / (1-rsq_tax)

rsq_Weight = smf.ols('Weight ~ Age_08_04 + HP + cc + Doors + Gears + Quarterly_Tax + KM' , data = corolla).fit().rsquared
vif_Weight = 1 / (1-rsq_Weight)

corolla.columns

VIF = pd.DataFrame({'variables' : ['Age_08_04', 'KM', 'HP', 'cc', 'Doors', 'Gears', 'Quarterly_Tax', 'Weight'] , 
                    'vif' : [vif_age, vif_km, vif_HP, vif_cc, vif_Doors, vif_Gears,
       vif_tax, vif_Weight]}) 

VIF # No VIF values are above 10; There is not multi collinearity

# Transforming cc and Doors
ml2 = smf.ols('Price ~ Age_08_04 + KM + HP + np.log(cc) + np.log(Doors) + Gears + Quarterly_Tax + Weight',data=corolla).fit() # regression model
ml2.summary()

# Output Summary:
# 1. p-values of most of the coeffs are less than 0.05 but Doors is insignificant; Transformation worked for cc
# R squared adj and R squared are closeby
# R squared value is 0.867 - Determination is strong - Better than model 1
# AIC value is 2473 lower than model 1
# Durbin-Watson is in the normal range

sns.distplot(corolla.Doors) # Looks to be a Categorical Feature


# Removing Door from the model
ml3 = smf.ols('Price ~ Age_08_04 + KM + HP + np.log(cc) + Gears + Quarterly_Tax + Weight',data=corolla).fit() # regression model
ml3.summary()

# Output Summary:
# 1. p-values of the coeffs are less than 0.05 
# R squared adj and R squared are closeby
# R squared value is 0.867 - Determination is strong - Better than model 1 and same as model 2
# AIC value is 2473 lower than model 1
# Durbin-Watson is in the normal range

# splitting the data into train and test

from sklearn.model_selection import train_test_split
corolla_train,corolla_test  = train_test_split(corolla,test_size = 0.3) # 30% test data

ml_final = smf.ols('Price ~ Age_08_04 + KM + HP + np.log(cc) + Gears + Quarterly_Tax + Weight',data=corolla_train).fit() # regression model
ml_final.summary()


train_pred = ml_final.predict(corolla_train)

train_res = train_pred - corolla_train.Price

train_rmse = np.sqrt(np.mean(train_res * train_res))

test_pred = ml_final.predict(corolla_test)
test_res = test_pred - corolla_test.Price
test_rmse = np.sqrt(np.mean(test_res * test_res))

train_rmse
test_rmse

# Train RMSE and Test RMSE are close by - Good Fitting