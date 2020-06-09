import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

comp.head()
comp = comp.iloc[:,1:] # Removing index column
# Business Problem - Predicting Price of Computer

# Data Preparation & EDA

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

# Model Building
import statsmodels.formula.api as smf
ml1 = smf.ols('price ~ speed + hd + ram + screen + cd + multi + premium + ads + trend' , data = comp).fit()
ml1.summary()

# Output Summary:
# 1. p-values of coeff are less than 0.05
# R squared adj and R squared are closeby
# R squared value is 0.776 - Determination is moderate
# AIC value is 88100
# Durbin-Watson is in the normal range

# Variance Inflation Factors (VIF) for Continuous variables
rsq_speed = smf.ols('speed ~ hd + ram + screen + cd + multi + premium + ads + trend' , data = comp).fit().rsquared
vif_speed = 1 / (1-rsq_speed)

rsq_hd = smf.ols('hd ~ speed + ram + screen + cd + multi + premium + ads + trend' , data = comp).fit().rsquared
vif_hd = 1 / (1-rsq_hd)

rsq_ram = smf.ols('ram ~ hd + speed + screen + cd + multi + premium + ads + trend' , data = comp).fit().rsquared
vif_ram = 1 / (1-rsq_ram)

rsq_screen = smf.ols('screen ~ hd + ram + speed + cd + multi + premium + ads + trend' , data = comp).fit().rsquared
vif_screen = 1 / (1-rsq_screen)

rsq_ads = smf.ols('ads ~ hd + ram + screen + cd + multi + premium + speed + trend' , data = comp).fit().rsquared
vif_ads = 1 / (1-rsq_ads)

rsq_trend = smf.ols('trend ~ hd + ram + screen + cd + multi + premium + ads + speed' , data = comp).fit().rsquared
vif_trend = 1 / (1-rsq_trend)

VIF = pd.DataFrame({'variables' : ['speed', 'hd', 'ram', 'screen', 'ads', 'trend'] , 
                    'vif' : [vif_speed , vif_hd , vif_ram, vif_screen, vif_ads, vif_trend]}) 
VIF # No VIF values are above 10; There is not multi collinearity

# Since the input variables distributionsa are not normal, applying log transformation

ml2 = smf.ols('price ~ np.log(speed) + np.log(hd) + np.log(ram) + np.log(screen) + cd + multi + premium + np.log(ads) + np.log(trend)' , data = comp).fit()
ml2.summary()

# Output Summary:
# 1. p-values of coeff are less than 0.05
# R squared adj and R squared are closeby
# R squared value is 0.743 - Determination is moderate ; lower than model 1
# AIC value is 8896 higher than model 1
# Durbin-Watson is in the normal range


# Applying log transformation, changes the distribution of hd and ram
ml3 = smf.ols('price ~ speed + np.log(hd) + np.log(ram) + screen + cd + multi + premium + speed + trend', data = comp).fit()
ml3.summary()

# Output Summary:
# 1. p-values of coeff are less than 0.05
# R squared adj and R squared are closeby
# R squared value is 0.761 - Determination is moderate lower than model 1
# AIC value is 88500 higher than model 1
# Durbin-Watson is in the normal range
df = comp.drop(['cd' , 'multi' , 'premium'] , axis = 1)
from sklearn.preprocessing import StandardScaler
df.values.reshape(1,-1)
scaler = StandardScaler()
scaled_feature = scaler.fit_transform(df)

scaled_feature = pd.DataFrame(scaled_feature)
scaled_feature.columns = df.columns

scaled_feature['cd'] = comp.cd
scaled_feature['multi'] = comp.multi
scaled_feature['premium'] = comp.premium

# Model building
ml4 = smf.ols('price ~ speed + hd + ram + screen + cd + multi + premium + speed + trend', data = scaled_feature).fit()
ml4.summary()

# Output Summary:
# 1. p-values of coeff are less than 0.05
# R squared adj and R squared are closeby
# R squared value is 0.770 - Determination is moderate lower than model 1
# AIC value is 8590 lower than model 1
# Durbin-Watson is in the normal range

plt.boxplot(df['price']) # outlier values above 3750

new_comp = comp[comp['price'] <= 3750]
new_comp.shape # 90 outliers removed
new_comp.boxplot()

ml5 = smf.ols('price ~ speed + hd + ram + screen + cd + multi + premium + ads + trend' , data = new_comp).fit()
ml5.summary()

# Output Summary:
# 1. p-values of coeff are less than 0.05
# R squared adj and R squared are closeby
# R squared value is 0.783 - Determination is moderate higher than model 1
# AIC value is 8567, lower than model 1
# Durbin-Watson is in the normal range


# Model Conclusion - Fifth model seems to be a better model with R sqaured - 0.783
from sklearn.model_selection import train_test_split
comp_train,comp_test  = train_test_split(new_comp,test_size = 0.3) # 30% test data

ml_final = smf.ols('price ~ speed + hd + ram + screen + cd + multi + premium + ads + trend' , data = comp_train).fit()
ml_final.summary()

train_pred = ml_final.predict(comp_train)

train_res = train_pred - comp_train.price

train_rmse = np.sqrt(np.mean(train_res * train_res))

test_pred = ml_final.predict(comp_test)
test_res = test_pred - comp_test.price
test_rmse = np.sqrt(np.mean(test_res * test_res))

# Train RMSE and Test RMSE are close by - Good Fitting