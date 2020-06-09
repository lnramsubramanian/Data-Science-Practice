import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

mdata = pd.read_csv("C:/Users/lnram/Desktop/ExcelR/Module 10 - Multinomial Regression/Assignments/mdata.csv")
mdata.head()

# Business Objective - Predicting Whether the client has subscribed a term deposit or not 

mdata.columns
mdata = mdata[['female', 'ses', 'schtyp', 'prog', 'read', 'write',
       'math', 'science', 'honors']] # Removing ID column

# Check for missing values
mdata.isna().sum() # first row has NA values

# Basic Statistics
mdata.describe()

mdata.dtypes # All are numerical variables

#Skeweness
mdata.skew() # Few features are highly skewed

#Kurtosis
mdata.kurt() # Most of the skewed features are having extreme curtosis

#Graphical Representation
mdata.hist(grid=False)

mdata.columns
mdata['prog'].value_counts(normalize=True) #Balanced Data Set

sns.pairplot(data = mdata , hue = 'prog') 

# Correlation Matrix
sns.heatmap(mdata.corr(),annot=True)  # Looks to be partial correlation

#Converting Categorical Variables to numberical variables
mdata.dtypes
cols = ['female' , 'ses', 'schtyp', 'honors']
mdata = pd.get_dummies(data = mdata , columns = cols)

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
mdata['prog'] = label.fit_transform(mdata['prog'])


#Multinomial Model Building
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
mdata.columns
X = mdata[['read', 'write', 'math', 'science', 'female_female',
       'female_male', 'ses_high', 'ses_low', 'ses_middle', 'schtyp_private',
       'schtyp_public', 'honors_enrolled', 'honors_not enrolled']]
y = mdata['prog']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
model = LogisticRegression(multi_class="multinomial",solver="newton-cg").fit(X_train,y_train)

#Predicting output not the probabilities
train_predict = model.predict(X_train) # Train predictions 
test_predict = model.predict(X_test) # Test predictions

# Train accuracy 
accuracy_score(y_train,train_predict) # 65.71%
# Test accuracy 
accuracy_score(y_test,test_predict) # 55%

#Confusion Matrix
from sklearn.metrics import classification_report
print(classification_report(y_test,test_predict))

