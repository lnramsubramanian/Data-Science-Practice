import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

affair = pd.read_csv("C:/Users/lnram/Desktop/ExcelR/Module 9 - Logistic Regression/Assignments/Affairs.csv")
affair.head()

# Business Objective - Predicting an affair

affair.columns
affair = affair.drop('Unnamed: 0',1) # Removing Index column

# The years of marriage is split into 6 columns which can be converted into one column

affair['yrsmarr1'] = affair['yrsmarr1'].apply(lambda x : 1 if x == 1 else 0)
affair['yrsmarr2'] = affair['yrsmarr2'].apply(lambda x : 2 if x == 1 else 0)
affair['yrsmarr3'] = affair['yrsmarr3'].apply(lambda x : 3 if x == 1 else 0)
affair['yrsmarr4'] = affair['yrsmarr4'].apply(lambda x : 4 if x == 1 else 0)
affair['yrsmarr5'] = affair['yrsmarr5'].apply(lambda x : 5 if x == 1 else 0)
affair['yrsmarr6'] = affair['yrsmarr6'].apply(lambda x : 6 if x == 1 else 0)

affair['yrsmarr'] = affair['yrsmarr1'] + affair['yrsmarr2'] + affair['yrsmarr3'] + affair['yrsmarr4'] + affair['yrsmarr5'] + affair['yrsmarr6']
affair['yrsmarr'].value_counts()

affair = affair.drop(['yrsmarr1','yrsmarr2','yrsmarr3','yrsmarr4','yrsmarr5','yrsmarr6'],1) # Removing individual years

# Check for missing values
affair.isna().sum() # No missing values

# Basic Statistics
affair.describe()

affair.dtypes # All are count variables

#Skeweness
affair.skew() # Admin value is skewed negatively

#Kurtosis
affair.kurt() 

#Graphical Representation
affair.columns

sns.countplot(affair['naffairs']) # There are affairs more than 2, removing them for logistics regression
affair['naffairs'] = affair['naffairs'].apply(lambda x : 1 if x >= 1 else 0)

sns.countplot(affair['kids'])
sns.countplot(affair['vryunhap'])
sns.countplot(affair['unhap'])
sns.countplot(affair['avgmarr'])
sns.countplot(affair['hapavg'])
sns.countplot(affair['vryhap'])
sns.countplot(affair['antirel'])
sns.countplot(affair['notrel'])
sns.countplot(affair['slghtrel'])
sns.countplot(affair['smerel'])
sns.countplot(affair['vryrel'])
sns.countplot(affair['yrsmarr'])



# Correlation Matrix
plt.figure(figsize=(12,8))
sns.heatmap(affair.corr(),annot=True)  # None of the variables have a strong correlation with the affair

affair.columns
### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
X = affair[['kids', 'vryunhap', 'unhap', 'avgmarr', 'hapavg', 'vryhap', 'antirel', 'notrel', 'slghtrel', 'smerel', 'vryrel', 'yrsmarr']]
y = affair['naffairs']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Model building 
from sklearn.linear_model import LogisticRegression
lm = LogisticRegression()
lm.fit(X_train,y_train)

predictions = lm.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

score = lm.score(X_test,y_test)
print(score)

# Feature Selection
from sklearn.feature_selection import RFECV
rfecv = RFECV(estimator=LogisticRegression(), step=1, cv=13, scoring='accuracy')
rfecv.fit(X, y)
print("Optimal number of features: %d" % rfecv.n_features_)
print('Selected features: %s' % list(X.columns[rfecv.support_]))

X.columns
# Plot number of features VS. cross-validation scores
plt.figure(figsize=(10,6))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show() # Selecting all the features gives the better result


from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss

# check classification scores of logistic regression
y_pred = lm.predict(X_test)
y_pred_proba = lm.predict_proba(X_test)[:, 1]
[fpr, tpr, thr] = roc_curve(y_test, y_pred_proba)
print('Train/Test split results:')
print(lm.__class__.__name__+" accuracy is %2.3f" % accuracy_score(y_test, y_pred))
print(lm.__class__.__name__+" log_loss is %2.3f" % log_loss(y_test, y_pred_proba))
print(lm.__class__.__name__+" auc is %2.3f" % auc(fpr, tpr))

idx = np.min(np.where(tpr > 0.95)) # index of the first threshold for which the sensibility > 0.95

plt.figure()
plt.plot(fpr, tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot([0,fpr[idx]], [tpr[idx],tpr[idx]], 'k--', color='blue')
plt.plot([fpr[idx],fpr[idx]], [0,tpr[idx]], 'k--', color='blue')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
plt.ylabel('True Positive Rate (recall)', fontsize=14)
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.show()

print("Using a threshold of %.3f " % thr[idx] + "guarantees a sensitivity of %.3f " % tpr[idx] +  
      "and a specificity of %.3f" % (1-fpr[idx]) + 
      ", i.e. a false positive rate of %.2f%%." % (np.array(fpr[idx])*100))