import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

election = pd.read_csv("C:/Users/lnram/Desktop/ExcelR/Module 9 - Logistic Regression/Assignments/election_data.csv")
election.head()

# Business Objective - Predicting Whether the client has subscribed a term deposit or not 

election.columns
election = election.drop('Election-id',1) # Removing ID column

# Check for missing values
election.isna().sum() # first row has NA values

election.dropna(inplace=True) # Removing NA values

# Basic Statistics
election.describe()

election.dtypes # All are numerical variables

#Skeweness
election.skew() # Few features are highly skewed

#Kurtosis
election.kurt() # Most of the skewed features are having extreme curtosis

#Graphical Representation
election.hist(grid=False)

election.columns
election['Result'].value_counts(normalize=True) #Balanced Data Set

sns.pairplot(data = election , hue = 'Result') #Higher the Year, Amount Spent and Popularity Rank, gives a positive election result


# Correlation Matrix
sns.heatmap(election.corr(),annot=True)  # Popularity Rank has a strong negative correlation

election.columns
### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
X = election[['Year', 'Amount Spent', 'Popularity Rank']]
y = election['Result']

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
rfecv = RFECV(estimator=LogisticRegression(), step=1, cv=5, scoring='accuracy')
rfecv.fit(X, y)
print("Optimal number of features: %d" % rfecv.n_features_)
print('Selected features: %s' % list(X.columns[rfecv.support_]))

# Plot number of features VS. cross-validation scores
plt.figure(figsize=(10,6))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show() 

# Feature selection doesn't improve the accuracy score

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