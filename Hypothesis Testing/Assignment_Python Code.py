import pandas as pd
import numpy as np

cutlet = pd.read_csv('C:/Users/lnram/Desktop/ExcelR/Module 5 - Hypothesis Testing/Assignment/Cutlets.csv')

cutlet.columns = "unitA" , "unitB"

cutlet.isna().sum()
cutlet = cutlet.dropna()

from scipy import stats
print(stats.shapiro(cutlet.unitA))  # p= 0.32 high null fly - Data is normal
print(stats.shapiro(cutlet.unitB))  # p=0.5225 high null fly - Data is normal

print(stats.levene(cutlet.unitA , cutlet.unitB)) # p = 0.417 P high null fly - Variances are equal

stats.ttest_ind(cutlet.unitA , cutlet.unitB)  # p = 0.4723 P high null fly - diameters are equal

lab = pd.read_csv('C:/Users/lnram/Desktop/ExcelR/Module 5 - Hypothesis Testing/Assignment/labTAT.csv')

lab.columns = "one","two","three","four"


lab.isna().sum()
lab = lab.dropna()

stats.shapiro(lab.one)  # p= 0.55 high null fly - Data is normal
stats.shapiro(lab.two)   # p= 0.86 high null fly - Data is normal
stats.shapiro(lab.three)   # p= 0.42 high null fly - Data is normal
stats.shapiro(lab.four)   # p= 0.66 high null fly - Data is normal

stats.levene(lab.one,lab.two)  # p = 0.0675 P high low - Variances are unequal
stats.levene(lab.three,lab.two)  # p = 0.332 P high null fly - Variances are equal
stats.levene(lab.four,lab.three)  # p = 0.154 P high null fly - Variances are equal

# One Way Anova
from statsmodels.formula.api import ols
import statsmodels.api as sm
anov_mod = ols('one ~ two + three + four' , data = lab).fit()
aov_table=sm.stats.anova_lm(anov_mod, type=2)
print(aov_table)
  

buyer = pd.read_csv('C:/Users/lnram/Desktop/ExcelR/Module 5 - Hypothesis Testing/Assignment/BuyerRatio.csv')

count=pd.crosstab(buyer["Direction"],buyer["observed"])
count
Chisquares_results=stats.chi2_contingency(count)

Chi_square=[['','Test Statistic','p-value'],['Sample Data',Chisquares_results[0],Chisquares_results[1]]]
Chi_square  # p = 0.6603 P high null fly - Variances are equal


fan = pd.read_csv('C:/Users/lnram/Desktop/ExcelR/Module 5 - Hypothesis Testing/Assignment/Fantaloons.csv')

fan.isna().sum()
fan = fan.dropna()


count=pd.crosstab(fan["Weekdays"],fan["Weekend"])
count
Chisquares_results = stats.chi2_contingency(count)

Chi_square=[['','Test Statistic','p-value'],['Sample Data',Chisquares_results[0],Chisquares_results[1]]]
Chi_square  # p = 0.942 P high null fly - Variances are equal


