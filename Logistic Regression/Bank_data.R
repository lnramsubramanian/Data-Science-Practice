
library(readr)

bank <- read_csv(file.choose())
View(bank)

# Data Preparation & EDA

# checking for missing values

sum(is.na(bank)) # No missing values

# 6 - point Summary

summary(bank)
colnames(bank)

attach(bank)

library(dplyr)
summarise_if(bank , is.numeric, sd) # Standard Deviation
summarise_if(bank , is.numeric, max) - summarise_if(bank , is.numeric, min) # Range

# Skewness
#install.packages("moments")
library(moments)
skewness(bank)

#Kurtosis
kurtosis(bank)

#Graphical Representation

#Distribution
dens <- apply(bank , 2 , density)
plot(NA, xlim=range(sapply(dens, "[", "x")), ylim=range(sapply(dens, "[", "y")))
mapply(lines, dens, col=1:length(dens))
legend("topright", legend=names(dens), fill=1:length(dens))

#Box Plot
boxplot(bank)


# Pair plot
pairs(bank)

# Logistic Regression Model
library(glmnet)
model <- glm(y~.,data=bank,family = "binomial")
summary(model) #Output Summary

predict(model,bank,type="response") 
# Output will be probability values

prob <- predict(model,bank,type="response")
confusion<-table(prob>0.5,bank$y)
Accuracy<-sum(diag(confusion)/sum(confusion))

#ROC Value , Accuracy, and Log Loss

library(ROCR)
rocrpred<-prediction(prob,bank$y)
rocrperf<-performance(rocrpred,'tpr','fpr')

auc <- performance(rocrpred , measure = "auc")
auc <- auc@y.values
round(as.numeric(auc), digits = 4)

# ROC Curve
rocrpred<-prediction(prob,bank$nbanks)
rocrperf<-performance(rocrpred,'tpr','fpr')
plot(rocrperf,colorize=T,text.adj=c(-0.2,1.7))
