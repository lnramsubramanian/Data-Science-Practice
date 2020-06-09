
library(readr)

election <- read_csv(file.choose())
View(election)

election = election[-1,]

election <- election[c("Result" , "Year" , "Amount Spent" , "Popularity Rank")]

colnames(election)
# Data Preparation & EDA

# checking for missing values

sum(is.na(election)) # No missing values

# 6 - point Summary

summary(election)
colnames(election)

attach(election)

library(dplyr)
summarise_if(election , is.numeric, sd) # Standard Deviation
summarise_if(election , is.numeric, max) - summarise_if(election , is.numeric, min) # Range

# Skewness
#install.packages("moments")
library(moments)
skewness(election)

#Kurtosis
kurtosis(election)

#Graphical Representation

#Distribution
dens <- apply(election , 2 , density)
plot(NA, xlim=range(sapply(dens, "[", "x")), ylim=range(sapply(dens, "[", "y")))
mapply(lines, dens, col=1:length(dens))
legend("topright", legend=names(dens), fill=1:length(dens))

#Box Plot
boxplot(election)


# Pair plot
pairs(election)

# Logistic Regression Model
library(glmnet)
model <- glm(Result~.,data=election,family = "binomial")
summary(model) #Output Summary

predict(model,election,type="response") 
# Output will be probability values

prob <- predict(model,election,type="response")
confusion<-table(prob>0.5,election$Result)
Accuracy<-sum(diag(confusion)/sum(confusion))

#ROC Value , Accuracy, and Log Loss

library(ROCR)
rocrpred<-prediction(prob,election$Result)
rocrperf<-performance(rocrpred,'tpr','fpr')

auc <- performance(rocrpred , measure = "auc")
auc <- auc@y.values
round(as.numeric(auc), digits = 4)

# ROC Curve
rocrpred<-prediction(prob,election$Result)
rocrperf<-performance(rocrpred,'tpr','fpr')
plot(rocrperf,colorize=T,text.adj=c(-0.2,1.7))
