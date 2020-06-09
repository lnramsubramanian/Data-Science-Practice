
library(readr)

affair <- read_csv(file.choose())
View(affair)

# Data Preparation & EDA

# checking for missing values

sum(is.na(affair)) # No missing values

# 6 - point Summary

summary(affair)
colnames(affair)

# Dropping index column
affair <- affair[c( "naffairs" , "kids" , "vryunhap" , "unhap" , "avgmarr" , "hapavg" , "vryhap" , "antirel" , "notrel",
                                "slghtrel" , "smerel" , "vryrel"  , "yrsmarr1" , "yrsmarr2" , "yrsmarr3" , "yrsmarr4" , "yrsmarr5" , "yrsmarr6")]

attach(affair)

library(dplyr)
summarise_if(affair , is.numeric, sd) # Standard Deviation
summarise_if(affair , is.numeric, max) - summarise_if(affair , is.numeric, min) # Range

# Skewness
#install.packages("moments")
library(moments)
skewness(affair)

#Kurtosis
kurtosis(affair)

#Graphical Representation

#Distribution
dens <- apply(affair , 2 , density)
plot(NA, xlim=range(sapply(dens, "[", "x")), ylim=range(sapply(dens, "[", "y")))
mapply(lines, dens, col=1:length(dens))
legend("topright", legend=names(dens), fill=1:length(dens))

#Box Plot
boxplot(affair) # The affair values are above 1

affair$naffairs <- ifelse(affair$naffairs >= 1 , 1 , 0) 

# Pair plot
pairs(affair)

# Logistic Regression Model
library(glmnet)
model <- glm(naffairs~.,data=affair,family = "binomial")
summary(model) #Output Summary

predict(model,affair,type="response") 
# Output will be probability values

prob <- predict(model,affair,type="response")
confusion<-table(prob>0.5,affair$naffairs)
Accuracy<-sum(diag(confusion)/sum(confusion))

#ROC Value , Accuracy, and Log Loss

library(ROCR)
rocrpred<-prediction(prob,affair$naffairs)
rocrperf<-performance(rocrpred,'tpr','fpr')

auc <- performance(rocrpred , measure = "auc")
auc <- auc@y.values
round(as.numeric(auc), digits = 4)

# ROC Curve
rocrpred<-prediction(prob,affair$naffairs)
rocrperf<-performance(rocrpred,'tpr','fpr')
plot(rocrperf,colorize=T,text.adj=c(-0.2,1.7))
