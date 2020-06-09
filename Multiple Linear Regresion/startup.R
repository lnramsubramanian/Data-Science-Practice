getwd()
list.files()
library(readr)

startup <- read_csv("50_Startups.csv")
View(startup)

# Data Preparation & EDA

# checking for missing values

sum(is.na(startup)) # No missing values

# 6 - point Summary

summary(startup)
colnames(startup)
colnames(startup) <- c("RD_spend" , "Admin" , "M_spend" , "state" , "profit")

attach(startup)

library(dplyr)
summarise_if(startup , is.numeric, sd) # Standard Deviation
summarise_if(startup , is.numeric, max) - summarise_if(startup , is.numeric, min) # Range

# Skewness
#install.packages("moments")
library(moments)
skewness(startup[-4])

#Kurtosis
kurtosis(startup[-4])

#Graphical Representation

#Distribution
dens <- apply(startup[-4] , 2 , density)
plot(NA, xlim=range(sapply(dens, "[", "x")), ylim=range(sapply(dens, "[", "y")))
mapply(lines, dens, col=1:length(dens))
legend("topright", legend=names(dens), fill=1:length(dens))

#Box Plot
boxplot(startup[-4])

# Pair plot
pairs(startup[-4])


#Linear Model
model1 <- lm(profit ~ RD_spend + Admin + M_spend + state)
summary(model1)

# Model Summary
# R Squared Value: 0.9439 and R Squared adjusted value are closeby
# Most of the P values are above 0.05 
# Fstatistics p value is less than 0.05 


# Correlation Matrix
panel.cor <- function(x, y, digits=2, prefix="", cex.cor)
{
  #usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  r = (cor(x, y))
  txt <- format(c(r, 0.123456789), digits=digits)[1]
  txt <- paste(prefix, txt, sep="")
  if(missing(cex.cor)) cex <- 0.4/strwidth(txt)
  text(0.5, 0.5, txt, cex = cex)
}
pairs(startup[-4], upper.panel=panel.cor,main="Scatter Plot Matrix with Correlation Coefficients")

#Partial Correlation
#install.packages("corpcar")
library(copcar)
cor(startup)

cor2pcor(cor(startup))


# Diagnostic Plots
#install.packages("car")
library(car)
plot(model1)# Residual Plots, QQ-Plos, Std. Residuals vs Fitted, Cook's distance


# Deletion Diagnostics for identifying influential variable
influence.measures(model1)
influenceIndexPlot(model1, id.n=3) 
influencePlot(model1, id.n=3) # Points 49 & 50 seems to be outliers

#Linear Model excluding 49 & 50
model2 <- lm(profit ~ RD_spend + Admin + M_spend + state , data=startup[-c(49,50),])
summary(model2)

# Model Summary
# R Squared Value: 0.9628 and R Squared adjusted value are closeby - Better than model 1
# Most of the P values are above 0.05 
# Fstatistics p value is less than 0.05 


# Variance Inflation Factors
vif(model2)  # No collinearity

# Added Variable Plots 
avPlots(model2, id.n=2, id.cex=0.8, col="red")

# Spliting data into train and test
n=nrow(startup)
n1=n*0.7
n2=n-n1
train=sample(1:n,n1)
test = startup[-train,]

pred=predict(model2,newdat=test)
actual=test$profit
error=actual-pred

test.rmse=sqrt(mean(error**2))
test.rmse

train.rmse = sqrt(mean(model2$residuals**2))
train.rmse 
