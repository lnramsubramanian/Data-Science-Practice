corolla <- read_csv(file.choose())
View(corolla)

corolla <- corolla[c("Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight")]
View(corolla)

# Data Preparation & EDA

# checking for missing values

sum(is.na(corolla)) # No missing values

# 6 - point Summary

summary(corolla)
colnames(corolla)

attach(corolla)

library(dplyr)
summarise_if(corolla , is.numeric, sd) # Standard Deviation
summarise_if(corolla , is.numeric, max) - summarise_if(corolla , is.numeric, min) # Range

# Skewness
#install.packages("moments")
library(moments)
skewness(corolla)

#Kurtosis
kurtosis(corolla)

#Graphical Representation

#Distribution
dens <- apply(corolla , 2 , density)
plot(NA, xlim=range(sapply(dens, "[", "x")), ylim=range(sapply(dens, "[", "y")))
mapply(lines, dens, col=1:length(dens))
legend("topright", legend=names(dens), fill=1:length(dens))

#Box Plot
boxplot(corolla)

# Pair plot
pairs(corolla)

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
pairs(corolla, upper.panel=panel.cor,main="Scatter Plot Matrix with Correlation Coefficients")


#Linear Model
model1 <- lm(Price ~ . , data = corolla)
summary(model1)

# Output Summary:
# 1. p-values of most of the coeffs are less than 0.05 but cc and Doors are insignificant
# R squared adj and R squared are closeby
# R squared value is 0.864 - Determination is strong 


# Diagnostic Plots
#install.packages("car")
library(car)
plot(model1)


# Deletion Diagnostics for identifying influential variable
influence.measures(model1)
influenceIndexPlot(model1, id.n=3) 
influencePlot(model1, id.n=3) # Points 81, 222 & 961 seems to be outliers

#Linear Model excluding 81, 222 & 961
model2 <- lm(Price ~ . , data=corolla[-c(81 , 222 , 961),])
summary(model2)

# Model Summary
# 1. p-values the coeffs are less than 0.05
# R squared adj and R squared are closeby
# R squared value is 0.885 - Determination is strong ; better than model 1

# Variance Inflation Factors
vif(model2)  # No collinearity

# Added Variable Plots 
avPlots(model2, id.n=2, id.cex=0.8, col="red")

#Linear Model excluding 81, 222 & 961
model3 <- lm(Price ~ Age_08_04 + KM + HP + log(cc) + log(Doors) + Gears + Quarterly_Tax + Weight , data=corolla[-c(81 , 222 , 961),])
summary(model3)

# Model Summary
# 1. p-values the coeffs are less than 0.05
# R squared adj and R squared are closeby
# R squared value is 0.8857 - Determination is strong ; better than model 1


# Spliting data into train and test
n=nrow(corolla)
n1=n*0.7
n2=n-n1
train=sample(1:n,n1)
test = corolla[-train,]

pred=predict(model2,newdat=test)
actual=test$Price
error=actual-pred

test.rmse=sqrt(mean(error**2))
test.rmse

train.rmse = sqrt(mean(model2$residuals**2))
train.rmse 
