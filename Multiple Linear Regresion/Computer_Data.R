comp <- read_csv(file.choose())

View(comp)

comp = comp[-1]

# Data Preparation & EDA

# checking for missing values

sum(is.na(comp)) # No missing values

# 6 - point Summary

summary(comp)
colnames(comp)

attach(comp)

library(dplyr)
summarise_if(comp , is.numeric, sd) # Standard Deviation
summarise_if(comp , is.numeric, max) - summarise_if(comp , is.numeric, min) # Range

# Skewness
#install.packages("moments")
library(moments)
skewness(comp[-c(7,8,9)]) # Excluding categorical features

#Kurtosis
kurtosis(comp[-c(7,8,9)])  # Excluding categorical features

#Graphical Representation

#Distribution
dens <- apply(comp[-c(7,8,9)] , 2 , density)
plot(NA, xlim=range(sapply(dens, "[", "x")), ylim=range(sapply(dens, "[", "y")))
mapply(lines, dens, col=1:length(dens))
legend("topright", legend=names(dens), fill=1:length(dens))

#Box Plot
boxplot(comp[-c(7,8,9)])

# Pair plot
pairs(comp[-c(7,8,9)])

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
pairs(comp[-c(7,8,9)], upper.panel=panel.cor,main="Scatter Plot Matrix with Correlation Coefficients")


#Linear Model
model1 <- lm(price ~ . , data = comp)
summary(model1)

# Output Summary:
# 1. p-values less than alpha
# R squared adj and R squared are closeby
# R squared value is 0.7756 - Determination is strong 


# Diagnostic Plots
#install.packages("car")
library(car)
plot(model1)


# Deletion Diagnostics for identifying influential variable
influence.measures(model1)
influenceIndexPlot(model1, id.n=3) 
influencePlot(model1, id.n=3) # Points 1441 , 1701 seems to be outliers

#Linear Model excluding 1441 , 1701
model2 <- lm(price ~ . , data=comp[-c(1441 , 1701),])
summary(model2)

# Model Summary
# 1. p-values the coeffs are less than 0.05
# R squared adj and R squared are closeby
# R squared value is 0.777 - Determination is strong ; better than model 1

# Variance Inflation Factors
vif(model2)  # No collinearity

# Spliting data into train and test
n=nrow(comp)
n1=n*0.7
n2=n-n1
train=sample(1:n,n1)
test = comp[-train,]

pred=predict(model2,newdat=test)
actual=test$price
error=actual-pred

test.rmse=sqrt(mean(error**2))
test.rmse

train.rmse = sqrt(mean(model2$residuals**2))
train.rmse 
