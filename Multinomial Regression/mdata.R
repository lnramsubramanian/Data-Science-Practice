mdata <- read_csv(file.choose())

View(mdata)

mdata = mdata[c(-1,-2)]

# Data Preparation & EDA

# checking for missing values

sum(is.na(mdata)) # No missing values

# 6 - point Summary

summary(mdata)
colnames(mdata)

attach(mdata)

library(dplyr)
summarise_if(mdata , is.numeric, sd) # Standard Deviation
summarise_if(mdata , is.numeric, max) - summarise_if(mdata , is.numeric, min) # Range

# Skewness
#install.packages("moments")
library(moments)
skewness(mdata[-c(1,2,3,4,9)]) # Excluding categorical features

#Kurtosis
kurtosis(mdata[-c(1,2,3,4,9)])  # Excluding categorical features

#Graphical Representation

#Distribution
dens <- apply(mdata[-c(1,2,3,4,9)] , 2 , density)
plot(NA, xlim=range(sapply(dens, "[", "x")), ylim=range(sapply(dens, "[", "y")))
mapply(lines, dens, col=1:length(dens))
legend("topright", legend=names(dens), fill=1:length(dens))

#Box Plot
boxplot(mdata[-c(1,2,3,4,9)])

# Pair plot
pairs(mdata[-c(1,2,3,4,9)])

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
pairs(mdata[-c(1,2,3,4,9)], upper.panel=panel.cor,main="Scatter Plot Matrix with Correlation Coefficients")

# Output variable - EDA
table(prog)

# Multinomial Model
colnames(mdata)
model <- multinom(prog ~ ., data=mdata)
summary(model)

##### Significance of Regression Coefficients###
z <- summary(model)$coefficients / summary(model)$standard.errors
p_value <- (1-pnorm(abs(z),0,1))*2

summary(model)$coefficients
p_value

# odds ratio 
exp(coef(model))

# Probabilities
prob <- fitted(model)
prob

# Model Accuracy
class(prob) #Type is Matrix
prob <- data.frame(prob)
prob["pred"] <- NULL #Creating a new column

# Custom function that returns the predicted value based on probability
get_names <- function(i){
  return (names(which.max(i)))
}

pred_name <- apply(prob,1,get_names)
prob$pred <- pred_name

# Confusion matrix
table(pred_name,mdata$prog)

# confusion matrix visualization
barplot(table(pred_name,mdata$prog),beside = T,col=c("red","lightgreen","blue","orange"),legend=c("bus","car","carpool","rail"),main = "Predicted(X-axis) - Legends(Actual)",ylab ="count")


# Accuracy 
mean(pred_name==mdata$prog)
