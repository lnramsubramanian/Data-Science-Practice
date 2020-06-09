install.packages("readr")
library(readr)

cutlet <- read_csv(file.choose())
View(cutlet)
colnames(cutlet) <- c("unitA","unitB")

attach(cutlet)

shapiro.test(unitA) # p= 0.32 high null fly - Data is normal

shapiro.test(unitB) # p=0.5225 high null fly - Data is normal

var.test(unitA,unitB) # p = 0.3136 P high null fly - Variances are equal

t.test(unitA , unitB , alternative = "two.sided" , conf.level = 0.95)  # p = 0.4723 P high null fly - diameters are equal

lab <- read_csv(file.choose())
View(lab)

colnames(lab) <- c("one","two","three","four")
attach(lab)

shapiro.test(one)  # p= 0.55 high null fly - Data is normal
shapiro.test(two)   # p= 0.86 high null fly - Data is normal
shapiro.test(three)   # p= 0.42 high null fly - Data is normal
shapiro.test(four)   # p= 0.66 high null fly - Data is normal

var.test(one,two)  # p = 0.1675 P high null fly - Variances are equal
var.test(three,two)  # p = 0.2742 P high null fly - Variances are equal
var.test(four,three)  # p = 0.3168 P high null fly - Variances are equal

stacked_data <- stack(lab)

anova_results <- aov(values ~ ind , data = stacked_data)
summary(anova_results)  # p = 2e-16 P low null go - Atleast one TAT is unequal


buyer <- read_csv(file.choose())

attach(buyer)

table(observed,Direction)


chisq.test(table(observed,Direction))  # p = 0.6603 P high null fly - Variances are equal


cof <- read_csv(file.choose())
stacked_data <- stack(cof)

table(stacked_data)

chisq.test(table(stacked_data))  # p = 0.2771 P high null fly - error %  are equal

fan <- read_csv(file.choose())

stacked_data <- stack(fan)

chisq.test(table(stacked_data)) # p = 8.53e-05 P low null go - There is a significance differnce in % of male & female walk in to the store depending on the days
