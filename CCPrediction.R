
library(tidyverse)
library(e1071)
library(ggplot2)
library(caret)
library(rmarkdown)
library(corrplot)
library(psych)
library(rpart)
library(rpart.plot)
library(kernlab)
library(pROC)
library(randomForest)
library(ggpubr)
library(PRROC)
options("scipen"=100, "digits"=4)


#Importing the data
client_data.df <- read.csv("UCI_Credit_Card.csv")
client_data.df <- client_data.df[-1]


#Data cleaning and Exploration

## Categorical variable cleaning
#Cleaning SEX variable
table(client_data.df$s)
ggplot(client_data.df, aes(SEX)) + geom_bar() #Looks good
#Cleaning EDUCATION variable
table(client_data.df$EDUCATION) #Values 0,5,6 are present and they are undocumented
client_data.df$EDUCATION[client_data.df$EDUCATION %in% c(0,5,6)] <- 4 #Assigning 0,5,6 as 4
ggplot(client_data.df, aes(EDUCATION)) + geom_bar()
#Cleaning MARRIAGE variable
table(client_data.df$MARRIAGE)  #Undocumented value 0 present
client_data.df$MARRIAGE[client_data.df$MARRIAGE == 0] <- 3
ggplot(client_data.df, aes(MARRIAGE)) + geom_bar()
#Cleaning PAY_X variables
names(client_data.df)[6] <- "PAY_1" #Renaming PAY_0 to PAY_1
paylabels <- c("PAY_1","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6")
for (p in paylabels) {
  client_data.df[client_data.df[,p] %in% c(-2,-1),p] <- 0
}
p1 <- ggplot(client_data.df, aes(PAY_1)) + geom_bar()
p2 <- ggplot(client_data.df, aes(PAY_2)) + geom_bar()
p3 <- ggplot(client_data.df, aes(PAY_3)) + geom_bar()
p4 <- ggplot(client_data.df, aes(PAY_4)) + geom_bar()
p5 <- ggplot(client_data.df, aes(PAY_5)) + geom_bar()
p6 <- ggplot(client_data.df, aes(PAY_6)) + geom_bar()
ggarrange(p1,p2,p3,p4,p5,p6)
#Making them factors
catLabels <- c("SEX", "EDUCATION", "MARRIAGE", "default.payment.next.month", paylabels)
client_data.df[catLabels] <- lapply(client_data.df[catLabels], as.factor)
## Numerical variables cleaning
#LIMIT_BAL variable
boxplot(client_data.df$LIMIT_BAL, main = "Boxplot of LIMIT_BAL variable", col = "RED") #Looks good
ggplot(client_data.df, aes(LIMIT_BAL)) + geom_histogram()
#AGE variable
hist(client_data.df$AGE, breaks = 12, col = "gray", main = "Histogram of Age", ylim = c(0,8000), labels = TRUE, xlab = "Age")
hist(client_data.df$AGE[client_data.df$default.payment.next.month == 1], breaks = 12, col = "gray", main = "Age-wise payment default chart", ylim = c(0,1500), labels = TRUE, xlab = "Age") #Looks good
#BILL_AMTX variables
billLabels <- c("BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6")
summary(client_data.df[billLabels]) #Looks good
#PAY_AMTX variables
amtLabels <- c("PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6")
summary(client_data.df[amtLabels]) #Looks good
summary(client_data.df[c("LIMIT_BAL","AGE", billLabels, amtLabels)])
sum(is.na(client_data.df[c("LIMIT_BAL","AGE", billLabels, amtLabels)]))
```

_#Correlation plots and distribution wrt output variable

# SEX, EDUCATION, MARRIAGE, AGE, LIMIT_BAL vs default.next.payment.month 
ggplot(client_data.df, aes(x=client_data.df$SEX, fill = default.payment.next.month)) + geom_bar() +
  xlab("Sex") + theme_bw() + ggtitle("Distribution of defaults across SEX") #Looks balanced; Female gender has more defaults
ggplot(client_data.df, aes(x=client_data.df$EDUCATION, fill = default.payment.next.month)) + geom_bar() +
  xlab("Education") + theme_bw() + ggtitle("Distribution of defaults across EDUCATION") #Looks balanced; University education has more defaults
ggplot(client_data.df, aes(x=client_data.df$MARRIAGE, fill = default.payment.next.month)) + geom_bar() +
  xlab("Marriage") + theme_bw() + ggtitle("Distribution of defaults across MARRIAGE") #Looks balanced; Singles have more defaults
ggplot(client_data.df, aes(AGE, fill = default.payment.next.month)) + geom_bar() +
  xlab("Age") + theme_classic() + ggtitle("Distribution of defaults across AGE") #Looks balanced; Age group 25-35 have more defaults
ggplot(client_data.df, aes(x=client_data.df$LIMIT_BAL, fill = default.payment.next.month)) + geom_histogram(binwidth = 1000) +
  stat_bin(bins = 30) + xlab("Credit Limit") + theme_classic() + ggtitle("Distribution of defaults across LIMIT_BAL") #Looks balanced
# Correlation among Numerical variables
corrplot(cor(client_data.df[billLabels]), method = "color", type = "upper",
         addCoef.col = "black",number.cex = 0.75) #Correlation among variables high; Expected because monthly bills are more or less same for a specific customer
corrplot(cor(client_data.df[amtLabels]), method = "color", type = "upper",
         addCoef.col = "black", number.cex = 0.75) #No Correlations exist


## Decision Tree implementation

set.seed(40)
train1.index <- sample(c(1:nrow(client_data.df)), nrow(client_data.df)*0.7)
train1.df <- client_data.df[train1.index, ]
valid1.df <- client_data.df[-train1.index, ]
train.ct <- rpart(default.payment.next.month ~ ., data = train1.df, method = "class", cp = 0.001, maxdepth = 4)
prp(train.ct, type = 1, extra = 1, under = TRUE)
train.ct.pred <- predict(train.ct, train1.df, type = "class")
confusionMatrix(train.ct.pred, as.factor(train1.df$default.payment.next.month))
valid.ct.pred <- predict(train.ct, valid1.df, type = "class")
confusionMatrix(valid.ct.pred, as.factor(valid1.df$default.payment.next.month))
#accuracy - 0.822, roc - 0.691 
valid1.roc <- predict(train.ct, valid1.df, type = "prob")
roc.dt <- roc(valid1.df$default.payment.next.month, valid1.roc[,2])
plot.roc(roc.dt)
auc(roc.dt)

