library(dplyr) # for data manipulation
library(stringr) # for data manipulation
library(caret) # for sampling
library(caTools) # for train/test split
library(ggplot2) # for data visualization
library(DMwR) # for smote implementation

library(ROSE)# for ROSE sampling + ROC curve
library(rpart)# for decision tree model


library(xgboost) # for xgboost model

library(corrplot) # for correlations
library(Rtsne) # for tsne plotting
library(Rborist)# for random forest model


print("heeelllooo")

# function to set plot height and width
fig <- function(width, heigth){
  options(repr.plot.width = width, repr.plot.height = heigth)
}

df <- read.csv("creditcard.csv")






#Remove 'Time' variable
df <- df[,-1]

#Change 'Class' variable to factor
df$Class <- as.factor(df$Class)
levels(df$Class) <- c("Not_Fraud", "Fraud")

#Scale numeric variables

df[,-30] <- scale(df[,-30])

head(df)


set.seed(123)
split <- sample.split(df$Class, SplitRatio = 0.7)
train <-  subset(df, split == TRUE)
test <- subset(df, split == FALSE)



# smote
set.seed(9560)
smote_train <- SMOTE(Class ~ ., data  = train)

table(smote_train$Class)



### decison trees
#CART Model Performance on imbalanced data
set.seed(5627)

orig_fit <- rpart(Class ~ ., data = train)

#Evaluate model performance on test set
pred_orig <- predict(orig_fit, newdata = test, method = "class")

roc.curve(test$Class, pred_orig[,2], plotit = TRUE)


# Build smote model
set.seed(5627)
smote_fit <- rpart(Class ~ ., data = smote_train)


# AUC on up-sampled data
pred_smote <- predict(smote_fit, newdata = test)
roc.curve(test$Class, pred_smote[,2], plotit = FALSE)





