### Load the necessary libraries
library(tidyverse)
library(DMwR)
library(caTools)
library(smotefamily)
library(mlr)
library(pROC)
library(ROSE)


### Load the data
setwd("C:/Users/Vincent Bl/Desktop/DAC/")
ccdata <- read.csv("creditcard.csv")

# Remove 'Time' variable and scale the amount column
ccdata <- ccdata[,-1]
ccdata[,-30] <- scale(ccdata[,-30])

# Split into training/test data
set.seed(123)
split <- sample.split(ccdata$Class, SplitRatio = 0.7)
train <-  subset(ccdata, split == TRUE)
test <- subset(ccdata, split == FALSE)


## train the random forest algorithm on the training data using the mlr and tidyverse packages

# Define the task
classif_task2 <- makeClassifTask(id = "credit", data = train, target = "Class")

# Set the learner
learner2 <- makeLearner("classif.xgboost", predict.type = "prob", fix.factors.prediction = TRUE)

# Define the parameter set
params2 <- makeParamSet(
  makeIntegerParam("nrounds", lower = 100, upper = 100),
  makeNumericParam("eta", lower = 0.3, upper = 0.3),
  makeNumericParam("max_depth", lower = 3, upper = 3),
  makeNumericParam("min_child_weight", lower = 3, upper = 3),
  makeNumericParam("subsample", lower = 0.5, upper = 0.5),
  makeNumericParam("colsample_bytree", lower = 0.5, upper = 0.5)
)

# Set the control for tuning
ctrl2 <- makeTuneControlGrid()

# Set resampling strategy
rdesc2 <- makeResampleDesc("CV", iters = 5)

# Tune the hyperparameters
tune_result2 <- tuneParams(learner2, task = classif_task2, resampling = rdesc2, par.set = params2, control = ctrl2)

# Print the results
print(tune_result2)

# Set the tuned parameters
learner_tuned2 <- setHyperPars(learner2, par.vals = tune_result2$x)

# Train the model
final_model2 <- mlr::train(learner_tuned2, classif_task2)


# Make predictions on the test set
test_pred2 <- predict(final_model2, newdata = test, type = "prob")

# Calculate AUC
roc_obj2 <- performance(test_pred2, measures = mlr::auc)
print(roc_obj2)
#####  auc 
#####  0.972706

au2 <- roc(test$Class, as.numeric(test_pred2$data$response))
plot(auc2, main = paste0("AUC= ", round(pROC::auc(auc2),4)), col = "blue")


## Generate confusion matrix
# Convert to factor
test_pred2$data$response <- as.factor(test_pred2$data$response)
test_pred2$data$truth <- as.factor(test_pred2$data$truth)

cm2 = confusionMatrix(data = test_pred2$data$response, reference = test_pred2$data$truth)
print(cm2)

