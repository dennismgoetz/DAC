# Load the necessary libraries
library(tidyverse)
library(randomForest)
library(caTools)
library(smotefamily)
library(caret)
library(mlr)
library(tibble)
library(xgboost)
library(pROC)

# Load the data
setwd("C:/Users/Vincent Bl/Desktop/DAC/")
ccdata <- read.csv("creditcard.csv")

# Preprocessing
ccdata <- ccdata[,-1]
ccdata[,-30] <- scale(ccdata[,-30])

# Split into training/test data
set.seed(123)
split <- sample.split(ccdata$Class, SplitRatio = 0.7)
train <- subset(ccdata, split == TRUE)
test <- subset(ccdata, split == FALSE)

# Perform SMOTE
set.seed(1234)
smote_ <- smotefamily::SMOTE(X = train[,-31], target = train$Class, K = 5, dup_size = 577)
training <- smote_$data
training <- training[,-31]
training$Class <- as.factor(training$Class)


## train the random forest algorithm on the training data using the mlr and tidyverse packages

# Define the task
classif_task <- makeClassifTask(data = training, target = "Class")

# Set the learner
learner <- makeLearner("classif.logreg", predict.type = "prob", fix.factors.prediction = TRUE)


# Define the parameter set
  ## es gubt keine hyperparameter

# Set the control for tuning
#ctrl <- makeTuneControlRandom(maxit = 2)

# Set resampling strategy
#rdesc <- makeResampleDesc("CV", iters = 5L)

# Tune the hyperparameters
#tune_result <- tuneParams(learner, task = classif_task, resampling = rdesc,  measures = mlr::auc)

# Print the results
#print(tune_result)

# Set the tuned parameters
#learner_tuned <- setHyperPars(learner, par.vals = tune_result$x)

# Train the model
final_model <- mlr::train(learner, classif_task)

# Make predictions on the test set
test$Class <- as.factor(test$Class)

# Make predictions on the test set
test_pred <- predict(final_model, newdata = test, type = "prob")

# Calculate AUC
roc_obj <- performance(test_pred, measures = mlr::auc)
print(roc_obj)
######      auc 
######   0.9765212

auc <- roc(test$Class, as.numeric(test_pred$data$response))
plot(auc, main = paste0("AUC= ", round(pROC::auc(auc),4)), col = "blue")


## Generate confusion matrix
# Convert to factor
test_pred$data$response <- as.factor(test_pred$data$response)
test_pred$data$truth <- as.factor(test_pred$data$truth)

cm = confusionMatrix(data = test_pred$data$response, reference = test_pred$data$truth)
print(cm)
