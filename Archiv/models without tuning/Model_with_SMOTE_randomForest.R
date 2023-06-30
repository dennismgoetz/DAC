library(tidyverse)
library(DMwR)
library(caTools)
library(smotefamily)
library(mlr)
library(pROC)
library(ROSE)

setwd("C:/Users/Vincent Bl/Desktop/DAC/")
ccdata <- read.csv("creditcard.csv")

ccdata <- ccdata[,-1]
ccdata[,-30] <- scale(ccdata[,-30])

set.seed(123)
split <- sample.split(ccdata$Class, SplitRatio = 0.7)
train <- subset(ccdata, split == TRUE)
test <- subset(ccdata, split == FALSE)


# Perform SMOTE
set.seed(1234)
smote_3 <- smotefamily::SMOTE(X = train[,-31], target = train$Class, K = 5, dup_size = 577)
training3 <- smote_3$data
training3 <- training3[,-31]
training3$Class <- as.factor(training3$Class)

# Define task
trainTask3 <- makeClassifTask(data = training3, target = "Class")

# Set up the random forest learner
learner3 <- makeLearner("classif.randomForest", predict.type = "prob", fix.factors.prediction = T)

# Define the parameter set
params3 <- makeParamSet(
  makeIntegerParam("ntree", lower = 10, upper = 10),
  makeIntegerParam("mtry", lower = 6, upper = 6),
  makeIntegerParam("nodesize", lower = 2, upper = 2),
  makeIntegerParam("maxnodes", lower = 2, upper = 2)
)

# Define the resampling strategy
rdesc3 <- makeResampleDesc("CV", iters = 5L)

# Set up the tuning
ctrl3 <- makeTuneControlGrid()

# Perform the tuning
res3 <- tuneParams(learner3, task = trainTask3, resampling = rdesc3, par.set = params3, control = ctrl3)

####### Extract the best model
####### best_model <- res$learner.model

# Re-train the model with optimal parameters
hyperp3 <- setHyperPars(learner3, par.vals = res3$x)
final_model3 <- mlr::train(hyperp3, trainTask3)

# Make predictions on the test set
test_pred3 <- predict(final_model3, newdata = test)
test_pred3$data$response
# Calculate AUC
roc_obj3 <- performance(test_pred3, measures = mlr::auc)
print(roc_obj3)
#####      auc 
#####    0.9572794

auc3 <- roc(test$Class, as.numeric(test_pred3$data$response))
plot(auc3, main = paste0("AUC= ", round(pROC::auc(auc3),4)), col = "blue")# auc 0.916

## Generate confusion matrix
# Convert to factor
test_pred3$data$response <- as.factor(test_pred3$data$response)
test_pred3$data$truth <- as.factor(test_pred3$data$truth)

cm3 = confusionMatrix(data = test_pred3$data$response, reference = test_pred3$data$truth)
print(cm3)

