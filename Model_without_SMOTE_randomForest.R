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




# Define task
trainTask1 <- makeClassifTask(data = train, target = "Class")

# Set up the random forest learner
learner1 <- makeLearner("classif.randomForest", predict.type = "prob", fix.factors.prediction = T)

# Define the parameter set
params1 <- makeParamSet(
  makeIntegerParam("ntree", lower = 10, upper = 10),
  makeIntegerParam("mtry", lower = 6, upper = 6),
  makeIntegerParam("nodesize", lower = 2, upper = 2),
  makeIntegerParam("maxnodes", lower = 2, upper = 2)
)

# Define the resampling strategy
rdesc1 <- makeResampleDesc("CV", iters = 5L)

# Set up the tuning
ctrl1 <- makeTuneControlGrid()

# Perform the tuning
res1 <- tuneParams(learner1, task = trainTask1, resampling = rdesc1, par.set = params1, control = ctrl1)

####### Extract the best model
####### best_model <- res$learner.model

# Re-train the model with optimal parameters
hyerp1 <- setHyperPars(learner1, par.vals = res1$x)
final_model1 <- mlr::train(hyerp1, trainTask1)

# Make predictions on the test set
test_pred1 <- predict(final_model1, newdata = test)

# Calculate AUC
roc_obj1 <- performance(test_pred1, measures = mlr::auc)
print(roc_obj1)
#####      auc 
#####    0.901754 

auc1 <- roc(test$Class, as.numeric(test_pred1$data$response))
plot(auc1, main = paste0("AUC= ", round(pROC::auc(auc1),4)), col = "blue")




## Generate confusion matrix
# Convert to factor
test_pred1$data$response <- as.factor(test_pred1$data$response)
test_pred1$data$truth <- as.factor(test_pred1$data$truth)

cm1 = confusionMatrix(data = test_pred1$data$response, reference = test_pred1$data$truth)
print(cm1)

