#load the necessary libraries
library(tidyverse)
library(randomForest)
library(caTools)
library(smotefamily)
library(caret)
library(mlr)
library(tibble)
library(corrplot)

# Load the data
setwd("C:/Users/Vincent Bl/Desktop/DAC/")
ccdata <- read.csv("creditcard.csv")

# look at the data
View(ccdata)
summary(ccdata)         # summary statistics of the data
colSums(is.na(ccdata))  # check for NA in the data
table(ccdata$Class)     # absolute amount of class membership


### some data exploration (plot of time and amount for the 2 classes)

# correlation
ccdata$Class <- as.numeric(ccdata$Class)
corr_plot <- corrplot(cor(ccdata[,-c(1)]), method = "circle", type = "upper")

# data visualization
ggplot(ccdata, aes(x = V1, y = V2, color = factor(Class))) +geom_point() + ggtitle("Class distribution before SMOTE")+ scale_color_manual(values = c("#E69F00", "#56B4E9"))




### Preprocessing

#Remove time and scale amount 
ccdata <- ccdata[,-1]

ccdata[,-30] <- scale(ccdata[,-30])


### Split into training/test set
set.seed(123)
split <- sample.split(ccdata$Class, SplitRatio = 0.7)
train <-  subset(ccdata, split == TRUE)
test <- subset(ccdata, split == FALSE)
table(train$Class)


















target <- ccdata$Class
feature <- ccdata[,-c(1,31)]

### SMOTE
set.seed(1234)
smote_ <- smotefamily::SMOTE(X = feature, target= target, K= 5, dup_size=577)
training <- (smote_$data)
training <- training[,-31]

table(training$Class)
prop.table(table(training$Class))






trainingTib <- as_tibble(training)


## Model training
task <- makeClassifTask(data= trainingTib, target="class")
forest <- makeLearner("classif.randomForest")

forestParamSet <- makeParamSet(makeIntegerParam("ntree", lower=300, upper=300),
                               makeIntegerParam("mtry", lower=6, upper=12),
                               makeIntegerParam("nodesize", lower=1, upper=5),
                               makeIntegerParam("maxnodes", lower=5, upper=20))

randsearch <- makeTuneControlRandom(maxit = 100)
cvForTuning <- makeResampleDesc("CV",iters = 5)

tunedForestPars <- tuneParams(forest, task= trainingTib, resampling= cvForTuning,
                              par.set = forestParamSet, control = randsearch)



tunedRorestPars$x


tunedForest <- setHyperPars(forest, par.vals = tunedForestPars$x)

tunedForestModel <- train(tunedForest, task)

forestModelData <- getLearnerModel(tunedForestModel)

species <- colnames(forestModelData$err.rate)













# Cross-validation
k <- 5 # Number of folds
folds <- createFolds(training$Class, k = k, list = TRUE, returnTrain = FALSE)

# Model Training using cross-validation
for (i in 1:k) {
  # Get the training and validation sets for the current fold
  train_fold <- training[-folds[[i]], ]
  validation_fold <- training[folds[[i]], ]
  
  # Fit Random Forest model using the training data
  model <- randomForest(target ~ ., data = train_fold)
  
  # Make predictions on the validation set
  predictions <- predict(model, newdata = validation_fold)
  
  # Evaluate the performance of the model
  performance <- your_performance_metric(validation_fold$Class, predictions)
  
  # Print the performance for the current fold
  cat("Fold", i, "Performance:", performance, "\n")
}

