#load the necessary libraries
library(tidyverse)
library(randomForest)
library(caTools)
library(smotefamily)
library(caret)
library(mlr)
library(tibble)
library(corrplot)

# Load the dataset
#setwd("C:/Users/Vincent Bl/Desktop/DAC/")
setwd("C:/Users/Dennis/OneDrive/Dokumente/03_Master/05_Kurse/01_BA/04_DAC/")
ccdata <- read.csv("creditcard.csv")

# look at the data
View(ccdata)
summary(ccdata)         # summary statistics of the data
colSums(is.na(ccdata))  # check for NA in the data
table(ccdata$Class)     # absolute amount of class membership


### Preprocessing
# Split into training/test set
set.seed(123)
split <- sample.split(ccdata$Class, SplitRatio = 0.9)
train <- subset(ccdata, split == TRUE)
test <- subset(ccdata, split == FALSE)
table(train$Class)

# Split features from the target value
y_train <- train$Class
X_train <- train[,-30]

# Drop column 'Time' and scale column 'Amount' individually for train and test set
train <- train[,-1] %>% mutate(Amount = scale(Amount))
test <- test[,-1] %>% mutate(Amount = scale(Amount))


### some data exploration (plot of time and amount for the 2 classes)
# correlation
corr_plot <- corrplot(cor(train[,-c(1)]), method = "circle", type = "upper")



### Apply the original SMOTE Algorithmus to the train set
# Calculate number of synthetic samples for each minority instance
n_smote <- as.integer((255884 - 443)/443)

set.seed(1234)
smote_ <- smotefamily::SMOTE(X = X_train, target= y_train, K= 5, dup_size=n_smote)
train_smote <- (smote_$data)

# View the new balance in the dataset
table(train_smote$Class)
prop.table(table(train_smote$Class))

# data visualization
# Plot the first two features with the target value before SMOTE
ggplot(train, aes(x = V1, y = V2, color = factor(Class))) +geom_point() + ggtitle("Class distribution before SMOTE") + scale_color_manual(values = c("#E69F00", "#56B4E9"))

# Plot the first two features with the target value before SMOTE
ggplot(train_smote, aes(x = V1, y = V2, color = factor(class))) + geom_point() + ggtitle("Class distribution after SMOTE")+ scale_color_manual(values = c("#E69F00", "#56B4E9"))




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

