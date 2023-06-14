################## VORBEREITUNG ################## 
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
setwd("C:/Users/noel2/OneDrive/Studium Workspace/M.Sc. Betriebswirtschaftslehre/BAOR_Data Analytics Challange/DAC Shared Workspace/R Workspace")
ccdata <- read.csv("creditcard.csv")

# Preprocessing
ccdata <- ccdata[,-1]
ccdata[,-30] <- scale(ccdata[,-30])



################## FUNKTIONEN ################## 


##### Für K-fold Cross-Validation benötigte Basisfunktionen:
## Problem: 70% training ist schwierig mit Kreuzvalidierung abzubilden, d.h. 
## anpassen auf 80%
## -> K = 5 als Standardwert

# Aufteilen des Datensatzes in 5 teile:

# Input: Anzahl in wie viele subsets geteilt werden soll, Datensatz
splitIntoKSubsets <- function(K = 5, dataset = ccdata){
  subsets <- list()
  # split von 1-5
  split <- sample(1:K, size = length(dataset[,1]), replace = TRUE)
  for (i in 1:K){
    subsets <- append(subsets, list(subset(dataset, split == i)))
  }
  return (subsets)
}
# Output: list mit den subsets

## auswählen des x-ten Datensatzes als Testdatensatz, zusammenfassen der 
## restlichen als Trainingsdatensatz

# Input: Index des Subsets welcher Trainingsdatensatz sein soll, Liste mit Subsets
mergeSubsetsIntoTrainAndTest <- function(whichSubsetAsTest = 1, subsetList = splitIntoKSubsets()){
  test <- subsetList[[whichSubsetAsTest]]
  train <- data.frame()
  for (i in (1:length(subsetList))[-whichSubsetAsTest]){
    train <- rbind(train, subsetList[[i]])
  }
  return(list(
    "test" = test,
    "train" = train))
}
# Output: List mit train und test - Datensatz


##### Single run of Model (SMOTE and ML algorithm):

# Input: Hyperparameter von SMOTE, Train und Testdatensatz
singleModelRun <- function (kSMOTE = 5, nSMOTE = 577, train, test){
  
  # Perform SMOTE
  set.seed(1234)
  smote_ <- smotefamily::SMOTE(X = train[,-31], target = train$Class, K = kSMOTE, dup_size = nSMOTE)
  training <- smote_$data
  training <- training[,-31]
  training$Class <- as.factor(training$Class)
  
  
  ## train the random forest algorithm on the training data using the mlr and tidyverse packages
  
  # Define the task
  task <- makeClassifTask(data = training, target = "Class")
  
  # Set the learner
  learner <- makeLearner("classif.xgboost", predict.type = "prob")
  
  # Define the parameter set
  params <- makeParamSet(
    makeIntegerParam("nrounds", lower = 100, upper = 100),
    makeNumericParam("eta", lower = 0.3, upper = 0.3),
    makeNumericParam("max_depth", lower = 3, upper = 3),
    makeNumericParam("min_child_weight", lower = 3, upper = 3),
    makeNumericParam("subsample", lower = 0.5, upper = 0.5),
    makeNumericParam("colsample_bytree", lower = 0.5, upper = 0.5)
  )
  
  # Set the control for tuning
  ctrl <- makeTuneControlGrid()
  
  # Set resampling strategy
  rdesc <- makeResampleDesc("CV", iters = 5L)
  
  # Tune the hyperparameters
  tune_result <- tuneParams(learner, task = task, resampling = rdesc, par.set = params, control = ctrl)
  
  # Print the results
  print(tune_result)
  
  # Set the tuned parameters
  learner_tuned <- setHyperPars(learner, par.vals = tune_result$x)
  
  # Train the model
  final_model <- mlr::train(learner_tuned, task)
  
  # Make predictions on the test set
  test$Class <- as.factor(test$Class)
  
  # Make predictions on the test set
  test_pred <- predict(final_model, newdata = test, type = "prob")
  
  # Calculate AUC
  roc_obj <- performance(test_pred, measures = mlr::auc)
  #print(roc_obj)
  ######      auc 
  ######   0.9765212
  
  auc <- roc(test$Class, as.numeric(test_pred$data$response))
  #plot(auc, main = paste0("AUC= ", round(pROC::auc(auc),4)), col = "blue")
  
  
  ## Generate confusion matrix
  # Convert to factor
  test_pred$data$response <- as.factor(test_pred$data$response)
  test_pred$data$truth <- as.factor(test_pred$data$truth)
  
  cm = confusionMatrix(data = test_pred$data$response, reference = test_pred$data$truth)
  #print(cm)
  
  return(list(
    "ROC" = roc_obj, 
    "AUC" = auc$auc, 
    "ConfusionMatrix" = cm))
}
# Output: List mit auc, roc und confusion matrix


##### K-fold Cross-Validation:
## Einmal datensatz splitten
## schleife von 1:5, jedes mal ein anderes Subset als Testdatensatz wählen
## in der Schleife: Aktuellen Test und Train-Datensatz in Model geben & AOC speichern
## Nach durchlauf der Schlaufe: Durchschnitt von AOC bilden & returnen

# Input: Anzahl der gewünschten Subsets für Crossvalidation, Datensatz, Parameter für SMOTE
kFoldCrossValidate <- function(kForFold = 5, 
                               entireDataset = ccdata, 
                               kForSMOTE = 5,
                               nDesiredRatioOfClassesForSMOTE = 1.00){
  subsets <- splitIntoKSubsets(K = kForFold, dataset = entireDataset)
  vectorOfAOCresults <- c()
  for (i in 1:length(subsets)){
    inputDataForModelRun <- mergeSubsetsIntoTrainAndTest(whichSubsetAsTest = i, subsetList = subsets)
    # da in jedem subset ein leicht anderes anderes Verhältnis von min zu maj
    # klasse vorliegt, sollte der N parameter für jedes subset bestimmt werden.
    # nDesiredRatioOfClassesForSMOTE von 100% (= 1) bedeutet dabei, 
    # dass nach SMOTE gleich viele minority wie 
    # majority samples im Datensatz sein sollen (100% bedeutet also gleich viele,
    # das sollte wieder einen absoluten n Wert von ca. 577 ergeben)
    # Das Hyperparametertuning erfolgt somit später mit prozentwerten statt
    # absoluten werten. z.B. 120% (=1.2) würde also bedeuten, wir wollen
    # 20% mehr minority als majority samples nach SMOTE
    nAbsoluteSMOTE <- table(inputDataForModelRun$train$Class)[1]/table(inputDataForModelRun$train$Class)[2]
    nAbsoluteSMOTE <- round((nAbsoluteSMOTE-1)*nDesiredRatioOfClassesForSMOTE)
    
    resultsOfSingleRun <- singleModelRun(kSMOTE = kForSMOTE, 
                                         nSMOTE = nAbsoluteSMOTE, 
                                         train = inputDataForModelRun$train, 
                                         test = inputDataForModelRun$test)
    
    vectorOfAOCresults <- c(vectorOfAOCresults, resultsOfSingleRun$AUC)
  }

  return(list(
    "AUCResultsOfSingleRuns" = vectorOfAOCresults, 
    "AUCAverage" = mean(vectorOfAOCresults) 
  ))
}
# Output: List mit Durchschnitts-AUC und Vektor der AUCs der einzelnen Durchläufe 


##### Hyperparameteroptimierung:
## 
## Zunächst Bereiche der Hyperparameter definieren
## Für kForSmote: 1-15
## Für nDesiredRatioOfClassesForSMOTE: 0.5 - 1.2 (= 50-120%)
## 
## Dann Anzahl der random choises



# Input: kFold (für crossvalidation), der Datensatz, der auszuprobierende Wertebereich
#       für k (SMOTE) und n (SMOTE, siehe auch Dokumentation von kFoldCrossValidate),
tryRandomHyperparameter <- function (kForFold = 5,
                                     entireDataset = ccdata,
                                     kForSMOTERange = (1:15),
                                     nDesiredRatioSMOTERange = c(0.5,1.2)
                                     ){
  
  nDesiredRatioSMOTERange <- (nDesiredRatioSMOTERange[1]*100):(nDesiredRatioSMOTERange[2]*100)
  nRatioToTest <- sample(nDesiredRatioSMOTERange,1)/100
  kToTest <- sample(kForSMOTERange,1)
  
  crossValidatedAUC <- kFoldCrossValidate(kForFold, entireDataset, kForSMOTE = kToTest, nDesiredRatioOfClassesForSMOTE = nRatioToTest)
  
  resultVector <- c(kForFold, kToTest, nRatioToTest, crossValidatedAUC$AUCAverage)
  return(resultVector)
}
# Output: Vector der Testergebnisse mit: kFold (für crossvalidation), 
#         kSMOTE, nRatioSMOTE (siehe Dokumentation von kFoldCrossValidate) und die AUC



  

################## DURCHLAUF ################## 
## Die Durchschnittsperformance aus der Cross Validation sollte dauerhaft
## abgespeichert werden, da das Berechnen dieser Kombinationen extrem rechen-
## intensiv ist. D.h. sollen alle Parameter in hyperparameterTuning.csv 
## gespeichert werden

testedHyperparameters <<- read.csv("hyperparameterTuning.csv")[,-1]
colnames(testedHyperparameters) <- c("kFold", "kSMOTE", "nRatioSMOTE", "AUC Average")

for (i in 1:20){
  newRandomHyperparameterTestResult <- tryRandomHyperparameter(
    kForFold = 5,
    entireDataset = ccdata,
    kForSMOTERange = (1:15),
    nDesiredRatioSMOTERange = c(0.5,1.2)
  )
  
  testedHyperparameters <- rbind(testedHyperparameters, newRandomHyperparameterTestResult)
  cat("---------------------------- Abgeschlossener Durchlauf: ", i, " ----------------------------")
  
  write.csv(testedHyperparameters, "hyperparameterTuning.csv", row.names=TRUE)
}




