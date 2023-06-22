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
library(ggplot2)
library(parallelMap)
library(parallel)
library(glmnet)
library(tidymodels)

parallelStartSocket(detectCores())

# Load the data
setwd("C:/Users/noel2/OneDrive/Studium Workspace/M.Sc. Betriebswirtschaftslehre/BAOR_Data Analytics Challange/DAC Shared Workspace/R Workspace")
#setwd("C:/Users/Dennis/OneDrive/Dokumente/03_Master BAOR/05_Kurse/01_Business Analytics/04_Data Analytics Challenge/")
#setwd("C:/Users/Vincent Bl/Desktop/DAC/")

ccdata <- read.csv("creditcard.csv")

# Preprocessing
ccdata <- ccdata[,-1]
ccdata[,-30] <- scale(ccdata[,-30])



################## FUNKTIONEN 1: Basis-Funktionen ################## 


##### Für K-fold Cross-Validation benötigte Basisfunktionen:
## Problem: 70% training ist schwierig mit Kreuzvalidierung abzubilden, d.h. 
## anpassen auf 75%
## -> K = 4 als Standardwert

# Aufteilen des Datensatzes in k teile:

# Input: Anzahl in wie viele subsets geteilt werden soll, Datensatz
splitIntoKSubsets <- function(K = 4, dataset = ccdata){
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

# Input: Index des Subsets welcher Testdatensatz sein soll, Liste mit Subsets
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

##### Für Random - Undersampling
# Input: undersamplingFaktor (0 = kein Undersampling, 0.99 = Um 99 %, also 1% der urspr. Anzahl), Trainingsdatensatz
undersample <- function (undersamplingFactor = 0.99, dataset = ccdata){

  
  undersampled_dataset <- dataset[dataset$Class == 1,]
  
  maj_class_samples <- dataset[dataset$Class==0,]
  wanted_number_of_maj_samples <- round(length(maj_class_samples[,1])*(1-undersamplingFactor))
  
  maj_class_selection_vector <- as.logical(maj_class_samples$Class)
  
  while (sum(maj_class_selection_vector) < wanted_number_of_maj_samples){
    not_yet_included_maj_samples <- (1:length(maj_class_samples[,1]))[maj_class_selection_vector==FALSE]
    index_of_random_maj_sample_to_include <- sample(not_yet_included_maj_samples,1)
    maj_class_selection_vector[index_of_random_maj_sample_to_include] <- TRUE
  }
  
  undersampled_dataset <- rbind(undersampled_dataset, maj_class_samples[maj_class_selection_vector==TRUE,])
  
  return (undersampled_dataset)
}
# Output: Datensatz nach Undersampling



##### ASN-SMOTE
asn_smote <- function(train, K, dup_size) {
  
  train_feat <- train[,1:29] #Features of train  (= T in the Pseudo Code)
  train_target <- train$Class #Target value of train
  
  
  train_feat_matrix <- as.matrix(train_feat)
  train_Majority <- train[train_target == 0,]
  train_Minority <- train[train_target == 1,]
  
  ####### [1:29] muss man noch Ã¤ndern, sodass man Function fÃ¼r andere DatensÃ¤tze replizieren kann
  train_Minority_feat <- train_Minority[,1:29]   #Features of Minority set (= P in the Pseudo code)
  
  # Algorithm 1: Noise filtering
  dis_matrix <- proxy::dist(train_Minority_feat, train_feat)
  
  
  ##########################################################################################################
  
  
  index_knn <- list()
  
  # Tests
  #dis_matrix[1,]
  #order(dis_matrix[1,])
  #order(dis_matrix[1,])[1:6]
  #dis_matrix[1, 490]
  #dis_matrix[1, 71985]
  #dis_matrix[1, 24231]
  
  #train[490,]
  #train_Minority_feat[1,]
  
  #sum(train_Minority_feat[1,] - train_feat[490,]) #sollte 0 sein (passt)
  #sum(train_Minority_feat[1,] - train_feat[24977,]) #sollte 4.352906 sein (passt nicht)
  #sum(abs(train_Minority_feat[1,] - train_feat[24977,]))
  #rownames(dis_matrix)[1]
  
  
  for (i in 1:nrow(train_Minority_feat)) {
    index_knn[[rownames(dis_matrix)[i]]] <- order(dis_matrix[i,])[2:(K+1)]
    for (j in 1:K) {
      if (train_target[index_knn[[i]][j]] == 0 ) {
        index_knn[[i]][j] <- NaN
      }
    }
  }
  
  
  Mu <- vector()
  for (i in length(index_knn):1) { 
    if (is.nan(index_knn[[i]][1])) {
      Mu[i] <- names(index_knn[i])
      index_knn <- index_knn[-i]
    }
  }
  
  Mu <- na.omit(Mu)
  Mu <- Mu[1:length(Mu)]
  
  # Variante Dennis
  for (i in 1:length(index_knn)) {
    for (j in 1:K) {
      if (is.nan(index_knn[[i]][j])) {
        index_knn[[i]] <- index_knn[[i]][1:(j-1)]
        break
      }
    }
  }
  
  
  
  # Check for duplicates in each list of qualified neighbors
  #  # Create a duplicate
  #  index_knn['258404']
  #  index_knn[['258404']][2] <- index_knn[['258404']][1]
  #  index_knn['258404']
  
  
  #  duplicates_list <- list()
  #  for (i in 1:length(index_knn)) {
  
  #    duplicates <- duplicated(index_knn[[i]])
  
  #    if (any(duplicates)) {
  #      duplicates_list[[i]] <- index_knn[[i]][duplicates]
  #    }
  #  }
  #  duplicates_list
  
  synthetic <- list()
  for(i in names(index_knn)) {
    for(j in seq_len(dup_size)) {
      random_n <- sample(seq_along(index_knn[[i]]), 1)   # random number in the length of the best index
      dif <- train_feat_matrix[index_knn[[i]][random_n],] - train_feat_matrix[i,]  ## dif von der dis matrix
      randomNum <- runif(1)
      synthetic_instance <- train_feat_matrix[i,] + randomNum * dif
      synthetic[[length(synthetic) + 1]] <- synthetic_instance
    }
  }
  
  
  ###########################################################################################################
  
  # assign "Class" label = 1 to the synthtic points
  synthetic_df <- do.call(rbind, synthetic)
  synthetic_df <- as.data.frame(synthetic_df)
  synthetic_labels <- rep(1, length(synthetic))
  synthetic_df$Class <- synthetic_labels
  
  # Combine original train set with synthetic set
  asn_train <- rbind(train, synthetic_df)
  
  # remove unqualified points of minority class
  #asn_train <<- asn_train[!(rownames(asn_train) %in% Mu), ] #warum löschen?
  
  return(asn_train)
}






################## FUNKTIONEN 2: Kombi-Funktionen von SMOTE & Classifier ################## 



# Input: Hyperparameter von SMOTE, Train und Testdatensatz
run_smote_with_xgboost <- function (kSMOTE = 5, nSMOTE = 577, train, test){
  
  # Perform SMOTE
  set.seed(1234)
  smote_ <- smotefamily::SMOTE(X = train[,-31], target = train$Class, K = kSMOTE, dup_size = nSMOTE)
  training <- smote_$data
  training <- training[,-31]
  training$Class <- as.factor(training$Class)
  print("diag")
  print(table(training$Class))
  
  ## train the random forest algorithm on the training data using the mlr and tidyverse packages
  
  # Define the task
  task <- makeClassifTask(data = training, target = "Class")
  
  # Set the learner
  learner <- makeLearner("classif.xgboost", predict.type = "prob")
  
  # Define the parameter set
  params <- makeParamSet(
    makeIntegerParam("nrounds", lower = 100, upper = 500),  # 100 500
    makeNumericParam("eta", lower = 0.05, upper = 0.3)  # 0.05 0.3
  )
  
  # Set the control for tuning
  ctrl <- makeTuneControlRandom(maxit=1L)
  
  # Set resampling strategy
  rdesc <- makeResampleDesc("CV", iters = 2L)
  
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


# Input: Hyperparameter von SMOTE, Train und Testdatensatz
run_asn_smote_with_xgboost <- function (kSMOTE = 5, nSMOTE = 577,  train_undersampled, train_not_undersampled, test){
  
  # Perform SMOTE
  set.seed(1234)
  # ASN mit nicht undersampled trainingsdatensatz
  asn_train <- asn_smote(train_not_undersampled, K = kSMOTE, dup_size = nSMOTE)
  # print("asn_train")
  # print(table(asn_train$Class))
  
  # print("train_undersampled")
  # print(table(train_undersampled$Class))
  
  # Zusammenfügen von minority class aus ASN Ergebnis und majority class aus undersampling datensatz
  training <- rbind(train_undersampled[train_undersampled$Class==0,], asn_train[asn_train$Class==1,])
  # print("training")
  # print(table(training$Class))
  
  training$Class <- as.factor(training$Class)
  
  
  ## train the random forest algorithm on the training data using the mlr and tidyverse packages
  
  # Define the task
  task <- makeClassifTask(data = training, target = "Class")
  
  # Set the learner
  learner <- makeLearner("classif.xgboost", predict.type = "prob")
  
  # Define the parameter set
  params <- makeParamSet(
    makeIntegerParam("nrounds", lower = 100, upper = 500),  # 100 500
    makeNumericParam("eta", lower = 0.05, upper = 0.3)  # 0.05 0.3
  )
  
  # Set the control for tuning
  ctrl <- makeTuneControlRandom(maxit=1L)
  
  # Set resampling strategy
  rdesc <- makeResampleDesc("CV", iters = 2L)
  
  # Tune the hyperparameters
  tune_result <- tuneParams(learner, task = task, resampling = rdesc, par.set = params, control = ctrl, measures= mlr::auc)
  
  # Print the results
  print(tune_result)
  
  # Set the tuned parameters
  learner_tuned <- setHyperPars(learner, par.vals = tune_result$x)
  
  # Train the model
  final_model <- mlr::train(learner_tuned, task)
  
  best_nrounds <- tune_result$x$nrounds
  best_eta <- tune_result$x$eta
  
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


# Input: Hyperparameter von SMOTE, Train und Testdatensatz
run_smote_with_logistic_regression <- function (kSMOTE = 5, nSMOTE = 577, train, test) {
  
  # Perform SMOTE
  set.seed(1234)
  smote_ <- smotefamily::SMOTE(X = train[,-31], target = train$Class, K = kSMOTE, dup_size = nSMOTE)
  training <- smote_$data
  training <- training[,-31]
  training$Class <- as.factor(training$Class)
  print("diag")	
  print(table(training$Class))
  
  ## Train the logistic regression algorithm on the training data using tidymodels
  
  # Define the logistic regression model
  log_reg <- logistic_reg(mode = "classification", engine = "glmnet", penalty = tune(), mixture = tune())
  
  # Define the grid search for the hyperparameters
  grid <- grid_regular(penalty(), mixture(), levels = c(penalty = 3, mixture = 4))
  
  # Define the workflow for the model
  log_reg_wf <- workflow() %>%
    add_model(log_reg) %>%
    add_formula(Class ~ .)
  
  # Define the resampling method for the grid search
  folds <- vfold_cv(data = training, v = 5)
  
  # Tune the hyperparameters using the grid search
  
  
  log_reg_tuned <- tune_grid(
    log_reg_wf,
    resamples = folds,
    grid = grid,
    control = control_grid(save_pred = TRUE)
  )
  
  # Select the best model based on the metric
  best_model <- select_best(log_reg_tuned, metric = "roc_auc")
  
  # Extract the best hyperparameters from the tibble object
  best_penalty <- best_model$penalty
  best_mixture <- best_model$mixture
  
  # Fit the model using the optimal hyperparameters
  log_reg_final <- logistic_reg(penalty = best_penalty, mixture = best_mixture) %>%
    set_engine("glmnet") %>%
    set_mode("classification") %>%
    fit(Class~., data = training)
  
  # Evaluate the model performance on the testing set
  pred_class <- predict(log_reg_final,
                        new_data = test,
                        type = "class")
  
  # Calculate AUC
  auc <- roc(test$Class, as.numeric(pred_class$.pred_class))
  
  ## Generate confusion matrix
  # Convert to factor
  model_pred <- as.factor(pred_class$.pred_class)
  model_true <- as.factor(test$Class)
  
  cm <- confusionMatrix(data = model_pred, reference =model_true)
  
  return(list( 
    "ROC" = NA,
    "AUC" = auc$auc,
    "ConfusionMatrix" = cm))
}
# Output: List mit auc, roc und confusion matrix


# Input: Hyperparameter von SMOTE, Train und Testdatensatz
run_asn_smote_with_logistic_regression <- function (kSMOTE = 5, nSMOTE = 577,  train_undersampled, train_not_undersampled, test){	
  
  # Perform SMOTE	
  set.seed(1234)	
  # ASN mit nicht undersampled trainingsdatensatz	
  asn_train <- asn_smote(train_not_undersampled, K = kSMOTE, dup_size = nSMOTE)	
  # print("asn_train")	
  # print(table(asn_train$Class))	
  
  # print("train_undersampled")	
  # print(table(train_undersampled$Class))	
  
  # Zusammenfügen von minority class aus ASN Ergebnis und majority class aus undersampling datensatz	
  training <- rbind(train_undersampled[train_undersampled$Class==0,], asn_train[asn_train$Class==1,])	
  # print("training")	
  # print(table(training$Class))	
  
  training$Class <- as.factor(training$Class)
  
  
  ## Train the logistic regression algorithm on the training data using tidymodels
  
  # Define the logistic regression model
  log_reg <- logistic_reg(mode = "classification", engine = "glmnet", penalty = tune(), mixture = tune())
  
  # Define the grid search for the hyperparameters
  grid <- grid_regular(penalty(), mixture(), levels = c(penalty = 3, mixture = 4))
  
  # Define the workflow for the model
  log_reg_wf <- workflow() %>%
    add_model(log_reg) %>%
    add_formula(Class ~ .)
  
  # Define the resampling method for the grid search
  folds <- vfold_cv(data = training, v = 5)
  
  # Tune the hyperparameters using the grid search
  
  
  log_reg_tuned <- tune_grid(
    log_reg_wf,
    resamples = folds,
    grid = grid,
    control = control_grid(save_pred = TRUE)
  )
  
  # Select the best model based on the metric
  best_model <- select_best(log_reg_tuned, metric = "roc_auc")
  
  # Extract the best hyperparameters from the tibble object
  best_penalty <- best_model$penalty
  best_mixture <- best_model$mixture
  
  # Fit the model using the optimal hyperparameters
  log_reg_final <- logistic_reg(penalty = best_penalty, mixture = best_mixture) %>%
    set_engine("glmnet") %>%
    set_mode("classification") %>%
    fit(Class~., data = training)
  
  # Evaluate the model performance on the testing set
  pred_class <- predict(log_reg_final,
                        new_data = test,
                        type = "class")
  
  # Calculate AUC
  auc <- roc(test$Class, as.numeric(pred_class$.pred_class))
  
  ## Generate confusion matrix
  # Convert to factor
  model_pred <- as.factor(pred_class$.pred_class)
  model_true <- as.factor(test$Class)
  
  cm <- confusionMatrix(data = model_pred, reference =model_true)
  
  return(list( 
    "ROC" = NA,
    "AUC" = auc$auc,
    "ConfusionMatrix" = cm))
}
# Output: List mit auc, roc und confusion matrix



################## FUNKTIONEN 3: Funktionen, die SMOTE&Classifier aufrufen ################## 

##### K-fold Cross-Validation:
## Einmal datensatz splitten
## schleife von 1:5, jedes mal ein anderes Subset als Testdatensatz wählen
## in der Schleife: Aktuellen Test und Train-Datensatz in Model geben & AOC speichern
## Nach durchlauf der Schlaufe: Durchschnitt von AOC bilden & returnen

# Input: gewünschte Algorithmen, Anzahl der gewünschten Subsets für Crossvalidation, Datensatz, Parameter für SMOTE
kFoldCrossValidate <- function(oversampling_algorithm, classifier, kForFold, entireDataset, undersampling_factor, kForSMOTE, nDesiredRatioOfClassesForSMOTE){
  subsets <- splitIntoKSubsets(K = kForFold, dataset = entireDataset)
  vectorOfAOCresults <- c()
  for (i in 1:length(subsets)){
    inputDataForModelRun <- mergeSubsetsIntoTrainAndTest(whichSubsetAsTest = i, subsetList = subsets)
    
    # UNDERSAMPLING : 
    #print(undersampling_factor)
    #print(table(inputDataForModelRun$train$Class))
    undersampled_train <- undersample(undersampling_factor, inputDataForModelRun$train)
    #print(table(inputDataForModelRun$train$Class))
    
    
    
    # da in jedem subset ein leicht anderes anderes Verhältnis von min zu maj
    # klasse vorliegt, sollte der N parameter für jedes subset bestimmt werden.
    # nDesiredRatioOfClassesForSMOTE von 100% (= 1) bedeutet dabei, 
    # dass nach SMOTE gleich viele minority wie 
    # majority samples im Datensatz sein sollen (100% bedeutet also gleich viele,
    # das sollte wieder einen absoluten n Wert von ca. 577 ergeben)
    # Das Hyperparametertuning erfolgt somit später mit prozentwerten statt
    # absoluten werten. z.B. 120% (=1.2) würde also bedeuten, wir wollen
    # 20% mehr minority als majority samples nach SMOTE
    nAbsoluteSMOTE <- table(undersampled_train$Class)[1]/table(undersampled_train$Class)[2]
    nAbsoluteSMOTE <- round((nAbsoluteSMOTE-1)*nDesiredRatioOfClassesForSMOTE)
    
    
    resultsOfSingleRun <- list()
    
    if (classifier == "xgboost"){
      if (oversampling_algorithm == "smote"){
        cat("run_smote_with_xgboost - with N = ",nAbsoluteSMOTE)
        resultsOfSingleRun <- run_smote_with_xgboost(kSMOTE = kForSMOTE, nSMOTE = nAbsoluteSMOTE, train = undersampled_train, test = inputDataForModelRun$test)
      }
      else if (oversampling_algorithm == "asn"){
        cat("run_asn_smote_with_xgboost - with N = ",nAbsoluteSMOTE)
        resultsOfSingleRun <- run_asn_smote_with_xgboost(kSMOTE = kForSMOTE, nSMOTE = nAbsoluteSMOTE,  train_undersampled = undersampled_train, train_not_undersampled = inputDataForModelRun$train, test = inputDataForModelRun$test)
      }
      else {cat("FEHLER - FEHLER - FEHLER - FEHLER - FEHLER")}
    }
    else if (classifier == "logistic_regression"){
      if (oversampling_algorithm == "smote"){
        cat("run_smote_with_logistic_regression - with N = ",nAbsoluteSMOTE)
        resultsOfSingleRun <- run_smote_with_logistic_regression(kSMOTE = kForSMOTE, nSMOTE = nAbsoluteSMOTE, train = undersampled_train, test = inputDataForModelRun$test)
      }
      else if (oversampling_algorithm == "asn"){
        cat("run_asn_smote_with_logistic_regression - with N = ",nAbsoluteSMOTE)
        resultsOfSingleRun <- run_asn_smote_with_logistic_regression(kSMOTE = kForSMOTE, nSMOTE = nAbsoluteSMOTE, train_undersampled = undersampled_train, train_not_undersampled = inputDataForModelRun$train, test = inputDataForModelRun$test)
      }
      else {cat("FEHLER - FEHLER - FEHLER - FEHLER - FEHLER")}
    }
    else {cat("FEHLER - FEHLER - FEHLER - FEHLER - FEHLER")}
    
    
    vectorOfAOCresults <- c(vectorOfAOCresults, resultsOfSingleRun$AUC)
  }

  return(list(
    "AUCResultsOfSingleRuns" = vectorOfAOCresults, 
    "AUCAverage" = mean(vectorOfAOCresults),
    "oversampling_algorithm" = oversampling_algorithm, 
    "classifier" = classifier
  ))
}
# Output: verwendete Algorithmen, List mit Durchschnitts-AUC, Vektor der AUCs der einzelnen Durchläufe und dem verwendeten Classifier und oversampling algorithmus


##### Hyperparameteroptimierung:


# Input: zu nutzende Algorithmen, kFold (für crossvalidation), der Datensatz, der auszuprobierende Wertebereich
#       für k (SMOTE) und n (SMOTE, siehe auch Dokumentation von kFoldCrossValidate),
#       gewünschtes Undersampling (0 = kein Undersampling)
tryRandomHyperparameter <- function (oversampling_algorithm, classifier, kForFold, entireDataset, kForSMOTE, nDesiredRatioSMOTERange, undersamplingFactorRange){
  kToTest <- numeric()
  if (length(kForSMOTE)==1){
    kToTest <- kForSMOTE[1]
  } else {
    kToTest <- sample(kForSMOTE,1)
  }
  
  nRatioToTest <- numeric()
  if (length(nDesiredRatioSMOTERange)==1){
    nRatioToTest <- nDesiredRatioSMOTERange[1]
  } else {
    nRatioToTest <- sample(nDesiredRatioSMOTERange,1)
  }

  undersamplingFactorToTest <- numeric()
  if (length(undersamplingFactorRange)==1){
    undersamplingFactorToTest <- undersamplingFactorRange[1]
  } else {
    undersamplingFactorToTest <- sample(undersamplingFactorRange,1)
  }


  cat("---------------------------- getestete Parameter: nRatio=", nRatioToTest, ", kSMOTE=", kToTest,", undersamplingFactor=", undersamplingFactorToTest,  " ----------------------------")
  
  
  crossValidatedAUC <- kFoldCrossValidate(oversampling_algorithm, classifier, kForFold, entireDataset, undersamplingFactorToTest, kForSMOTE = kToTest, nDesiredRatioOfClassesForSMOTE = nRatioToTest)
  
  resultVector <- c(crossValidatedAUC$oversampling_algorithm, crossValidatedAUC$classifier, kForFold, undersamplingFactorToTest, kToTest, nRatioToTest, crossValidatedAUC$AUCAverage)
  return(resultVector)
}
# Output: Vector der Testergebnisse mit: genutzte algorithmen, kFold (für crossvalidation), 
#         kSMOTE, nRatioSMOTE und die AUC

























################## DURCHLAUF ################## 
## Die Durchschnittsperformance aus der Cross Validation sollte dauerhaft
## abgespeichert werden, da das Berechnen dieser Kombinationen extrem rechen-
## intensiv ist. D.h. sollen alle Parameter in hyperparameterTuning.csv 
## gespeichert werden

testedHyperparameters <- read.csv("hyperparameter_tuning_V3.csv")[,-1]
colnames(testedHyperparameters) <- c("oversampling_altorithm", "classifier", "kFold", "undersamplingFactor", "kSMOTE", "nRatioSMOTE", "AUCAverage")

for (i in 1:500){
  newRandomHyperparameterTestResult <- tryRandomHyperparameter(
    oversampling_algorithm = sample(c("smote", "asn"),1), 
    classifier = sample(c("xgboost", "logistic_regression"),1), 
    kForFold = 10,
    entireDataset = ccdata,
    kForSMOTE = c(2:10,12,14,16,18,20,22,24,30,35,40,45,50,55,60,65,70),
    nDesiredRatioSMOTERange = 1.0,
    undersamplingFactor = c(0.75, 0.80, 0.85, 0.90, 0.95, 0.99)
  )
  
  testedHyperparameters <- rbind(testedHyperparameters, newRandomHyperparameterTestResult)
  cat("---------------------------- Abgeschlossener Durchlauf: ", i, " ----------------------------")
  
  write.csv(testedHyperparameters, "hyperparameter_tuning_V3.csv", row.names=TRUE)
}





################## PLOTTEN ################## 

## Daten plotten - Variante 1 (nSMOTE x kSMOTE plotten): 


dataToPlot <- subset(testedHyperparameters, as.logical((
  testedHyperparameters$nRatioSMOTE > 0.0) * 
    (testedHyperparameters$kSMOTE > 0) * 
    (testedHyperparameters$kFold==10) *
    (testedHyperparameters$undersamplingFactor == 0)
))

# Convert nRATIO column from character to numeric
dataToPlot$nRatioSMOTE <- as.numeric(gsub(",", ".", dataToPlot$nRatioSMOTE))

# Create a scatter plot - Variant 1 
ggplot(dataToPlot, aes(x = kSMOTE, y = nRatioSMOTE, color = AUCAverage)) +
  geom_point(size = 5) +
  scale_color_gradient(low = "red", high = "blue") +
  labs(x = "kSMOTE", y = "nRatioSMOTE", color = "AUCAverage") +
  theme_minimal()


## Daten plotten - Variante 2 (Undersampling x kSMOTE plotten): 


dataToPlot <- subset(testedHyperparameters, as.logical((
  testedHyperparameters$nRatioSMOTE == 1) * 
    (testedHyperparameters$kSMOTE < 200) * 
    (testedHyperparameters$kFold==10) *
    (testedHyperparameters$undersamplingFactor >= 0)
))

# Convert nRATIO column from character to numeric
dataToPlot$nRatioSMOTE <- as.numeric(gsub(",", ".", dataToPlot$nRatioSMOTE))

# Create a scatter plot 
ggplot(dataToPlot, aes(x = kSMOTE, y = undersamplingFactor, color = AUCAverage)) +
  geom_point(size = 5) +
  scale_color_gradient(low = "red", high = "blue") +
  labs(x = "kSMOTE", y = "undersamplingFactor", color = "AUCAverage") +
  theme_minimal()



## Daten plotten - Variante 3 (Undersampling x nSMOTE plotten): 


dataToPlot <- subset(testedHyperparameters, as.logical((
  testedHyperparameters$nRatioSMOTE > 0.2) * 
    (testedHyperparameters$kSMOTE < 55) * 
    (testedHyperparameters$kFold==10) *
    (testedHyperparameters$undersamplingFactor >= 0)
))

# Convert nRATIO column from character to numeric
dataToPlot$nRatioSMOTE <- as.numeric(gsub(",", ".", dataToPlot$nRatioSMOTE))

# Create a scatter plot 
ggplot(dataToPlot, aes(x = nRatioSMOTE, y = undersamplingFactor, color = AUCAverage)) +
  geom_point(size = 5) +
  scale_color_gradient(low = "red", high = "blue") +
  labs(x = "nRatioSMOTE", y = "undersamplingFactor", color = "AUCAverage") +
  theme_minimal()


