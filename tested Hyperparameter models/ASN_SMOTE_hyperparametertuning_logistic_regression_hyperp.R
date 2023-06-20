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



# Load the data
#setwd("C:/Users/noel2/OneDrive/Studium Workspace/M.Sc. Betriebswirtschaftslehre/BAOR_Data Analytics Challange/DAC Shared Workspace/R Workspace")
#setwd("C:/Users/Dennis/OneDrive/Dokumente/03_Master BAOR/05_Kurse/01_Business Analytics/04_Data Analytics Challenge/")
setwd("C:/Users/Vincent Bl/Desktop/DAC/")
ccdata <- read.csv("creditcard.csv")

# Preprocessing
ccdata <- ccdata[,-1]
ccdata[,-30] <- scale(ccdata[,-30])



################## FUNKTIONEN ################## 


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
undersample <- function (undersamplingFactor = 1.00, dataset = ccdata){
  # TRUE im SelectionVector heißt weiter benutzen, FALSE heißt verwerfen
  selectionVector <- dataset$Class==1
  selectionVector <- as.logical(
    selectionVector
    +
    sample(c(0,1),length(dataset[,1]), prob = c(undersamplingFactor,1-undersamplingFactor), replace = TRUE)
      )
  
  undersampledDataset <- dataset[selectionVector,]
  return (undersampledDataset)
}
# Output: Trainingsdatensatz nach Undersampling



# ASN SMOTE
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
    index_knn[[rownames(dis_matrix)[i]]] <- order(dis_matrix[i,])[2:(k+1)]
    for (j in 1:k) {
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
    for (j in 1:k) {
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
    for(j in seq_len(n)) {
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


##### Single run of Model (SMOTE and ML algorithm):
parallelStartSocket(detectCores())

# Input: Hyperparameter von SMOTE, Train und Testdatensatz
singleModelRun <- function (kSMOTE = 5, nSMOTE = 577, train, test) {
  

  # Perform SMOTE
  set.seed(1234)
  
  asn_smote <- asn_smote(train, K = kSMOTE, dup_size = nSMOTE)
  training <- asn_smote
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
    "AUC" = auc$auc, 
    "ConfusionMatrix" = cm,
    "Penalty" = best_penalty,
    "Mixture" = best_mixture))
}
# Output: List mit auc, roc und confusion matrix


##### K-fold Cross-Validation:
## Einmal datensatz splitten
## schleife von 1:5, jedes mal ein anderes Subset als Testdatensatz wählen
## in der Schleife: Aktuellen Test und Train-Datensatz in Model geben & AOC speichern
## Nach durchlauf der Schlaufe: Durchschnitt von AOC bilden & returnen

# Input: Anzahl der gewünschten Subsets für Crossvalidation, Datensatz, Parameter für SMOTE
kFoldCrossValidate <- function(kForFold = 4, 
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
    "AUCAverage" = mean(vectorOfAOCresults),
    "Hyper_pen" = resultsOfSingleRun$Penalty,
    "Hyper_mix" = resultsOfSingleRun$Mixture
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
#       gewünschtes Undersampling (0 = kein Undersampling)
tryRandomHyperparameter <- function (kForFold,
                                     entireDataset,
                                     kForSMOTERange,
                                     nDesiredRatioSMOTERange,
                                     undersamplingFactorRange
                                     ){
  kToTest <- numeric()
  if (length(kForSMOTERange)==1){
    kToTest <- kForSMOTERange[1]
  } else {
    kToTest <- sample(kForSMOTERange,1)
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
  
  undersampledDataset <- undersample(undersamplingFactorToTest, entireDataset)
  
  crossValidatedAUC <- kFoldCrossValidate(kForFold, undersampledDataset, kForSMOTE = kToTest, nDesiredRatioOfClassesForSMOTE = nRatioToTest)
  
  resultVector <- c(kForFold, undersamplingFactorToTest, kToTest, nRatioToTest, crossValidatedAUC$AUCAverage, crossValidatedAUC$Hyper_pen, crossValidatedAUC$Hyper_mix)

  
  return(resultVector)
}
# Output: Vector der Testergebnisse mit: kFold (für crossvalidation), 
#         kSMOTE, nRatioSMOTE (siehe Dokumentation von kFoldCrossValidate) und die AUC



  

################## DURCHLAUF ################## 
## Die Durchschnittsperformance aus der Cross Validation sollte dauerhaft
## abgespeichert werden, da das Berechnen dieser Kombinationen extrem rechen-
## intensiv ist. D.h. sollen alle Parameter in hyperparameterTuning.csv 
## gespeichert werden



#testedHyperparameters <- read.csv("testedHyperparameters_logreg2.csv")[,-1]
col_names <- c("kFold", "undersamplingFactor", "kSMOTE", "nRatioSMOTE", "AUCAverage", "Hyperp. penalty", "Hyperp. mixture")
hyperparameter_results <- data.frame(matrix(ncol= length(col_names),nrow=0))
colnames(hyperparameter_results) <- col_names

for (i in 1:500){
  newRandomHyperparameterTestResult <- tryRandomHyperparameter(
    kForFold = 10,
    entireDataset = ccdata,
    kForSMOTERange = c(2:10,12,14,16,18,20,22,24,30,35,40,45,50,55,60,65,70), # HIER GGF. PARAMETER ANPASSEN
    nDesiredRatioSMOTERange = c(0.9, 1.0, 1.1), # HIER GGF. PARAMETER ANPASSEN
    undersamplingFactor = c(0.75, 0.80, 0.85, 0.90, 0.95, 0.99)  # HIER GGF. PARAMETER ANPASSEN  
  )
  
  newRandomHyperparameterTestResult <- c(newRandomHyperparameterTestResult)
  hyperparameter_results <<- rbind(hyperparameter_results, newRandomHyperparameterTestResult)
  cat("---------------------------- Abgeschlossener Durchlauf: ", i, " ----------------------------")
  
  #write.csv(hyperparameter_results, "testedHyperparameters_logreg2.csv", row.names=TRUE)
}
hyperparameter_results

parallelStop()


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












