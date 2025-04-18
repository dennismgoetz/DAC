# Load necessary packages
library(tidyverse)
library(randomForest)
library(caTools)
library(smotefamily)
library(caret)
library(mlr)
library(tibble)
library(corrplot)


# Load and preprocess data
#setwd("C:/Users/Vincent Bl/Desktop/DAC/")
setwd("C:/Users/Dennis/OneDrive/Dokumente/03_Master/05_Kurse/01_BA/04_DAC/")
ccdata <- read.csv("creditcard.csv")#[1:10000,]

# Split into training/test set
set.seed(123)
split <- sample.split(ccdata$Class, SplitRatio = 0.9)
train <- subset(ccdata, split == TRUE)
test <- subset(ccdata, split == FALSE)

# Drop column 'Time' and scale column 'Amount'
train <- train[,-1] %>% mutate_at(vars(-Class), scale)
test <- test[,-1] %>% mutate_at(vars(-Class), scale)



# ASN-SMOTE Function
asn_smote <- function(train, n, k) {
  
  # Split the train set into features (= T in the Pseudo Code) and target value
  train_feat <- train[,1:29] 
  train_target <- train$Class
  
  # Create a matrix with the features and split the train set into majority and minority
  train_feat_matrix <- as.matrix(train_feat)
  train_Majority <- train[train_target == 0,]
  train_Minority <- train[train_target == 1,]
  
  # #Features of Minority set (= P in the Pseudo code)
  train_Minority_feat <- train_Minority[,1:29]
  
  # Calculate the distance of each minority instance to all samples of the train set
  dis_matrix <- proxy::dist(train_Minority_feat, train_feat)
  
  
  ##########################################################################################################
  k <- 5
  n <- 10
  
  
  # Tests
  dis_matrix[1,]
  order(dis_matrix[1,])
  order(dis_matrix[1,])[1:6]
  dis_matrix[1, 490]
  dis_matrix[1, 71985]
  dis_matrix[1, 24231]
  
  train[490,]
  train_Minority_feat[1,]
  
  sum(train_Minority_feat[1,] - train_feat[490,]) #sollte 0 sein (passt)
  sum(train_Minority_feat[1,] - train_feat["24231",]) #sollte 4.352906 sein (passt nicht)
  sum(abs(train_Minority_feat[1,] - train_feat[24231,]))#/29
  rownames(dis_matrix)[1]
  
  
  # Create a list with indices of the k-nearest minority neighbors of all minority instances (majority neighbors marked as NaN)
  index_knn <- list()
  
  for (i in 1:nrow(train_Minority_feat)) {
    index_knn[[rownames(dis_matrix)[i]]] <- order(dis_matrix[i,])[2:(k+1)]
    for (j in 1:k) {
      if (train_target[index_knn[[i]][j]] == 0 ) {
        index_knn[[i]][j] <- NaN
      }
    }
  }
  
  
  # Algorithm 1: Filter Noise
  # Drop minority instances with a majority (NaN) as nearest neighbor
  Mu <- vector()
  
  for (i in length(index_knn):1) { 
    if (is.nan(index_knn[[i]][1])) {
      Mu[i] <- names(index_knn[i])
      index_knn <- index_knn[-i]
    }
  }
  
  
  #Mu <- na.omit(Mu) #not needed
  #Mu <- Mu[1:length(Mu)]
  
  # Algorithm 2: Adaptive neighbor instances selection
  # Keep only the neighbors that are closer than the nearest majority (NaN) instance
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
  
  # Algorithm 3: Procedure of ASN-SMOTE (Create new synthetic minority samples)
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
  
  # Assign "Class" label = 1 to the synthtic points
  synthetic_df <- do.call(rbind, synthetic)
  synthetic_df <- as.data.frame(synthetic_df)
  synthetic_labels <- rep(1, length(synthetic))
  synthetic_df$Class <- synthetic_labels
  
  # Combine original train set with synthetic set
  asn_train <<- rbind(train, synthetic_df)
  
  (paste0("The ASN SMOTE was applied to the data. The new training dataset is saved as asn_train."))
  
  return (asn_train)
}

# Execute ASN-SMOTE function
asn_smote(train, n=10, k=5)

# View the new balance of the dataset
table(asn_train$Class)
