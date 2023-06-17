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
setwd("C:/Users/Vincent Bl/Desktop/DAC/")
ccdata <- read.csv("creditcard.csv")#[1:10000,]



# Split into training/test set
set.seed(123)
split <- sample.split(ccdata$Class, SplitRatio = 0.7)
train <- subset(ccdata, split == TRUE)
test <- subset(ccdata, split == FALSE)

train <- train[,-1] %>% mutate_at(vars(-Class), scale)
test <- test[,-1] %>% mutate_at(vars(-Class), scale)



### Algorithm 1: Noise filtering step
train_feat <- train[,1:29] #Features of train  (= T in the Pseudo Code)

train_target <- train$Class #Target value of train




#asn_smote <- function(data, train_feat, train_target, n, k) {
  
  train_feat_matrix <- as.matrix(train_feat)
  train_Majority <- train[train_target == 0,]
  train_Minority <- train[train_target == 1,]
  
  ####### [1:29] muss man noch ändern sodass man Function für andere Datensätze replizieren kann
  train_Minority_feat <- train_Minority[,1:29]   #Features of Minority set (= P in the Pseudo code)
  
  # Algorithm 1: Noise filtering
  dis_matrix <- proxy::dist(train_Minority_feat, train_feat)
  
  
  ##########################################################################################################
  k <- 5
  n <- 10
  
  
  index_knn <- list()
  
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
  
  
  for (i in 1:nrow(train_Minority_feat)) {
    if (i <= length(index_knn)) {
      for (j in 1:k) {
        if (is.nan(index_knn[[i]][j])) {
          index_knn[[i]] <- index_knn[[i]][1:(j-1)]
          break
        }
      }
    }
  }
  
  
  
#  duplicates_list <- list()
#  for (i in 1:length(index_knn)) {
#    
#    duplicates <- duplicated(index_knn[[i]])
#    
#    if (any(duplicates)) {
#      duplicates_list[[i]] <- index_knn[[i]][duplicates]
#    }
#    
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
  asn_train <<- asn_train[!(rownames(asn_train) %in% Mu), ]
  
  return(paste0("The ASN SMOTE was applied to the data. The new training dataset is saved as asn_train."))
#}


asn_smote(train, train_feat,train_target,10,5)




table(asn_train$Class)
