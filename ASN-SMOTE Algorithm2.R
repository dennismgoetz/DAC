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


k <- 5
n <- 10
#asn_smote <- function(data, train_feat, train_target, n, k) {  # !!! Minority instance = 1
  
  train_feat_matrix <- as.matrix(train_feat)
  train_Majority <- train[train_target == 0,]
  train_Minority <- train[train_target == 1,]
  
  ####### [1:29] muss man noch ändern sodass man Function für andere Datensätze replizieren kann
  train_Minority_feat <- train_Minority[,1:29]   #Features of Minority set (= P in the Pseudo code)
  
  # Algorithm 1: Noise filtering
  dis_matrix <- proxy::dist(train_Minority_feat, train_feat)
  
  
  ##########################################################################################################
 
  min_index1 <- list()
  for (i in 1:nrow(train_Minority_feat)) {
    min_index1[[rownames(dis_matrix)[i]]] <- order(dis_matrix[i,])[2:(k+1)]
   
  }
  
  for (i in 1:nrow(train_Minority_feat)) {
    for (j in 1:k) {
      if (train_target[min_index1[[i]][j]] == 0 )
       min_index1[[i]][j] <- NaN
    }
  }
  
  Mu <- vector()
  for (i in length(min_index1):1) {
    if (is.nan(min_index1[[i]][1])) {
      Mu[i] <- names(min_index1[i])
      min_index1 <- min_index1[-i]
    }
  }
  
  Mu <- na.omit(Mu) 
  Mu <- Mu[1:length(Mu)]

  for (i in 1:nrow(train_Minority_feat)) {
    for (j in 1:k) {
      if (is.nan(min_index1[[i]][j])) {
        min_index1[[i]] <- min_index1[[i]][1:(j-1)]
        break
      }
    }
  }

#  duplicates_list <- list()
#  for (i in 1:length(min_index1)) {
#    
#    duplicates <- duplicated(min_index1[[i]])
#    
#    if (any(duplicates)) {
#      duplicates_list[[i]] <- min_index1[[i]][duplicates]
#    }
#    
#  }
#  duplicates_list
  
  synthetic <- list()
  for(i in names(min_index1)) {
    for(j in seq_len(n)) {
      nn <- sample(seq_along(min_index1[[i]]), 1)   # random number in the length of the best index
      dif <- train_feat_matrix[min_index1[[i]][nn],] - train_feat_matrix[i,]  ## dif von der dis matrix
      gap <- runif(1)
      synthetic_instance <- train_feat_matrix[i,] + gap * dif
      synthetic[[length(synthetic) + 1]] <- synthetic_instance
    }
  }
 

  ###########################################################################################################

  # Combine the synthetic instances and their labels
  synthetic_df <- do.call(rbind, synthetic)
  synthetic_df <- as.data.frame(synthetic_df)
  synthetic_labels <- rep(1, length(synthetic))
  synthetic_df$Class <- synthetic_labels

  # Combine majority with new minority
  asn_train <- rbind(train, synthetic_df)

  # remove unqualified points
  asn_train <<- asn_train[!(rownames(asn_train) %in% Mu), ]
  
  return(paste0("The ASN SMOTE was applied to the data. The new training dataset is saved as asn_train."))
#}


asn_smote(train, train_feat,train_target,10,5)




table(asn_train$Class)
