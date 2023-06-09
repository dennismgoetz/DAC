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
ccdata <- read.csv("creditcard.csv")[1:10000,]
ccdata <- ccdata[,-1] %>% mutate_at(vars(-Class), scale)


# Split into training/test set
set.seed(123)
split <- sample.split(ccdata$Class, SplitRatio = 0.7)
train <- subset(ccdata, split == TRUE)
test <- subset(ccdata, split == FALSE)



### Algorithm 1: Noise filtering step
train_feat <- train[,1:29] #Features of train  (= T in the Pseudo Code)

train_target <- train$Class #Target value of train





asn_smote <- function(data, train_feat, train_target, n, k) {  # !!! Minority instance = 1
  
  train_feat_matrix <- as.matrix(train_feat)
  train_Majority <- data[train_target == 0,]
  train_Minority <- data[train_target == 1,]
  
  ####### [1:29] muss man noch ändern sodass man Function für andere Datensätze replizieren kann
  train_Minority_feat <- train_Minority[,1:29]   #Features of Minority set (= P in the Pseudo code)
  
  # Algorithm 1: Noise filtering
  dis_matrix <- proxy::dist(train_Minority_feat, train_feat)
  #dis_df <- as.data.frame.matrix(dis_matrix)
  
  
  Mu <- vector()  # Set of unqualified minority instances
  Mq <- vector()  # Set of qualified minority instances
  for (i in 1:nrow(train_Minority_feat)) {
    min_index <- order(dis_matrix[i,])[2]
    if (data[min_index,]$Class == 0) {
      # unqualified minority instance
      Mu <- rbind(Mu, train_Minority_feat[i, ])
    } else {
      # qualified minority instance
      Mq <- rbind(Mq, train_Minority_feat[i, ])
    }
  }
  #nrow(train_Minority_feat)
  
  #nrow(Mu)
  #nrow(Mq)
  
  synthetic <- list()
  
  for(i in seq_len(nrow(Mq))) {
    min_index <- order(dis_matrix[i,])[2:(k+1)]
    best_index <- vector()
    best_f <- 1
    for(h in min_index) {
      if(train_target[h] == 0) {
        best_index[best_f] <- h
        best_f <- best_f + 1
        break
      } else {
        best_index[best_f] <- h
        best_f <- best_f + 1
      }
    }
    
    
    # Create new synthetic minority samples
    for(j in seq_len(n)) {
      nn <- sample(seq_along(best_index), 1)
      dif <- train_feat_matrix[best_index[nn],] - Mq[i,]
      gap <- runif(1)
      synthetic_instance <- Mq[i, ] + gap * dif
      synthetic[[length(synthetic) + 1]] <- synthetic_instance
    }
  }
  
  
  # Combine all data frames in the synthetic list into a single data frame
  synthetic_df <- do.call(rbind, synthetic)
  
  
  # Combine the synthetic instances and their labels
  synthetic_labels <- rep(1, length(synthetic))
  Mq_labels <- rep(1,nrow(Mq))
  synthetic_df$Class <- synthetic_labels
  
  # Combine qualified instances with synthetic instances
  Mq$Class <- Mq_labels
  syntheticMq <- rbind(Mq, synthetic_df)
  samples <- train_Majority
  
  
  # Combine majority with new minority
  asn_train <<- rbind(samples, syntheticMq)
  return(paste0("The ASN SMOTE was applied to the data. The new training dataset is saved as asn_train."))
}


asn_smote(train, train_feat,train_target,10,5)




table(asn_train$Class)
