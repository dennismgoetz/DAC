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
ccdata <- read.csv("creditcard.csv")

# Split into training/test set
set.seed(123)
split <- sample.split(ccdata$Class, SplitRatio = 0.9)
train <- subset(ccdata, split == TRUE)
test <- subset(ccdata, split == FALSE)

# Drop column 'Time' and scale column 'Amount'
train <- train[,-1] %>% mutate(Amount = scale(Amount))
test <- test[,-1] %>% mutate(Amount = scale(Amount))

# ASN-SMOTE function
asn_smote <- function(train, n, k) {
  
  # Split the train set into features (T in the Pseudo Code) and target value
  train_feat <- train[,1:29] 
  train_target <- train$Class
  
  # Create a matrix with the features and split the train set into majority and minority
  train_feat_matrix <- as.matrix(train_feat)
  train_Majority <- train[train_target == 0,]
  train_Minority <- train[train_target == 1,]
  
  # Select only the features of the minority train set (P in the Pseudo code)
  train_Minority_feat <- train_Minority[,1:29]

  # Calculate the distance of each minority instance to all samples of the train set
  dis_matrix <- proxy::dist(train_Minority_feat, train_feat)
  
  # Create a list with indices of the k-nearest minority neighbors of all minority instances 
  # (majority neighbors marked as NaN)
  index_knn <- list()
  
  for (i in 1:nrow(train_Minority_feat)) {
    index_knn[[rownames(dis_matrix)[i]]] <- order(dis_matrix[i,])[2:(k+1)]
    for (j in 1:k) {
      if (train_target[index_knn[[i]][j]] == 0 ) {
        index_knn[[i]][j] <- NaN
      }
    }
  }
  
  print("Distance matrix calculated and nearest neighbors defined.")
  print("--------------------------------------------------------------------------------")
  
  
  
  # Algorithm 1: Filter Noise
  # Drop minority instances with a majority (NaN) as nearest neighbor
  Mu <- vector()
  
  for (i in length(index_knn):1) { 
    if (is.nan(index_knn[[i]][1])) {
      Mu[i] <- names(index_knn[i])
      index_knn <- index_knn[-i]
    }
  }
  
  print(paste0("Number of qulaified minority instances: ", length(index_knn), 
               " of ", nrow(train_Minority)))
  print("Algorithm 1 successfully completed.")
  print("--------------------------------------------------------------------------------")
  
  
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

  print(paste0("Mean qualified nearest neighbors: ", 
               round(sum(lengths(index_knn))/length(index_knn), 2), " of ", k))
  print("Algorithm 2 successfully completed.")
  print("--------------------------------------------------------------------------------")
  
  
  # Algorithm 3: Procedure of ASN-SMOTE (Create new synthetic minority samples)
  # Add to the feature values of each qualified minority instance the difference of the minority sample and one
  # random selected neighbor of their qualified neighbors multiplied with a random number between 0 and 1 for n times.
  synthetic <- list()
  for(i in names(index_knn)) {
    for(j in seq_len(n)) {
      random_n <- sample(seq_along(index_knn[[i]]), 1)
      dif <- train_feat_matrix[index_knn[[i]][random_n],] - train_feat_matrix[i,]
      randomNum <- runif(1)
      synthetic_instance <- train_feat_matrix[i,] + randomNum * dif
      synthetic[[length(synthetic) + 1]] <- synthetic_instance
    }
  }
  
  print(paste0("Number of generated synthetic minority samples: ", length(synthetic)))
  print("Algorithm 3 successfully completed.")
  print("--------------------------------------------------------------------------------")
  
  
  
  # Assign "Class" label = 1 to the synthtic points
  synthetic_df <- do.call(rbind, synthetic)
  synthetic_df <- as.data.frame(synthetic_df)
  synthetic_labels <- rep(1, length(synthetic))
  synthetic_df$Class <- synthetic_labels
  
  # Combine original train set with synthetic set
  asn_train <<- rbind(train, synthetic_df)
  print("The ASN-SMOTE was applied to the data.")
  
  return (print("The new training dataset is saved as 'asn_train'."))
}


# Execute ASN-SMOTE function
asn_smote(train, n=10, k=5)

# Balanced dataset
asn_smote(train, n=700, k=10)

# View the new balance of the dataset
table(asn_train$Class)

# data visualization after ASN-SMOTE
ggplot(asn_train, aes(x = V1, y = V2, color = factor(class))) + geom_point() + ggtitle("Class distribution after ASN-SMOTE")+ scale_color_manual(values = c("#E69F00", "#56B4E9"))
