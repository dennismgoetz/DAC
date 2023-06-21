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
ccdata <- ccdata[1:10000,]

### Preprocessing

#Remove column time and scale column amount 
ccdata <- ccdata[,-1]

ccdata[,-30] <- scale(ccdata[,-30])

### Split into training/test set
set.seed(123)
split <- sample.split(ccdata$Class, SplitRatio = 0.7)
train <-  subset(ccdata, split == TRUE)
test <- subset(ccdata, split == FALSE)
table(train$Class)


###### ASN SMOTE  ##########

### Algorithm 1: Noise filtering step
samples_X <- train[,1:29] #Features
samples_Y <- train$Class #Target value
All_X <- as.matrix(samples_X)



Majority_train <- train[train$Class == 0,]

Minority_sample <- train[train$Class == 1,] #Minority set
Minority_sample_X <- Minority_sample[,1:29] #Features of Minority set

P <- Minority_sample_X
T <- samples_X

print(Minority_sample)
print(nrow(Minority_sample))



### Algorithm 1
Mu <- vector()  # Set of unqualified minority instances
Mq <- vector()  # Set of qualified minority instances
dis_matrix <- matrix(0, nrow = nrow(P), ncol = nrow(T))


for (i in 1:nrow(P)) {
  nearest_distance <- Inf
  nearest_p <- NULL
  
  #  Calculate the Euclidean distance to each instance in T
  for (j in 1:nrow(T)) {
    if (all(P[i,] != T[j,])){
     distances <- sqrt(sum((P[i,] - T[j,])^2))
     dis_matrix[i, j] <- distances
     
     if (all(P[i,] == T[j,])) {
       dis_matrix[i, j] <- 99999
     }
     # Check if the nearest instance is in the minority class
      if (distances < nearest_distance) {
        nearest_distance <- distances
        index <- j
      }
    }
  }
  
  if (train[index,]$Class == 0) {
    # unqualified minority instance
    Mu <- rbind(Mu, P[i, ])
  } else {
    # qualified minority instance
    Mq <- rbind(Mq, P[i, ])
  }
}




# Algorithm 2 + 3

# Parameters for SMOTE
n <- 10
k <- 5

synthetic <- list()


for(i in seq_len(nrow(Mq))) {
  min_index <- order(dis_matrix[i,])[1:k]
  best_index <- integer(0)
  best_f <- 1
  for(h in min_index) {
    if(samples_Y[h] == 0) {
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
    dif <- All_X[best_index[nn],] - Mq[i,]
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
samples <- Majority_train


# Combine majority with new minority
examples <- rbind(samples, syntheticMq)
table(examples$Class)




