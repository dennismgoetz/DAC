library(tidyverse)
library(randomForest)
library(caTools)
library(smotefamily)
library(caret)
library(mlr)
library(tibble)
library(corrplot)

# Load the data
setwd("C:/Users/Dennis/OneDrive/Dokumente/03_Master BAOR/05_Kurse/01_Business Analytics/04_Data Analytics Challenge/05_Scripts")
ccdata <- read.csv("creditcard.csv")
#ccdata <- ccdata[1:50000,]

### Preprocessing

#Remove column time and scale column amount 
ccdata <- ccdata[,-1]

ccdata[,-30] <- scale(ccdata[,-30])

### Split into training/test set
set.seed(123)
split <- sample.split(ccdata$Class, SplitRatio = 0.01)
train <-  subset(ccdata, split == TRUE)
test <- subset(ccdata, split == FALSE)
train
table(train$Class)

#Minority_X <- c(Minority_sample_X)

###### ASN SMOTE  ##########

### Algorithm 1: Noise filtering step
samples_X <- train[,1:29] #Features
samples_Y <- train$Class #Target value

Minority_sample <- train[train$Class == 1,] #Minority set
Minority_sample_X <- Minority_sample[,1:29] #Features of Minority set

P <- Minority_sample_X
T <- samples_X

print(Minority_sample)
print(nrow(Minority_sample))

# Mu: set of unqualified minority instances
# Mq: set of qualified minority instances


############################ muss man bei nearest neighbor berechnung die class raus nehmen???
# Step 1: Initialization
Mu <- vector()  # Set of unqualified minority instances
Mq <- vector()  # Set of qualified minority instances

# Step 2: Iterate over each instance in the minority class
for (i in 1:nrow(P)) {
  nearest_distance <- Inf
  nearest_p <- NULL
  
  # Step 3: Calculate the Euclidean distance to each instance in T
  for (j in 1:nrow(T)) {
    if (all(P[i,] != T[j,])) {                                      #if (all(P[1,] == T[j,]) == FALSE) {
     distances <- sqrt(sum((P[i,] - T[j,])^2))
    
     # Step 4: Check if the nearest instance is in the minority class
      if (distances < nearest_distance) {
        nearest_distance <- distances
        index <- j
      }
    }
  }
  
  if (train[index,]$Class == 0) {
    # Step 5: Record the unqualified minority instance
    Mu <- rbind(Mu, P[i, ])
  } else {
    # Step 6: Record the qualified minority instance
    Mq <- rbind(Mq, P[i, ])
  }
}


### Algorithm 2



### Algorithm 3

