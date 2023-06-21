getwd()
setwd("C:/Users/Dennis/OneDrive/Dokumente/03_Master BAOR/05_Kurse/01_Business Analytics/04_Data Analytics Challenge/DAC")
dataset <- read.csv(file = "creditcard.csv")
dataset
typeof(dataset)

View(dataset)

#transform to tibble
library(tidyverse)
dataset <- as_tibble(dataset)

#filter for frauds
fraud_distribution <- filter(.data = dataset, Class == 1)
fraud_distribution

#############Analysis#############

###Descriptive Analyses###
no_fraud <- summary(dataset)
fraud <- summary(fraud_distribution)

fraud

###Distributions###
library(purrr)
library(tidyr)
library(ggplot2)

#All
dataset %>%
  keep(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram()

#No Fraud
no_fraud_distribution <- filter(.data = dataset, Class == 0)

no_fraud_distribution %>%
  keep(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram()

#Fraud
fraud_distribution %>%
  keep(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram()

#outlier detection
zscores <- as.data.frame(sapply(dataset, function(data) abs(data-mean(data))/sd(data)))
no_outliers <- zscores[!rowSums(zscores>3),]

fraud_no_outlier<- filter(.data = no_outliers, Class == 1)

dim(dataset)
dim(no_outliers)


no_outliers %>%
  keep(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram()

#Correlationsmatrix
#schmeiße 3 Variablen raus
subset <-  subset(no_outliers, select = -c(Time, Amount, Class))
correlation1 <- cor(subset, method = "pearson")

#install.packages("corrplot")
library(corrplot)

correlation <- corrplot(correlation1)
