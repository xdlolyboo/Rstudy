# K-Nearest Neighbors (KNN) Study Project
#
# Implements KNN classification on the Iris dataset to demonstrate
# distance-based learning. Includes standardization, elbow method
# for optimal K selection, and error rate visualization.
#
# Dataset: Iris (built-in via ISLR/datasets)
# Libraries: ISLR, caTools, class, ggplot2

library(ISLR)
library(caTools)
library(class)
library(ggplot2)

# Data Preparation

# Standardize features (mean=0, sd=1) for distance-based algorithm
stand.features <- scale(iris[1:4])

# Combine standardized features with species labels
final.data <- cbind(stand.features, iris[5])

# Train-Test Split

set.seed(101)
sample <- sample.split(final.data$Species, SplitRatio = 0.7)
train <- subset(final.data, sample == TRUE)
test <- subset(final.data, sample == FALSE)

# Initial KNN with K=1

# Classify test set using single nearest neighbor
predicted.species <- knn(train[1:4], test[1:4], train$Species, k = 1)

# Elbow Method: Find Optimal K

# Test K values from 1 to 10 and record error rates
predicted.species <- NULL
error.rate <- NULL

for (i in 1:10) {
  set.seed(101)
  predicted.species <- knn(train[1:4], test[1:4], train$Species, k = i)
  error.rate[i] <- mean(test$Species != predicted.species)
}

# Create data frame for visualization
k.values <- 1:10
error.df <- data.frame(error.rate, k.values)

# Visualize Error Rate vs K

pl <- ggplot(error.df, aes(x = k.values, y = error.rate)) + 
  geom_point(size = 3, color = 'red')
pl + geom_line(lty = "dotted", color = 'red') +
  labs(x = "K Value", y = "Error Rate", 
       title = "KNN Error Rate vs K (Elbow Method)") +
  theme_bw()
