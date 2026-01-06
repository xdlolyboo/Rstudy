# Decision Trees and Random Forests Study Project
#
# Classifies colleges as Private or Public using decision tree
# and random forest models. Demonstrates how ensemble methods
# improve generalization over single trees.
#
# Dataset: College (ISLR package)
# Libraries: caTools, rpart, rpart.plot, randomForest, ISLR, ggplot2

library(caTools)
library(rpart)
library(rpart.plot)
library(randomForest)
library(ISLR)
library(ggplot2)

# Load the College dataset from ISLR
df <- College

# Exploratory Data Analysis

# Scatter plot: Room & Board cost vs Graduation Rate
ggplot(df, aes(Room.Board, Grad.Rate)) + 
  geom_point(aes(color = Private), size = 4, alpha = 0.5) +
  theme_bw()

# Histogram: Full-time undergraduate enrollment by Private status
ggplot(df, aes(F.Undergrad)) + 
  geom_histogram(aes(fill = Private), color = 'black', bins = 50) +
  theme_bw()

# Histogram: Graduation rate distribution
ggplot(df, aes(Grad.Rate)) + 
  geom_histogram(aes(fill = Private), color = 'black', bins = 50) +
  theme_bw()

# Fix data error: Cap graduation rate at 100%
df['Cazenovia College', 'Grad.Rate'] <- 100

# Train-Test Split

set.seed(101)
sample <- sample.split(df$Private, SplitRatio = 0.7)
train <- subset(df, sample == TRUE)
test <- subset(df, sample == FALSE)

# Decision Tree Model

# Build classification tree using all features
tree <- rpart(Private ~ ., method = 'class', data = train)

# Generate predictions
tree.preds <- predict(tree, test)
tree.preds <- as.data.frame(tree.preds)

# Convert probabilities to class labels
joiner <- function(x) {
  if (x >= 0.5) {
    return('Yes')
  } else {
    return('No')
  }
}
tree.preds$Private <- sapply(tree.preds$Yes, joiner)

# Visualize decision tree
prp(tree)

# Confusion matrix for decision tree
table(test$Private, tree.preds$Private)

# Random Forest Model

# Train random forest with 100 trees
rf.model <- randomForest(Private ~ ., data = train, importance = TRUE)

# Generate predictions
rf.preds <- predict(rf.model, test)

# Confusion matrix for random forest
table(test$Private, rf.preds)

# View feature importance
importance(rf.model)
varImpPlot(rf.model)
