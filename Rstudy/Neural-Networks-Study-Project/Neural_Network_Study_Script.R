# Neural Networks Study Project
#
# Trains a neural network to detect forged banknotes using the
# UCI Banknote Authentication dataset. Compares neural network
# performance against random forest baseline.
#
# Dataset: bank_note_data.csv (UCI Banknote Authentication)
# Libraries: MASS, neuralnet, caTools, randomForest

library(MASS)
library(neuralnet)
library(caTools)
library(randomForest)

# Load banknote authentication data
df <- read.csv('bank_note_data.csv')

# Neural Network Model

# Train-test split (70/30)
set.seed(101)
split <- sample.split(df$Class, SplitRatio = 0.70)
train <- subset(df, split == TRUE)
test <- subset(df, split == FALSE)

# Build neural network with 10 hidden neurons
nn <- neuralnet(Class ~ Image.Var + Image.Skew + Image.Curt + Entropy,
                data = train,
                hidden = 10,
                linear.output = FALSE)

# Generate predictions on test set
predicted.nn.values <- compute(nn, test[, 1:4])

# Round predictions to obtain class labels
predictions <- sapply(predicted.nn.values$net.result, round)

# Confusion matrix for neural network
table(predictions, test$Class)

# Random Forest Comparison

# Convert target to factor for classification
df$Class <- factor(df$Class)

# Fresh train-test split for fair comparison
set.seed(101)
split <- sample.split(df$Class, SplitRatio = 0.70)
train <- subset(df, split == TRUE)
test <- subset(df, split == FALSE)

# Train random forest model
model <- randomForest(Class ~ Image.Var + Image.Skew + Image.Curt + Entropy,
                      data = train)

# Generate predictions
rf.pred <- predict(model, test)

# Confusion matrix for random forest
table(rf.pred, test$Class)
