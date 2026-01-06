# Support Vector Machines (SVM) Study Project
#
# Predicts loan repayment using SVM on Lending Club data.
# Compares linear kernel performance with tuned RBF kernel
# using grid search for hyperparameter optimization.
#
# Dataset: loan_data.csv (Lending Club)
# Libraries: ISLR, e1071, ggplot2, caTools

library(ISLR)
library(e1071)
library(ggplot2)
library(caTools)

# Load loan data
loans <- read.csv('loan_data.csv')

# Data Preprocessing

# Convert categorical variables to factors
loans$credit.policy <- factor(loans$credit.policy)
loans$inq.last.6mths <- factor(loans$inq.last.6mths)
loans$delinq.2yrs <- factor(loans$delinq.2yrs)
loans$pub.rec <- factor(loans$pub.rec)
loans$not.fully.paid <- factor(loans$not.fully.paid)

# Exploratory Data Analysis

# FICO score distribution by payment status
pl <- ggplot(loans, aes(x = fico))
pl <- pl + geom_histogram(aes(fill = not.fully.paid), color = 'black', 
                          bins = 40, alpha = 0.5)
pl + scale_fill_manual(values = c('green', 'red')) + theme_bw()

# Loan purpose distribution by payment status
pl2 <- ggplot(loans, aes(x = factor(purpose)))
pl2 <- pl2 + geom_bar(aes(fill = not.fully.paid), position = "dodge")
pl2 + theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Interest rate vs FICO score scatter plot
pl3 <- ggplot(loans, aes(int.rate, fico)) + 
  geom_point(aes(color = not.fully.paid), alpha = 0.3) + 
  theme_bw()

# Train-Test Split

set.seed(101)
sample <- sample.split(loans$not.fully.paid, SplitRatio = 0.7)
train <- subset(loans, sample == TRUE)
test <- subset(loans, sample == FALSE)

# SVM with Default Parameters

# Train SVM with default RBF kernel
model <- svm(not.fully.paid ~ ., data = train)

# Predict on test set
predicted.values <- predict(model, test[1:13])

# Confusion matrix
table(predicted.values, test$not.fully.paid)

# Hyperparameter Tuning

# Grid search for optimal cost and gamma parameters
tuned.results <- tune(svm, train.x = not.fully.paid ~ ., data = train,
                      kernel = 'radial',
                      ranges = list(cost = c(100, 200), gamma = c(0.1)))

# View best parameters
summary(tuned.results)

# Tuned SVM Model

# Train model with optimized parameters
tuned.model <- svm(not.fully.paid ~ ., data = train, cost = 100, gamma = 0.1)

# Generate predictions
tuned.predicted.values <- predict(tuned.model, test[1:13])

# Final confusion matrix
table(tuned.predicted.values, test$not.fully.paid)
