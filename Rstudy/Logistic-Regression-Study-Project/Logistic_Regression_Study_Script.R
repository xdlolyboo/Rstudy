# Logistic Regression Study Project
#
# Classifies individuals as earning above or below 50K USD using
# logistic regression on the UCI Adult Income dataset. Emphasizes
# data cleaning, encoding categorical variables, and model evaluation.
#
# Dataset: adult_sal.csv (UCI Adult Income)
# Libraries: ggplot2, dplyr, caTools, Amelia

library(ggplot2)
library(dplyr)
library(caTools)
library(Amelia)

# Load dataset and drop unnecessary index column
adult <- read.csv('adult_sal.csv')
adult <- select(adult, -X)

# Data Cleaning: Employment Type

# Combine unemployed categories
unemp <- function(job) {
  job <- as.character(job)
  if (job == 'Never-worked' | job == 'Without-pay') {
    return('Unemployed')
  } else {
    return(job)
  }
}
adult$type_employer <- sapply(adult$type_employer, unemp)

# Group government and self-employment types
group_emp <- function(job) {
  if (job == 'Local-gov' | job == 'State-gov') {
    return('SL-gov')
  } else if (job == 'Self-emp-inc' | job == 'Self-emp-not-inc') {
    return('self-emp')
  } else {
    return(job)
  }
}
adult$type_employer <- sapply(adult$type_employer, group_emp)

# Data Cleaning: Marital Status

# Simplify marital status into three categories
group_marital <- function(mar) {
  mar <- as.character(mar)
  if (mar == 'Separated' | mar == 'Divorced' | mar == 'Widowed') {
    return('Not-Married')
  } else if (mar == 'Never-married') {
    return(mar)
  } else {
    return('Married')
  }
}
adult$marital <- sapply(adult$marital, group_marital)

# Data Cleaning: Country to Region

# Define regional groupings
Asia <- c('China', 'Hong', 'India', 'Iran', 'Cambodia', 'Japan', 'Laos',
          'Philippines', 'Vietnam', 'Taiwan', 'Thailand')
North.America <- c('Canada', 'United-States', 'Puerto-Rico')
Europe <- c('England', 'France', 'Germany', 'Greece', 'Holand-Netherlands',
            'Hungary', 'Ireland', 'Italy', 'Poland', 'Portugal', 'Scotland',
            'Yugoslavia')
Latin.and.South.America <- c('Columbia', 'Cuba', 'Dominican-Republic', 'Ecuador',
                             'El-Salvador', 'Guatemala', 'Haiti', 'Honduras',
                             'Mexico', 'Nicaragua', 'Outlying-US(Guam-USVI-etc)',
                             'Peru', 'Jamaica', 'Trinadad&Tobago')

# Map countries to regions
group_country <- function(ctry) {
  if (ctry %in% Asia) {
    return('Asia')
  } else if (ctry %in% North.America) {
    return('North.America')
  } else if (ctry %in% Europe) {
    return('Europe')
  } else if (ctry %in% Latin.and.South.America) {
    return('Latin.and.South.America')
  } else {
    return('Other')
  }
}
adult$country <- sapply(adult$country, group_country)

# Handle Missing Values

# Replace '?' with NA
adult[adult == '?'] <- NA

# Convert cleaned columns to factors
adult$type_employer <- sapply(adult$type_employer, factor)
adult$country <- sapply(adult$country, factor)
adult$marital <- sapply(adult$marital, factor)

# Visualize missing data pattern
missmap(adult, y.at = c(1), y.labels = c(''), col = c('yellow', 'black'))

# Remove rows with missing values
adult <- na.omit(adult)

# Exploratory Data Analysis

# Age distribution by income level
ggplot(adult, aes(age)) + 
  geom_histogram(aes(fill = income), color = 'black', binwidth = 1) + 
  theme_bw()

# Hours per week distribution
ggplot(adult, aes(hr_per_week)) + 
  geom_histogram() + 
  theme_bw()

# Rename country to region for clarity
adult <- rename(adult, region = country)

# Income distribution by region
ggplot(adult, aes(region)) + 
  geom_bar(aes(fill = income), color = 'black') + 
  theme_bw()

# Model Training

# Convert income to binary (1 = >50K, 0 = <=50K)
adult$income <- ifelse(adult$income == '>50K', 1, 0)

# Split data: 70% train, 30% test
set.seed(101)
sample <- sample.split(adult$income, SplitRatio = 0.7)
train <- subset(adult, sample == TRUE)
test <- subset(adult, sample == FALSE)

# Train logistic regression model
model <- glm(income ~ ., family = binomial(link = 'logit'), data = train)

# Stepwise feature selection
new.step.model <- step(model)

# Model Evaluation

# Generate predictions on test set
test$predicted.income <- predict(model, newdata = test, type = 'response')

# Confusion matrix with 0.5 threshold
table(test$income, test$predicted.income > 0.5)

# Calculate metrics
acc <- (6372 + 1423) / (6372 + 1423 + 548 + 872)
rec <- 6372 / (6372 + 548)
prec <- 6372 / (6372 + 872)

cat("Accuracy:", acc, "\n")
cat("Recall:", rec, "\n")
cat("Precision:", prec, "\n")
