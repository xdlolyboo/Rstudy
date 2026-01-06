# K-Means Clustering Study Project
#
# Applies K-Means clustering to red and white wine samples from
# the UCI Wine Quality dataset. Validates unsupervised clustering
# quality by comparing cluster assignments to actual wine types.
#
# Dataset: winequality-red.csv, winequality-white.csv (UCI Wine Quality)
# Libraries: ISLR, ggplot2, cluster

library(ISLR)
library(ggplot2)
library(cluster)

# Load and Combine Datasets

# Read wine data (semicolon-separated)
df1 <- read.csv('winequality-red.csv', sep = ';')
df2 <- read.csv('winequality-white.csv', sep = ';')

# Add wine type labels
df1$label <- sapply(df1$pH, function(x) { 'red' })
df2$label <- sapply(df2$pH, function(x) { 'white' })

# Combine into single dataset
wine <- rbind(df1, df2)

# Exploratory Data Analysis

# Residual sugar distribution by wine type
pl <- ggplot(wine, aes(x = residual.sugar)) + 
  geom_histogram(aes(fill = label), color = 'black', bins = 50)
pl + scale_fill_manual(values = c('#ae4554', '#faf7ea')) + theme_bw()

# Citric acid distribution
pl2 <- ggplot(wine, aes(x = citric.acid)) + 
  geom_histogram(aes(fill = label), color = 'black', bins = 50)
pl2 + scale_fill_manual(values = c('#ae4554', '#faf7ea')) + theme_bw()

# Alcohol content distribution
pl3 <- ggplot(wine, aes(x = alcohol)) + 
  geom_histogram(aes(fill = label), color = 'black', bins = 50)
pl3 + scale_fill_manual(values = c('#ae4554', '#faf7ea')) + theme_bw()

# Scatter plot: citric acid vs residual sugar
pl4 <- ggplot(wine, aes(x = citric.acid, y = residual.sugar)) + 
  geom_point(aes(color = label), alpha = 0.2)
pl4 + scale_color_manual(values = c('#ae4554', '#faf7ea')) + theme_dark()

# Scatter plot: volatile acidity vs residual sugar
pl5 <- ggplot(wine, aes(x = volatile.acidity, y = residual.sugar)) + 
  geom_point(aes(color = label), alpha = 0.2)
pl5 + scale_color_manual(values = c('#ae4554', '#faf7ea')) + theme_dark()

# K-Means Clustering

# Select numeric features only (exclude label and quality score)
clus.data <- wine[, 1:12]

# Apply K-Means with k=2 (expecting red/white separation)
wine.cluster <- kmeans(clus.data, centers = 2, nstart = 25)

# Evaluate Clustering Quality

# Compare cluster assignments to actual labels
table(wine$label, wine.cluster$cluster)

# View cluster centers
wine.cluster$centers
