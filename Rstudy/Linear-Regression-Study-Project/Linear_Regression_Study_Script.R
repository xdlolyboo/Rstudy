# Linear Regression Study Project
#
# Predicts bike rental demand using linear regression on the 
# Kaggle Bike Sharing dataset. Focuses on exploratory analysis
# and demonstrates linear regression limitations with nonlinear data.
#
# Dataset: bikeshare.csv (Kaggle Bike Sharing Demand)
# Libraries: ggplot2, dplyr

library(ggplot2)
library(dplyr)

# Load dataset
bike <- read.csv("bikeshare.csv")

# Convert datetime string to POSIXct format
bike$datetime <- as.POSIXct(bike$datetime)

# Exploratory Data Analysis

# Scatter plot: count vs datetime, colored by temperature
pl <- ggplot(bike, aes(datetime, count)) + 
  geom_point(aes(color = temp), alpha = 0.5)
pl + scale_color_continuous(low = 'black', high = '#FFFF00') + theme_bw()

# Check correlation between temperature and rental count
cor(bike[, c('temp', 'count')])

# Box plot: rental count distribution by season
ggplot(bike, aes(factor(season), count)) + 
  geom_boxplot(aes(color = factor(season))) + 
  theme_bw()

# Feature Engineering

# Extract hour from datetime for hourly pattern analysis
bike$hour <- sapply(bike$datetime, function(x) { format(x, "%H") })
bike$hour <- sapply(bike$hour, as.numeric)

# Scatter plot: hourly rentals on working days
pl2 <- ggplot(filter(bike, workingday == 1), aes(hour, count))
pl2 <- pl2 + geom_point(position = position_jitter(w = 1, h = 0), 
                        aes(color = temp), alpha = 0.5)
pl2 <- pl2 + scale_color_gradientn(colors = c('darkblue', 'blue', 'lightblue', 
                                               'yellow', 'orange', 'red'))

# Scatter plot: hourly rentals on non-working days
pl3 <- ggplot(filter(bike, workingday == 0), aes(hour, count))
pl3 <- pl3 + geom_point(position = position_jitter(w = 1, h = 0), 
                        aes(color = temp), alpha = 0.5)
pl3 <- pl3 + scale_color_gradientn(colors = c('darkblue', 'blue', 'lightblue', 
                                               'yellow', 'orange', 'red'))

# Linear Regression Model

# Build model excluding redundant/derived columns
model <- lm(count ~ . - casual - registered - datetime - atemp, data = bike)

# View model summary
summary(model)
