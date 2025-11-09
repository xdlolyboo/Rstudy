library(ISLR)
library(caTools)
library(class)
library(ggplot2)
stand.features <- scale(iris[1:4])

final.data <- cbind(stand.features, iris[5])

set.seed(101)

sample <- sample.split(final.data$Species, SplitRatio = 0.7)

train <- subset(final.data, sample == T)
test <- subset(final.data, sample == F)

predicted.species <- knn(train[1:4], test[1:4], train$Species, k=1)


predicted.species <- NULL
error.rate <- NULL

for(i in 1:10){
  set.seed(101)
  predicted.species <- knn(train[1:4],test[1:4],train$Species,k=i)
  error.rate[i] <- mean(test$Species != predicted.species)
}

k.values <- 1:10
error.df <- data.frame(error.rate, k.values)

pl <- ggplot(error.df, aes(x=k.values, y=error.rate)) + geom_point()
pl + geom_line(lty="dotted",color='red')







