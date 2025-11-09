library(caTools)
library(rpart)
library(rpart.plot)
library(randomForest)
library(ISLR)
library(ggplot2)

df <- College
ggplot(df, aes(Room.Board,Grad.Rate)) + geom_point(aes(color=Private), size=4, alpha=0.5)

ggplot(df,aes(F.Undergrad)) + geom_histogram(aes(fill=Private),color='black',bins=50)

ggplot(df,aes(Grad.Rate)) + geom_histogram(aes(fill=Private),color='black',bins=50)

df['Cazenovia College','Grad.Rate'] <- 100

set.seed(101)


sample <- sample.split(df$Private, SplitRatio=0.7)
train <- subset(df, sample == TRUE)
test <- subset(df, sample == FALSE)

tree <- rpart(Private ~ .,method='class',data=train)
tree.preds <- predict(tree,test)

tree.preds <- as.data.frame(tree.preds)

joiner <- function(x){
  if (x>=0.5){
    return('Yes')
  }else{
    return("No")
  }
}


tree.preds$Private <- sapply(tree.preds$Yes,joiner)

prp(tree)

rf.model <- randomForest(Private ~ .,data=train,importance=TRUE)

rf.preds <- predict(rf.model,test)




