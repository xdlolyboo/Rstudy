library(MASS)
library(neuralnet)
library(caTools)
library(randomForest)


df <- read.csv('bank_note_data.csv')

set.seed(101)
split = sample.split(df$Class, SplitRatio = 0.70)

train = subset(df, split == TRUE)
test = subset(df, split == FALSE)

nn <- neuralnet(Class ~ Image.Var + Image.Skew + Image.Curt + Entropy,data=train,hidden=10,linear.output=FALSE)

predicted.nn.values <- compute(nn,test[,1:4])

predictions <- sapply(predicted.nn.values$net.result,round)


df$Class <- factor(df$Class)
library(caTools)
set.seed(101)
split = sample.split(df$Class, SplitRatio = 0.70)

train = subset(df, split == TRUE)
test = subset(df, split == FALSE)


model <- randomForest(Class ~ Image.Var + Image.Skew + Image.Curt + Entropy,data=train)

rf.pred <- predict(model,test)

