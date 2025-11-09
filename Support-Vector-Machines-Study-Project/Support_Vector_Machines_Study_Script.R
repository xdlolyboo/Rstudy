library(ISLR)
library(e1071)
library(ggplot2)
library(caTools)



loans <- read.csv('loan_data.csv')

loans$credit.policy <- factor(loans$credit.policy)
loans$inq.last.6mths <- factor(loans$inq.last.6mths)
loans$delinq.2yrs <- factor(loans$delinq.2yrs)
loans$pub.rec <- factor(loans$pub.rec)
loans$not.fully.paid <- factor(loans$not.fully.paid)


pl <- ggplot(loans,aes(x=fico)) 
pl <- pl + geom_histogram(aes(fill=not.fully.paid),color='black',bins=40,alpha=0.5)
pl + scale_fill_manual(values = c('green','red')) + theme_bw()


pl2 <- ggplot(loans,aes(x=factor(purpose))) 
pl2 <- pl2 + geom_bar(aes(fill=not.fully.paid),position = "dodge")
pl2 + theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1))

pl3 <- ggplot(loans,aes(int.rate,fico)) +geom_point(aes(color=not.fully.paid),alpha=0.3) + theme_bw()

set.seed(101)

sample <- sample.split(loans$not.fully.paid, 0.7)
train <- subset(loans, sample == TRUE)
test <- subset(loans, sample == FALSE)

model <- svm(not.fully.paid ~ .,data=train)

predicted.values <- predict(model,test[1:13])

tuned.results <- tune(svm,train.x=not.fully.paid~., data=train,kernel='radial',
                     ranges=list(cost=c(100,200), gamma=c(0.1)))



tuned.model <- svm(not.fully.paid ~ .,data=train,cost=100,gamma = 0.1)
tuned.predicted.values <- predict(tuned.model,test[1:13])
table(tuned.predicted.values,test$not.fully.paid) 









