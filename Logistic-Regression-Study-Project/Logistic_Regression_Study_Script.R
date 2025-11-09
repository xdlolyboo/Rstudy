library(ggplot2)
library(dplyr)
library(caTools)
library(corrplot)
library(Amelia)


adult <- read.csv('adult_sal.csv')
adult <- select(adult,-X)

unemp <- function(job){
  
  job <- as.character(job)
  if(job=='Never-worked' | job == 'Without-pay'){
    return('Unemployed')
  }else{
    return(job)
  }
}
  
adult$type_employer <- sapply(adult$type_employer, unemp) 

group_emp <- function(job){
  if (job=='Local-gov' | job=='State-gov'){
    return('SL-gov')
  }else if (job=='Self-emp-inc' | job=='Self-emp-not-inc'){
    return('self-emp')
  }else{
    return(job)
  }
}

adult$type_employer <- sapply(adult$type_employer, group_emp) 

group_marital <- function(mar){
  mar <- as.character(mar)
  

  if (mar=='Separated' | mar=='Divorced' | mar=='Widowed'){
    return('Not-Married')
    

  }else if(mar=='Never-married'){
    return(mar)
    

  }else{
    return('Married')
  }
}

adult$marital <- sapply(adult$marital,group_marital)


Asia <- c('China','Hong','India','Iran','Cambodia','Japan', 'Laos' ,
          'Philippines' ,'Vietnam' ,'Taiwan', 'Thailand')

North.America <- c('Canada','United-States','Puerto-Rico' )

Europe <- c('England' ,'France', 'Germany' ,'Greece','Holand-Netherlands','Hungary',
            'Ireland','Italy','Poland','Portugal','Scotland','Yugoslavia')

Latin.and.South.America <- c('Columbia','Cuba','Dominican-Republic','Ecuador',
                             'El-Salvador','Guatemala','Haiti','Honduras',
                             'Mexico','Nicaragua','Outlying-US(Guam-USVI-etc)','Peru',
                             'Jamaica','Trinadad&Tobago')
Other <- c('South')

group_country <- function(ctry){
  if (ctry %in% Asia){
    return('Asia')
  }else if (ctry %in% North.America){
    return('North.America')
  }else if (ctry %in% Europe){
    return('Europe')
  }else if (ctry %in% Latin.and.South.America){
    return('Latin.and.South.America')
  }else{
    return('Other')      
  }
}

adult$country <- sapply(adult$country,group_country)


adult[adult == '?'] <- NA


adult$type_employer <- sapply(adult$type_employer,factor)
adult$country <- sapply(adult$country,factor)
adult$marital <- sapply(adult$marital,factor)




missmap(adult,y.at=c(1),y.labels = c(''),col=c('yellow','black'))

adult <- na.omit(adult)

missmap(adult,y.at=c(1),y.labels = c(''),col=c('yellow','black'))

ggplot(adult,aes(age)) + geom_histogram(aes(fill=income),color='black',binwidth=1) + theme_bw()

ggplot(adult,aes(hr_per_week)) + geom_histogram() + theme_bw()

adult <- rename(adult, region = country)

pl <- ggplot(adult,aes(region)) + geom_bar(aes(fill=income),color='black')+theme_bw()


adult$income <- ifelse(adult$income == '>50K', 1, 0)

set.seed(101)

sample <- sample.split(adult$income, SplitRatio = 0.7)

train <- subset(adult, sample==T)
test <- subset(adult,sample==F)

model <- glm(income ~ ., family = binomial(link = 'logit'),data=train)

new.step.model <- step(model)

test$predicted.income <- predict(model, newdata = test, type = 'response')

table(test$income, test$predicted.income > 0.5)

acc <- (6372+1423)/(6372+1423+548+872)

rec <- 6732/(6372+548)

prec <- 6732/(6372+872)





