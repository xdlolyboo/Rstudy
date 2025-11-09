library(ISLR)
library(ggplot2)
library(cluster)

df1 <- read.csv('winequality-red.csv',sep=';')
df2 <- read.csv('winequality-white.csv',sep=';')

df1$label <- sapply(df1$pH,function(x){'red'})
df2$label <- sapply(df2$pH,function(x){'white'})

wine <- rbind(df1,df2)

pl <- ggplot(wine,aes(x=residual.sugar)) + geom_histogram(aes(fill=label),color='black',bins=50)
pl + scale_fill_manual(values = c('#ae4554','#faf7ea')) + theme_bw()

pl2 <- ggplot(wine,aes(x=citric.acid)) + geom_histogram(aes(fill=label),color='black',bins=50)
pl2 + scale_fill_manual(values = c('#ae4554','#faf7ea')) + theme_bw()

pl3 <- ggplot(wine,aes(x=alcohol)) + geom_histogram(aes(fill=label),color='black',bins=50)
pl3 + scale_fill_manual(values = c('#ae4554','#faf7ea')) + theme_bw()

pl4 <- ggplot(wine,aes(x=citric.acid,y=residual.sugar)) + geom_point(aes(color=label),alpha=0.2)
pl4 + scale_color_manual(values = c('#ae4554','#faf7ea')) +theme_dark()

pl5 <- ggplot(wine,aes(x=volatile.acidity,y=residual.sugar)) + geom_point(aes(color=label),alpha=0.2)
pl5 + scale_color_manual(values = c('#ae4554','#faf7ea')) +theme_dark()

clus.data <- wine[,1:12]
wine.cluster <- kmeans(clus.data,2)







