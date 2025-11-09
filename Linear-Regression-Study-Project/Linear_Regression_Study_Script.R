#linear regression study project script

library(ggplot2)
library(dplyr)

bike <- read.csv("bikeshare.csv")



bike$datetime <- as.POSIXct(bike$datetime)

pl <- ggplot(bike, aes(datetime, count)) + geom_point(aes(color = temp), alpha = 0.5)

pl + scale_color_continuous(low = 'black', high = '#FFFF00') + theme_bw()

cor(bike[,c('temp','count')])

ggplot(bike, aes(factor(season), count)) + geom_boxplot(aes(color=factor(season))) + theme_bw()

bike$hour <- sapply(bike$datetime, function(x){format(x, "%H")})
bike$hour <- sapply(bike$hour, as.numeric)

pl2 <- ggplot(filter(bike, workingday == 1), aes(hour, count))
pl2 <- pl2 + geom_point(position = position_jitter(w=1, h=0), aes(color = temp), alpha=0.5)
pl2 <- pl2 + scale_color_gradientn(colors = c('darkblue', 'blue', 'lightblue', 'yellow', 'orange', 'red'))

pl3 <- ggplot(filter(bike, workingday == 0), aes(hour, count))
pl3 <- pl3 + geom_point(position = position_jitter(w=1, h=0), aes(color = temp), alpha=0.5)
pl3 <- pl3 + scale_color_gradientn(colors = c('darkblue', 'blue', 'lightblue', 'yellow', 'orange', 'red'))


model <- lm(count ~ . -casual -registered -datetime -atemp , bike) 










