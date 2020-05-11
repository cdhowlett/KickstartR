# First load the three files downloaded from Github, unzip them and join them together 
# (they were split because Github has a 25MB file size limit)
if (!require(tidyverse)) install.packages('tidyverse')
if (!require(caret)) install.packages('caret')
if (!require(sentimentr)) install.packages('sentimentr')
library(sentimentr)
library(tidyverse)
library(caret)

ks1 <- read_csv(unzip("kickstarter1.csv.zip"))
ks2 <- read_csv(unzip("kickstarter2.csv.zip"))
ks3 <- read_csv(unzip("kickstarter3.csv.zip"))

ks <- rbind(ks1, ks2)
ks <- rbind(ks, ks3)

rm(ks1, ks2, ks3)

# Next we need to split off the validation data
set.seed(1, sample.kind="Rounding")

test_index <- createDataPartition(y = ks$binary_state, times = 1, p = 0.1, list = FALSE)
kstrain <- ks[-test_index,]
temp <- ks[test_index,]

validation <- temp %>% 
  semi_join(kstrain, by = "category_name") %>%
  semi_join(kstrain, by = 'location_country') %>%
  semi_join(kstrain, by = 'blurb_length')

removed <- anti_join(temp, validation)

kstrain <- rbind(kstrain, removed)

# Let's see what kind of data we have
# In kstrain we have 387843 Kickstarter projects
nrow(kstrain)

# These were from 2009 to 2019
min(unique(kstrain$year))
max(unique(kstrain$year))

# We can see the distribution across the years
kstrain %>% group_by(year) %>% summarize(n=n()) %>% ggplot(aes(year,n)) + geom_bar(stat="identity")

# We're predicting the 'binary_state' - whether the project was successful in raising its crowdfunding, of if it failed.
unique(kstrain$binary_state)

# We see that 40.3% were successful
mean(kstrain$binary_state == "successful")

# Let's see if that success percentage varies by year
kstrain %>% group_by(year) %>% summarize(mean_success=mean(binary_state == "successful")) %>% ggplot(aes(year,mean_success)) + geom_bar(stat="identity")

# So it looks like there is a clear effect by year

# We also have information about which month they were launched - perhaps there's a seasonal effect
kstrain %>% group_by(month) %>% summarize(mean_success=mean(binary_state == "successful")) %>% ggplot(aes(month,mean_success)) + geom_bar(stat="identity")

# There seems to be much less variation, although a launch in September, October or November seems to give a small boost

# We also have information about the day of the month of the launch
kstrain %>% group_by(day) %>% summarize(mean_success=mean(binary_state == "successful")) %>% ggplot(aes(day,mean_success)) + geom_bar(stat="identity")

# There seems to be a clear boost for projects launched on the 1st of the month, and for those launched in the middle of the month

# Next we have information about where the project was located - let's start by looking at how many countries are represented: 208
length(unique(kstrain$location_country))

# So what does the distribution look like of the number of projects by country?
kstrain %>% group_by(location_country) %>% summarize(n=n()) %>% ggplot(aes(n)) + geom_histogram(bins=200)

# It looks like almost all countries have very few project
# In fact we can see that only 19 countries have more than 1000 projects, and the great majority are in the US:
kstrain %>% group_by(location_country) %>% summarize(n=n(), mean_success=mean(binary_state=="successful")) %>% filter(n >= 1000) %>% arrange(desc(n))

# If we plot the mean success in a country against a log transformation of the number of projects a country has, 
# we see a small effect particularly for countries with a moderate number of projects, however, this looks like noise due to there being so few projects
kstrain %>% group_by(location_country) %>% summarize(n=n(), mean_success=mean(binary_state=="successful")) %>% ggplot(aes(log(n), mean_success)) + geom_point() + geom_smooth()

# Next let's look at the effect of the goal
# We see that there are 74674 different goal values in USD
length(unique(kstrain$goal_USD))

# If we look at the distribution of a log transformation of the goal, we can see it is roughly normal
kstrain %>% ggplot(aes(log(goal_USD))) + geom_density(adjust=2)

# So let's look at the implications of the goal on the success rate
kstrain %>% group_by(goal=round(goal_USD, -2)) %>% 
  summarize(n = n(), mean_success=mean(binary_state=="successful")) %>% 
  filter(n >= 10) %>% 
  ggplot(aes(log(goal), mean_success, color=log(n))) + 
  geom_point() + 
  scale_color_gradient(low="blue", high="red") + 
  geom_smooth()

# So we can see that the goal amount to raise has a strong influence on the success or otherwise of the project

# Now lets look at category. We see that there are 159 distinct categories
length(unique(kstrain$category_name))

# Let's plot success rate against category
every_nth = function(n) {return(function(x) {x[c(TRUE, rep(FALSE, n - 1))]})}

kstrain %>% group_by(category_name) %>% 
  summarize(success_rate=mean(binary_state=="successful")) %>% 
  arrange(desc(success_rate))  %>% 
  mutate(category_name=factor(category_name, levels=category_name)) %>% 
  ggplot(aes(category_name, success_rate)) + 
  geom_bar(stat="identity") + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  scale_x_discrete(breaks = every_nth(n = 10))

# We can see that category is very predictive of success

# We might be slightly worried that this is skewed by very small categories. Let's exclude those categories with fewer than 1000 projects
kstrain %>% group_by(category_name) %>% 
  summarize(success_rate=mean(binary_state=="successful"), n=n()) %>% filter(n>=1000) %>% 
  arrange(desc(success_rate))  %>% 
  mutate(category_name=factor(category_name, levels=category_name)) %>% 
  ggplot(aes(category_name, success_rate)) + 
  geom_bar(stat="identity") + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  scale_x_discrete(breaks = every_nth(n = 10))

# We still see a very strong effect and in fact, if we filter out categories with fewer than 1000 projects and just plot the number of projects against success_rate we still see an effect
kstrain %>% group_by(category_name) %>% 
  summarize(success_rate=mean(binary_state=="successful"), n=n()) %>% filter(n>=1000) %>% ggplot(aes(n, success_rate)) + 
  geom_point() + geom_smooth()


# Now let's look at blurb length
# There are only 37 unique values for blurb length, so let's plot against mean success rate 
length(unique(kstrain$blurb_length))

# We can see that the distribution is a kind of skewed normal
kstrain %>% ggplot(aes(blurb_length)) + geom_density(adjust=1.8)

# If we filter out rare long blurbs we see a humped distribution, suggesting some impact from blurb length on success
kstrain %>% group_by(blurb_length) %>% filter(blurb_length <= 30) %>% 
  summarize(mean_success=mean(binary_state=="successful")) %>% 
  ggplot(aes(blurb_length, mean_success)) + geom_bar(stat="identity")

# Now let's look at sentiment analysis of the blurb
blurb_sentences <- get_sentences(kstrain$blurb)
blurb_sentiments <- sentiment_by(blurb_sentences)

# If we plot the distribution of the sentiments, we see that it has two peaks - one at zero - neutral - and one a little bit more positive.
# Let's see if there is any difference in sentiment between those projects that succeeded and those that did not
data.frame(success=kstrain$binary_state, sentiment=blurb_sentiments$ave_sentiment) %>% ggplot(aes(success, sentiment)) + geom_point() + geom_boxplot()

# It's fair to say there is not a great deal of difference. 
# If we summarize the data, we see a small bias - the failed projects have slightly more positive blurbs than the successful ones
data.frame(success=kstrain$binary_state, sentiment=blurb_sentiments$ave_sentiment) %>% group_by(success) %>% summarize(mean(sentiment))

# Now let's look at backer's count. There are 4649 unique values for backers
length(unique(kstrain$backers_count))

#If we look at the distribution, we can see that a log transformation has a clear hump in the distribution
kstrain %>% ggplot(aes(log(backers_count))) + geom_density(adjust=3)

# There is also a clear correlation between the number of backers and the mean success
kstrain %>% group_by(backers=round(log(backers_count))) %>% summarize(mean_success=mean(binary_state=="successful")) %>% 
  ggplot(aes(backers, mean_success)) + geom_point()

#This is not surprising, since it is backers who provide the money

# Now let's look at the days to the deadline
# There are only 93 different values for the days to the deadline, so we can treat it as a category
# If we plot mean success against days to the deadline we see two distinct groups
kstrain %>% group_by(days_to_deadline) %>% summarize(mean_success=mean(binary_state=="successful")) %>% 
  ggplot(aes(days_to_deadline, mean_success)) + geom_point() + geom_smooth()

# There clearly seems to be an effect - 
# it seems some projects race out of the gates, and some take longer to get going. 
# The eventual success rate does not seem that different

# Now let's look at whether the project was a staff pick. This is a binary variable, so we can tabulate it
kstrain %>% group_by(staff_pick) %>% summarize(mean(binary_state=="successful"))

# We see that 76% of the time, a staff pick was successful, while a project that was not a staff pick was only successful 36% of the time.

# Now let's look at whether it was a spotlight project

# We can perform the same analysis for spotlight projects
kstrain %>% group_by(spotlight) %>% summarize(mean(binary_state=="successful"))

# Here we see that spotlight projects were always successful, while non-spotlight projects were only successful around 10% of the time. 
# This is because spotlight is a facility made available to projects by Kickstarter after they have been successful.

# Finally, let's look at the amount pledged.
kstrain %>% mutate(percent_pledged=log(usd_pledged/goal_USD)) %>% ggplot(aes(binary_state, percent_pledged)) + geom_point() + geom_boxplot()

# Clearly we see that for successful projects their percent pledged is above the target, wheras for unsuccessful ones this is below the target. 
# Indeed this is the definition of success 

# Having done this investigation, now let's turn to cleaning the data. There are only a few columns we want to keep. 
kstrain_clean <- kstrain %>% select(year, month, day, location_country, category_name, goal_USD, blurb_length, days_to_deadline, backers_count, staff_pick, binary_state, usd_pledged)
validation_clean <- validation %>% select(year, month, day, location_country, category_name, goal_USD, blurb_length, days_to_deadline, backers_count, staff_pick, binary_state, usd_pledged)

# We would also like to change the binary state to logical value, take a rounded log transformation of the goal and change the location_country and category_name vectors to factors
kstrain_clean <- kstrain_clean %>% mutate(location_country=factor(location_country), category_name=factor(category_name), binary_state=binary_state=="successful", log_goal=round(log(goal_USD)))
validation_clean <- validation_clean %>% mutate(location_country=factor(location_country), category_name=factor(category_name), binary_state=binary_state=="successful", log_goal=round(log(goal_USD)))

# The first 7 columns are effectively inputs, knowable at the start of the process, while the last 4 are outputs - 
# it would be good to predict at least binary_state from the first 7, but predicting usd_pledged, would be very good if possible.
# We may find that predicting backers_count and staff_pick are also interesting.

# Now we need to further split the kstrain_clean data set into a training and test set.

set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = kstrain_clean$binary_state, times = 1, p = 0.2, 
                                  list = FALSE)
train_set <- kstrain_clean[-test_index,]
temp <- kstrain_clean[test_index,]

test_set <- temp %>% 
  semi_join(train_set, by = "category_name") %>%
  semi_join(train_set, by = 'location_country') %>%
  semi_join(train_set, by = 'blurb_length')

removed <- anti_join(temp, test_set)

train_set <- rbind(train_set, removed)

# We want to regularize our input data, so we're going to need to calculate RMSE

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Let's calculate the most rudimentary prediction

mu <- mean(train_set$binary_state)

lambdas <- seq(0, 500, 10)

rmses <- sapply(lambdas, function(l){
  
  year_avgs <- train_set %>% 
    group_by(year) %>% 
    summarize(b_y = sum(binary_state - mu)/(n()+l))
  
  month_avgs <- train_set %>% 
    left_join(year_avgs, by='year') %>%
    group_by(month) %>%
    summarize(b_m = sum(binary_state - mu - b_y)/(n()+l))
  
  day_avgs <- train_set %>% 
    left_join(year_avgs, by='year') %>%
    left_join(month_avgs, by='month') %>%
    group_by(day) %>%
    summarize(b_d = sum(binary_state - mu - b_y - b_m)/(n()+l))
  
  country_avgs <- train_set %>% 
    left_join(year_avgs, by='year') %>%
    left_join(month_avgs, by='month') %>%
    left_join(day_avgs, by='day') %>%
    group_by(location_country) %>%
    summarize(b_co = sum(binary_state - mu - b_y - b_m - b_d)/(n()+l))
  
  category_avgs <- train_set %>% 
    left_join(year_avgs, by='year') %>%
    left_join(month_avgs, by='month') %>%
    left_join(day_avgs, by='day') %>%
    left_join(country_avgs, by='location_country') %>%
    group_by(category_name) %>%
    summarize(b_cat = sum(binary_state - mu - b_y - b_m - b_d - b_co)/(n()+l))
  
  goal_avgs <- train_set %>%
    left_join(year_avgs, by='year') %>%
    left_join(month_avgs, by='month') %>%
    left_join(day_avgs, by='day') %>%
    left_join(country_avgs, by='location_country') %>%
    left_join(category_avgs, by='category_name') %>%
    group_by(log_goal) %>%
    summarize(b_g = sum(binary_state - mu - b_y - b_m - b_d - b_co - b_cat)/(n()+l))
  
  blurb_avgs <- train_set %>%
    left_join(year_avgs, by='year') %>%
    left_join(month_avgs, by='month') %>%
    left_join(day_avgs, by='day') %>%
    left_join(country_avgs, by='location_country') %>%
    left_join(category_avgs, by='category_name') %>%
    left_join(goal_avgs, by='log_goal') %>%
    group_by(blurb_length) %>%
    summarize(b_bl = sum(binary_state - mu - b_y - b_m - b_d - b_co - b_cat - b_g)/(n()+l))
  
  deadline_avgs <- train_set %>%
    left_join(year_avgs, by='year') %>%
    left_join(month_avgs, by='month') %>%
    left_join(day_avgs, by='day') %>%
    left_join(country_avgs, by='location_country') %>%
    left_join(category_avgs, by='category_name') %>%
    left_join(goal_avgs, by='log_goal') %>%
    left_join(blurb_avgs, by='blurb_length') %>%
    group_by(days_to_deadline) %>%
    summarize(b_dd = sum(binary_state - mu - b_y - b_m - b_d - b_co - b_cat - b_g - b_bl)/(n()+l))
  
  predicted_ratings <- test_set %>% 
    left_join(year_avgs, by='year') %>%
    left_join(month_avgs, by='month') %>%
    left_join(day_avgs, by='day') %>%
    left_join(country_avgs, by='location_country') %>%
    left_join(category_avgs, by='category_name') %>%
    left_join(goal_avgs, by='log_goal') %>%
    left_join(blurb_avgs, by='blurb_length') %>%
    left_join(deadline_avgs, by='days_to_deadline') %>%
    mutate(pred = mu + b_y + b_m + b_d + b_co + b_cat +  b_g + 
             b_bl + b_dd) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, test_set$binary_state))
})
# Plotting the RMSE for each lambda shows us where the minimum RMSE is to be found
qplot(lambdas, rmses)

# So which lambda minimizes the RMSE? 160
lambdas[which.min(rmses)]

#The RMSE returned with a lambda of 160 on the test set is 0.4224825
min(rmses)

# So now we can create training and test sets with the biases added as columns

l <- 160

year_avgs <- train_set %>% 
  group_by(year) %>% 
  summarize(b_y = sum(binary_state - mu)/(n()+l))

month_avgs <- train_set %>% 
  left_join(year_avgs, by='year') %>%
  group_by(month) %>%
  summarize(b_m = sum(binary_state - mu - b_y)/(n()+l))

day_avgs <- train_set %>% 
  left_join(year_avgs, by='year') %>%
  left_join(month_avgs, by='month') %>%
  group_by(day) %>%
  summarize(b_d = sum(binary_state - mu - b_y - b_m)/(n()+l))

country_avgs <- train_set %>% 
  left_join(year_avgs, by='year') %>%
  left_join(month_avgs, by='month') %>%
  left_join(day_avgs, by='day') %>%
  group_by(location_country) %>%
  summarize(b_co = sum(binary_state - mu - b_y - b_m - b_d)/(n()+l))

category_avgs <- train_set %>% 
  left_join(year_avgs, by='year') %>%
  left_join(month_avgs, by='month') %>%
  left_join(day_avgs, by='day') %>%
  left_join(country_avgs, by='location_country') %>%
  group_by(category_name) %>%
  summarize(b_cat = sum(binary_state - mu - b_y - b_m - b_d - b_co)/(n()+l))

goal_avgs <- train_set %>%
  left_join(year_avgs, by='year') %>%
  left_join(month_avgs, by='month') %>%
  left_join(day_avgs, by='day') %>%
  left_join(country_avgs, by='location_country') %>%
  left_join(category_avgs, by='category_name') %>%
  group_by(log_goal) %>%
  summarize(b_g = sum(binary_state - mu - b_y - b_m - b_d - b_co - b_cat)/(n()+l))

blurb_avgs <- train_set %>%
  left_join(year_avgs, by='year') %>%
  left_join(month_avgs, by='month') %>%
  left_join(day_avgs, by='day') %>%
  left_join(country_avgs, by='location_country') %>%
  left_join(category_avgs, by='category_name') %>%
  left_join(goal_avgs, by='log_goal') %>%
  group_by(blurb_length) %>%
  summarize(b_bl = sum(binary_state - mu - b_y - b_m - b_d - b_co - b_cat - b_g)/(n()+l))

deadline_avgs <- train_set %>%
  left_join(year_avgs, by='year') %>%
  left_join(month_avgs, by='month') %>%
  left_join(day_avgs, by='day') %>%
  left_join(country_avgs, by='location_country') %>%
  left_join(category_avgs, by='category_name') %>%
  left_join(goal_avgs, by='log_goal') %>%
  left_join(blurb_avgs, by='blurb_length') %>%
  group_by(days_to_deadline) %>%
  summarize(b_dd = sum(binary_state - mu - b_y - b_m - b_d - b_co - b_cat - b_g - b_bl)/(n()+l))

train_bias <- train_set %>%
  left_join(year_avgs, by='year') %>%
  left_join(month_avgs, by='month') %>%
  left_join(day_avgs, by='day') %>%
  left_join(country_avgs, by='location_country') %>%
  left_join(category_avgs, by='category_name') %>%
  left_join(goal_avgs, by='log_goal') %>%
  left_join(blurb_avgs, by='blurb_length') %>%
  left_join(deadline_avgs, by='days_to_deadline')

test_bias <- test_set %>%
  left_join(year_avgs, by='year') %>%
  left_join(month_avgs, by='month') %>%
  left_join(day_avgs, by='day') %>%
  left_join(country_avgs, by='location_country') %>%
  left_join(category_avgs, by='category_name') %>%
  left_join(goal_avgs, by='log_goal') %>%
  left_join(blurb_avgs, by='blurb_length') %>%
  left_join(deadline_avgs, by='days_to_deadline')

# Let's start, by trying to train a linear model
fitlm <- lm(binary_state ~ b_y + b_m + b_d + b_co + b_cat + b_g + b_bl + b_dd, data=train_bias)
y_hatlm <- predict(fitlm, test_bias)
predictionslm <- ifelse(y_hatlm > 0.5, TRUE, FALSE)
cm_lm <- confusionMatrix(factor(predictions), factor(test_bias$binary_state))
# We see an overall accuracy of 0.7333677

# Now let's try a generalized linear model
fitglm <- glm(binary_state ~ b_y + b_m + b_d + b_co + b_cat + b_g + b_bl + b_dd, data=train_bias, family="binomial")
y_hatglm <- predict(fitglm, newdata=test_bias, type="response")
predictionsglm <- ifelse(y_hatglm > 0.5, TRUE, FALSE)
cm_glm <- confusionMatrix(factor(predictionsglm), factor(test_bias$binary_state))

# We see a slightly lower overall accuracy of 0.7332646

# Now let's try knn
train_bias <- train_bias %>% mutate(binary_state=factor(binary_state))
test_bias <- test_bias %>% mutate(binary_state=factor(binary_state))
fitknn <- train(binary_state ~ b_y + b_m + b_d + b_co + b_cat + b_g + b_bl + b_dd, 
                 method="knn",
                 data=train_bias, 
                 tuneGrid=data.frame(k=seq(3,30,3)))
y_hatknn <- predict(fitknn, newdata=test_bias, type="raw")
predictionsknn <- ifelse(y_hatknn > 0.5, TRUE, FALSE)
cm_knn <- confusionMatrix(factor(predictionsknn), test_bias$binary_state)
# Unfortunately knn did not complete after nearly 18 hours of processing

# Let's try QDA
fitqda <- train(binary_state ~ b_y + b_m + b_d + b_co + b_cat + b_g + b_bl + b_dd, 
                method="qda", data=train_bias)
predictionsqda <- predict(fitqda, newdata=test_bias)
cm_qda <- confusionMatrix(predictionsqda, test_bias$binary_state)

# Again, we see a lower accuracy of 0.7204

# Let's try random forest
fitrf <- train(binary_state ~ b_y + b_m + b_d + b_co + b_cat + b_g + b_bl + b_dd,
               method="rf", data=train_bias)
predictionsrf <- predict(fitrf, newdata=test_bias)
cm_rf <- confusionMatrix(predictionsrf, test_bias$binary_state)

# Unfortunately this didn't complete either.
# Because we're seeing such long training times, let's try with a much smaller training set
# First a training set with just 1% of the training data is created.
set.seed(1, sample.kind="Rounding")
train_index001 <- createDataPartition(
  y = train_bias$binary_state, 
  times = 1, p = 0.01, 
  list = FALSE)
train_bias001 <- train_bias[train_index001,]
train_bias001 <- train_bias001 %>% 
  mutate(binary_state=factor(binary_state)) %>%
  select(b_y, b_m, b_d, b_co, b_cat, b_g, b_bl, 
         b_dd, binary_state)

#Then a training set with 10% of the training data is made.
set.seed(1, sample.kind="Rounding")
train_index01 <- createDataPartition(
  y = train_bias$binary_state, times = 1, 
  p = 0.1, 
  list = FALSE)
train_bias01 <- train_bias[train_index01,]

train_bias01 <- train_bias01 %>% 
  mutate(binary_state=factor(binary_state)) %>% 
  select(b_y, b_m, b_d, b_co, b_cat, b_g, b_bl, 
         b_dd, binary_state)


#Finally a training set with 30% of the training data is put in place.
set.seed(1, sample.kind="Rounding")
train_index03 <- createDataPartition(
  y = train_bias$binary_state, times = 1, 
  p = 0.3, 
  list = FALSE)
train_bias03 <- train_bias[train_index03,]

train_bias03 <- train_bias03 %>% 
  mutate(binary_state=factor(binary_state)) %>% 
  select(b_y, b_m, b_d, b_co, b_cat, b_g, b_bl, 
         b_dd, binary_state)

# Let's try again with a linear model to see the effect on accuracy of using a much smaller training set
fitlm <- lm(binary_state ~ b_y + b_m + b_d + b_co + b_cat + b_g + b_bl + b_dd, data=train_bias001)
y_hatlm <- predict(fitlm, test_bias)
predictionslm <- ifelse(y_hatlm > 0.5, TRUE, FALSE)
cm_lm <- confusionMatrix(factor(predictions), factor(test_bias$binary_state))

# We see an accuracy of 0.7324 - not too different.

# Now let's try again with random forest this time we'll try Rborist as it may be faster
fitrbo <- train(binary_state ~ b_y + b_m + b_d + b_co + b_cat + b_g + b_bl + b_dd,
               method="Rborist", data=train_bias001)
predictionsrbo <- predict(fitrbo, newdata=test_bias)
cm_rbo <- confusionMatrix(predictionsrbo, factor(test_bias$binary_state))

# So this was quick, but gave us an accuracy of 0.6 - not very good. Let's try with 10x more data
fitrbo <- train(binary_state ~ b_y + b_m + b_d + b_co + b_cat + b_g + b_bl + b_dd,
                method="Rborist", data=train_bias01)
predictionsrbo <- predict(fitrbo, newdata=test_bias)
cm_rbo <- confusionMatrix(predictionsrbo, factor(test_bias$binary_state))

# Hmmm, now we get an accuracy of 0.5968 - even worse

# Let's see whether knn is any better

fitknn <- train(binary_state ~ b_y + b_m + b_d + b_co + b_cat + b_g + b_bl + b_dd, 
                method="knn",
                data=train_bias001, 
                tuneGrid=data.frame(k=7))
predictionsknn <- predict(fitknn, newdata=test_bias, type="raw")
cm_knn <- confusionMatrix(predictionsknn, factor(test_bias$binary_state))

#So we see an accuracy of 0.7025. Let's try to tune k
fitknn <- train(binary_state ~ b_y + b_m + b_d + b_co + b_cat + b_g + b_bl + b_dd, 
                method="knn",
                data=train_bias001, 
                tuneGrid=data.frame(k=seq(30,100,10)))
# We find the best k value is 50
predictionsknn <- predict(fitknn, newdata=test_bias, type="raw")
cm_knn <- confusionMatrix(predictionsknn, factor(test_bias$binary_state))

# Now we find a prediction accuracy of 0.7258 - still not as good as a linear model, but let's try to train now on a larger dataset
fitknn <- train(binary_state ~ b_y + b_m + b_d + b_co + b_cat + b_g + b_bl + b_dd, 
                method="knn",
                data=train_bias01, 
                tuneGrid=data.frame(k=50))

predictionsknn <- predict(fitknn, newdata=test_bias, type="raw")
cm_knn <- confusionMatrix(predictionsknn, factor(test_bias$binary_state))

# Now we find a prediction accuracy of 0.7301 - still not as good as a linear model, but let's try to train now on a larger dataset again
fitknn <- train(binary_state ~ b_y + b_m + b_d + b_co + b_cat + b_g + b_bl + b_dd, 
                method="knn",
                data=train_bias03, 
                tuneGrid=data.frame(k=50))

predictionsknn <- predict(fitknn, newdata=test_bias, type="raw")
cm_knn <- confusionMatrix(predictionsknn, factor(test_bias$binary_state))

# So now we get 0.7317 accuracy. 

# Let's have another go at random forest using ranger
fitra <- train(binary_state ~ b_y + b_m + b_d + b_co + b_cat + b_g + b_bl + b_dd,
                method="ranger", data=train_bias01, num.trees=50)
predictionsra <- predict(fitra, newdata=test_bias)
cm_ra <- confusionMatrix(predictionsra, factor(test_bias$binary_state))

# This gives us an accuracy of 0.7315 pretty quickly. Let's increase the number of trees
fitra <- train(binary_state ~ b_y + b_m + b_d + b_co + b_cat + b_g + b_bl + b_dd,
               method="ranger", data=train_bias01, num.trees=200)
predictionsra <- predict(fitra, newdata=test_bias)
cm_ra <- confusionMatrix(predictionsra, factor(test_bias$binary_state))
# This gives us 0.7359 - better than the linear model!
# Let's see if using more trees again improves the result
fitra <- train(binary_state ~ b_y + b_m + b_d + b_co + b_cat + b_g + b_bl + b_dd,
               method="ranger", data=train_bias01, num.trees=500)
predictionsra <- predict(fitra, newdata=test_bias)
cm_ra <- confusionMatrix(predictionsra, factor(test_bias$binary_state))
# This gives us an accuracy of 0.7365. Let's see what training on a larger data set gives us
fitra <- train(binary_state ~ b_y + b_m + b_d + b_co + b_cat + b_g + b_bl + b_dd,
               method="ranger", data=train_bias03, num.trees=500)
predictionsra <- predict(fitra, newdata=test_bias)
cm_ra <- confusionMatrix(predictionsra, factor(test_bias$binary_state))
# We now get an accuracy of 0.7437 That's a significant improvement.

# Now let's see if we can't get better results with a neural net.
fitnn <- train(binary_state ~ b_y + b_m + b_d + b_co + b_cat + b_g + b_bl + b_dd,
               method="nnet", data=train_bias001)
predictionsnn <- predict(fitnn, newdata=test_bias)
cm_nn <- confusionMatrix(predictionsnn, factor(test_bias$binary_state))

# Even on a very small data set, we get promising results: an accuracy of 0.7322

# Let's try on a larger training set
fitnn <- train(binary_state ~ b_y + b_m + b_d + b_co + b_cat + b_g + b_bl + b_dd,
               method="nnet", data=train_bias01)
predictionsnn <- predict(fitnn, newdata=test_bias)
cm_nn <- confusionMatrix(predictionsnn, factor(test_bias$binary_state))

# So now we get an accuracy of 0.7347 - once again, better than a linear model.
# Let's try again on yet another larger training data set
fitnn <- train(binary_state ~ b_y + b_m + b_d + b_co + b_cat + b_g + b_bl + b_dd,
               method="nnet", data=train_bias03)
predictionsnn <- predict(fitnn, newdata=test_bias)
cm_nn <- confusionMatrix(predictionsnn, factor(test_bias$binary_state))
# This time, we get an accuracy of 0.7339

# Now let's try to build an ensemble prediction
ensemble <- (as.numeric(predictionslm) + (as.numeric(predictionsknn)-1) + (as.numeric(predictionsra)-1) + (as.numeric(predictionsnn)-1)) / 4
ensemble <- ifelse(ensemble > 0.5, TRUE, FALSE)
ensemble <- factor(ensemble)
cm_en <- confusionMatrix(ensemble, factor(test_bias$binary_state))

#This ensemble only gives us an accuracy of 0.7371 - less than the random forest on it's own

# A table of the outputs on the test data
knitr::kable(
  data.frame(Models=c("Linear Regression", "Generalized Linear Model", "Quadratic Discriminant Analysis", "Linear Regression (1% of training data)", "Rborist (1% of training data)", "Rborist (10% of training data)", "K nearest neighbours (1% of training data, k=7", "K nearest neighbours (1% of training data, k=50)", "K nearest neighbours (10% of training data, k=50)", "K nearest neighbours (30% of training data, k=50)", "Ranger (10% of training data, 50 trees)", "Ranger (10% of training data, 200 trees)", "Ranger (10% of training data, 500 trees)", "Ranger (30% of training data, 500 trees)", "Neural net (1% of training data)", "Neural net (10% of training data)", "Neural net (30% of training data)", "Ensemble (Linear, KNN, Ranger, Neural net)"), Accuracy=c(0.7333677, 0.7332646, 0.7204, 0.7324, 0.6, 0.5968, 0.7025, 0.7258, 0.7301, 0.7317, 0.7315, 0.7359, 0.7365, 0.7437, 0.7322, 0.7347, 0.7339, 0.7371))
)

# Now test on the validation set. So we join the boas terms to the validation data
validation_bias <- validation_clean %>%
  left_join(year_avgs, by='year') %>%
  left_join(month_avgs, by='month') %>%
  left_join(day_avgs, by='day') %>%
  left_join(country_avgs, by='location_country') %>%
  left_join(category_avgs, by='category_name') %>%
  left_join(goal_avgs, by='log_goal') %>%
  left_join(blurb_avgs, by='blurb_length') %>%
  left_join(deadline_avgs, by='days_to_deadline')

# Finally we make a final prediction - result: 0.7427
predictionsra_v <- predict(fitra, newdata=validation_bias)
cm_ra_v <- confusionMatrix(
  predictionsra_v, 
  factor(validation_bias$binary_state)
)
