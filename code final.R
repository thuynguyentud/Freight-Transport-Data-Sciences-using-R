library(readxl)
library(caret)
library(mlr3)
library(mlr3verse)
library(mlr3learners)
library(dplyr)
library(tidyverse)
library(ggplot2)
library(ggcorrplot)
library(reshape2)
library(sf)
library(plotly)
library(forecast)
library(GGally)

set.seed(123) #Reproducibility

#import data
dataset2 <- read_excel("D:/5. Study at TUD/1. SoSe 23.24/3. Application in DA/Input_data.xlsx")

#Plot total volume by goods type by years
mainset$Volume <- as.numeric(mainset$Volume)  # Ensure Volume is numeric, handling NAs
mainset$Volume[is.na(mainset$Volume)] <- 0    # Replace NA with 0




dataset2$Volume  =  gsub("-", NA, dataset2$Volume)  # Replace '-' with NA


#preprocess data by years:
yearly_totals <- dataset2 %>%
  group_by(Years) %>%
  summarise(Total_Volume = sum(Volume, na.rm = TRUE))  # Sum up Volume, removing NAs if any remain

#Plot total volume by years
ggplot(yearly_totals, aes(x = Years, y = Total_Volume)) +
  geom_col(fill = "steelblue") +  # Using geom_col to create a bar chart
  labs(title = "Total Volume of Goods Transported by Year",
       x = "Year",
       y = "Total Volume") +
  theme_minimal()
view(mainset)

dataset2$Volume  =  as.numeric(dataset2$Volume)  #convert target colume to numeric
dataset2  =  na.omit(dataset2)
view(dataset2)
categorical_columns <- c("Origin", "Destination", "Goods")  # Adjust this based on your actual column names
dataset2[categorical_columns] <- lapply(dataset2[categorical_columns], factor)
dataset2$Years =  as.integer(as.character(dataset2$Years))
train_data2 = subset(dataset2, Years <= 2021)
test_data2 = subset(dataset2, Years >= 2022)

task01 = TaskRegr$new(id = "freight_volume", backend = train_data2, target = "Volume")
task02 = TaskRegr$new(id = "freight_volume", backend = test_data2, target = "Volume")

#create list of learners - default hyperparameters
learner_lr = lrn("regr.lm") 
learner_svm = lrn("regr.svm", kernel = "linear") #numerical value only
learner_dt = lrn("regr.rpart")
learner_rf = lrn("regr.ranger")
learner_nn = lrn("regr.nnet", size = 5, decay = 1e-4, maxit = 100) #numerical value only
learner_gb = lrn("regr.xgboost", nrounds = 100, eta = 0.3, max_depth = 6) #numerical value only

#check parameter set of learners
learner_lr$param_set

#measure performance
measures = msrs(c("regr.rmse", "regr.mae", "regr.mape", "regr.rsq"))

#train and predict and measure - linear regression
learner_lr$train(task01)
prediction_lr = learner_lr$predict(task02)
prediction_lr$score(measures)

#train and predict and measure - gb


learner_nn$train(task01)
prediction_nn = learner_nn$predict(task02)
prediction_nn$score(measures)

#train and predict and measure - random forest
learner_rf$train(task01)
prediction_rf = learner_rf$predict(task02)
prediction_rf$score(measures)

#resampling
rsmp_kf= rsmp("cv", folds = 10L)
rsmp_kf$instantiate(task01)

rr_rf = resample(task01, learner_rf, rsmp_kf)
as.data.table(rr_rf)[, list(rr_rf, iteration, prediction)]

rr_rf$aggregate(measures)





