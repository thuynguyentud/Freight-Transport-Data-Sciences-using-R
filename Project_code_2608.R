set.seed(123) #global seed for reproducibility, set seed again everytime training models

library(caret)
library(dplyr)
library(forecast)
library(GGally)
library(ggcorrplot)
library(ggplot2)
library(gridExtra)
library(iml)
library(mlr3)
library(mlr3filters)
library(mlr3fselect)
library(mlr3learners)
library(mlr3tuning)
library(mlr3verse)
library(mlr3viz)
library(paradox)
library(plotly)
library(readxl)
library(reshape2)
library(sf)
library(stargazer)
library(stars)
library(tidyverse)
library(xgboost)

### import given main dataset: transport performance
dataset_1 <- read_excel("D:/5. Study at TUD/1. SoSe 23.24/3. Application in DA/ADA_rail_transport/set1_transport_performance.xlsx")

#transform matrix data to standard tabular data and filter unnecessary column
dataset_1 <- dataset_1 %>%
  pivot_longer(cols = -c(Years, Goods, Origin), names_to = "Destination", values_to = "Tpt_perfm")
dataset_1 <- dataset_1 %>% filter(!(Origin == "Total" | Destination == "Total" | Goods == "Total")) %>% 
  filter(!(Origin == "Foreign_countries" | Destination == "Foreign_countries"))
#preprocess tpt_perfm values
dataset_1$Tpt_perfm <- as.character(dataset_1$Tpt_perfm)  
dataset_1$Tpt_perfm[dataset_1$Tpt_perfm == "-"] <- NA  
dataset_1$Tpt_perfm <- as.numeric(dataset_1$Tpt_perfm) 

### 3.1. Pick out the most important goods for analysis
#filter out unidentified goods or goods not specific to industry
excluded_goods <- c("Unidentifiable goods in containers or swap bodies", 
                    "Other unidentifiable goods",
                    "Other waste and secondary raw materials")
dataset_1 <- dataset_1 %>% filter(!Goods %in% excluded_goods)

#Summarize transport performance by types of goods
sum_by_goods <- dataset_1 %>% group_by(Goods) %>%
  summarise(Tpt_perfm = sum(Tpt_perfm, na.rm = TRUE)) %>% arrange(desc(Tpt_perfm))

#Get the top 10 goods with largest total transport performance
top_10_goods <- sum_by_goods %>% arrange(desc(Tpt_perfm))%>%  pull(Goods) %>% .[1:10]

#Group remaining goods as "Other goods"
sum_by_goods <- sum_by_goods %>%
  mutate(Goods = ifelse(Goods %in% top_10_goods, Goods, "Other goods")) %>%
  group_by(Goods) %>% summarise(Tpt_perfm = sum(Tpt_perfm, na.rm = TRUE)) %>% ungroup()

# Calculate percentage contribution of other goods to total volumes
total_tpt_perfm <- sum(sum_by_goods$Tpt_perfm)
sum_by_goods <- sum_by_goods %>% mutate(Percentage = (Tpt_perfm / total_tpt_perfm) * 100)

# Reorder the Goods factor by Tpt_perfm for plotting
sum_by_goods$Goods <- factor(sum_by_goods$Goods, 
                             levels = sum_by_goods$Goods[order(-sum_by_goods$Tpt_perfm)])

# Plot the column chart - figure 3.1 in paper
fig3.1 <- ggplot(sum_by_goods, aes(x = Goods, y = Percentage, fill = Goods)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Transport Performance of goods categories by Percentage",
       x = "Goods",
       y = "Percentage of Total Transport Performance") +
  geom_text(aes(label = sprintf("%.2f%%", Percentage)), vjust = -0.5) + 
  theme(axis.text.x = element_blank())
# dev.off() -> run this code if there's error "invalid graphics state"
ggsave("fig3.1.png", plot = fig3.1, width = 10, height = 5, dpi = 300)

#Filter only top 10 goods for further analysis
dataset_1 <- dataset_1 %>% filter(Goods %in% top_10_goods)


### import additional predictors data
#import and combine regional GDP data columns
dataset_3 <- read_excel("D:/5. Study at TUD/1. SoSe 23.24/3. Application in DA/ADA_rail_transport/set3_regional_gdp.xlsx")
# Merge GDP data for Origin region
main_data <- dataset_1 %>%
  left_join(dataset_3, by = c("Origin" = "Region_NUTS_2")) %>%
  rename(gdp_origin = gdp_region)
# Merge GDP data for Destination region
main_data <- main_data %>%
  left_join(dataset_3, by = c("Destination" = "Region_NUTS_2")) %>%
  rename(gdp_destination = gdp_region)

#import and combine other yearly predictors data
dataset_4 <- read_excel("D:/5. Study at TUD/1. SoSe 23.24/3. Application in DA/ADA_rail_transport/set4_yearly_features.xlsx")
main_data <- merge(main_data, dataset_4, by = "Years")

#import and combine rail volume in tons data 
dataset_2 <- read_excel("D:/5. Study at TUD/1. SoSe 23.24/3. Application in DA/ADA_rail_transport/set2_rail_volume_tons.xlsx")
dataset_2 <- dataset_2 %>% filter(!Goods == "Containers and swap bodies in service, empty") #filter out irrelevant goods
main_data <- full_join(main_data, dataset_2, by = c("Years", "Goods", "Origin", "Destination"))
main_data$total_vol_ton <- as.numeric(main_data$total_vol_ton)

# View the full dataset
head(main_data)


### 3.2. Descriptive statistics of the data set
## summary statistics table
summary(main_data)
main_data <- as.data.frame(main_data)
stargazer(main_data, type = "text") #figure 3.1 in paper

# boxplot to check distribution of all numerical variables 
#figure 3.3 in paper
par(mfrow=c(1, 2))
boxplot(main_data$Tpt_perfm,main="Transport performance", xlab = "", ylab="value")
boxplot(main_data$total_vol_ton,main="Rail volume intons", xlab = "", ylab="value")
par(mfrow=c(1, 1))

#use value from original dataset for yearly additional predictors due to different in yearly observation in full dataset
#figure 3.4 in paper
par(mfrow=c(2, 3))
boxplot(dataset_4$year_rail_volume,main="Rail national volume", xlab = "", ylab="value")
boxplot(dataset_4$year_road_volume,main="Road nationalvolume", xlab = "", ylab="value")
boxplot(dataset_4$year_water_volume,main="Inland waterway national volume", xlab = "", ylab="value")
boxplot(dataset_3$gdp_region,main="Regional GDP", xlab = "", ylab="value")
boxplot(dataset_4$pro_ind_energy,main="Production index - energy industry", xlab = "", ylab="value")
boxplot(dataset_4$pro_ind_mining,main="Production index - mining & quarrying", xlab = "", ylab="value")
par(mfrow=c(1, 1))


##3.3. Correlation matrix - figure 3.5 in paper
numeric_data <- main_data %>% select_if(is.numeric)
cor_matrix <- cor(numeric_data, use = "complete.obs")  # 'use' parameter to handle missing values
melted_cor_matrix <- melt(cor_matrix)

fig3.5 <- ggplot(melted_cor_matrix, aes(Var1, Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0, limit = c(-1,1), space = "Lab", name="Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, size = 12, hjust = 1),
        axis.text.y = element_text(size = 12)) +
  labs(x = "", y = "", title = "Correlation Matrix") +
  coord_fixed()

ggsave("fig3.5.png", plot = fig3.5, width = 10, height = 10, dpi = 300)

##3.4. Inspect temporal aspect
#Check change in transport performance by goods over 13 years 
#figure 3.6 in paper
yearly_by_goods <- main_data %>%
  group_by(Years,Goods) %>%
  summarise(Total_Value = sum(Tpt_perfm, na.rm = TRUE), .groups = 'drop')

fig3.6 <- ggplot(yearly_by_goods, aes(x = Years, y = Total_Value, group = 1)) +
  geom_line(size = 1) +
  geom_point() +
  facet_wrap(~ Goods, ncol = 5, labeller = label_wrap_gen(width = 20)) +
  labs(title = "Transport performance by Goods",
       x = "Years",
       y = "Transport performance") +
  theme_minimal() +
  theme(axis.text.x = element_text(size = 8, angle = 45, hjust = 1),
        axis.text.y = element_text(size = 9, hjust = 1),
        strip.text = element_text(size = 11))

ggsave("fig3.6.png", plot = fig3.6, width = 12, height = 5, dpi = 300)

##3.5. Inspect spatial aspect
#Read and plot map of Germany level NUTS-2 administration
geo_data <- st_read("D:/5. Study at TUD/1. SoSe 23.24/3. Application in DA/ADA_rail_transport/3_mittel.geo.json")
plot(st_geometry(geo_data))
ggplot(data = geo_data) +
  geom_sf() +
  theme_minimal()

#rename column name of NUTS-2 region in geo data
geo_data <- geo_data %>% 
  rename(Origin = NAME_2 )

#calculates the total tpt_perfm of goods for each combination of these following categories
comb1 <- main_data %>%
  group_by(Origin, Goods, Years) %>%
  summarise(Total_TKM = sum(Tpt_perfm, na.rm = TRUE), .groups = 'drop')

#rename column "Destination" in geo data
geo_data <- geo_data %>% 
  mutate (Destination = Origin)

#calculates the total volume of goods for each combination of these following categories
comb2 <- main_data %>%
  group_by(Destination, Goods, Years) %>%
  summarise(Total_TKM = sum(Tpt_perfm, na.rm = TRUE), .groups = 'drop')
print(comb2)

## figure 3.7 in paper
#pick 1 goods and 1 year for plotting
chosen_goods_1 <- "Basic iron and steel and ferro-alloys etc."
chosen_year_1 <- 2023

#ensure using the same scale for plots over 13 years for 1 good
scale_range_ori <- comb1 %>%  filter(Goods == chosen_goods_1 )
scale_range_de <- comb2 %>% filter(Goods == chosen_goods_1)
common_limits <- range(c(scale_range_ori$Total_TKM, scale_range_de$Total_TKM), na.rm = TRUE)

# Filter for Origin for specific year
geo_data_ori_1 <- scale_range_ori %>% filter(Years == chosen_year_1)
geo_data_ori_1 <- full_join(geo_data, geo_data_ori_1, by = "Origin")

# Filter for Destination by year (here: year is 2023)
geo_data_de_1 <- scale_range_de %>%  filter(Years == chosen_year_1)
geo_data_de_1 <- full_join(geo_data, geo_data_de_1, by = "Destination")

# Create maps 
map1 <- ggplot(data = geo_data_ori_1) +
  geom_sf(aes(fill = Total_TKM)) +
  scale_fill_gradient(low = "lightblue", high = "red", limits = common_limits) +  
  labs(title = paste("Transport performance by region of", chosen_goods_1, "in", chosen_year_1),
       subtitle = "Origin", fill = "Ton km") +
  theme_minimal() 

map2 <- ggplot(data = geo_data_de_1) +
  geom_sf(aes(fill = Total_TKM)) +
  scale_fill_gradient(low = "lightblue", high = "red", limits = common_limits) +  
  labs(title = "", subtitle = "Destination", fill = "Ton km") +
  theme_minimal()

#combine 2 maps into 1 - figure 3.7 in paper
fig3.7 <- grid.arrange(map1, map2, ncol = 2)
ggsave("fig3.7.png", plot = fig3.7, width = 10, height = 6, dpi = 300)

## figure 3.8 in paper
# Calculate the spatial difference from the previous year for specific good and year
#pick types of goods and year for plot
chosen_goods_2 <- "Basic iron and steel and ferro-alloys etc."
chosen_year_2 <- 2023
#For origin
diff_ori <- comb1 %>% filter(Goods == chosen_goods_2) %>% 
  arrange(Origin, Years) %>% group_by(Origin) %>% 
  mutate(Difference = Total_TKM - lag(Total_TKM)) # Calculate absolute difference
#For destination
diff_de <- comb2 %>% filter(Goods == chosen_goods_2) %>% 
  arrange(Destination, Years) %>% group_by(Destination) %>%       
  mutate(Difference = Total_TKM - lag(Total_TKM)) # Calculate absolute difference
# Display results
print(diff_de)
#Ensure using the same scale for plots
common_limits <- range(c(diff_ori$Difference, diff_de$Difference), na.rm = TRUE)
# Filter for Origin
filtered_data_ori <- diff_ori %>%
  filter(Years == chosen_year_2)
merged_dt_or <- full_join(geo_data, filtered_data_ori, by = "Origin")

# Filter for Destination
filtered_data_de <- diff_de %>%
  filter(Years == chosen_year_2)
merged_dt_de <- full_join(geo_data, filtered_data_de, by = "Destination")

# Create maps with different colors for negative and positive differences
map3 <- ggplot(data = merged_dt_or) +
  geom_sf(aes(fill = Difference)) +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", 
                       midpoint = 0, limits = common_limits, 
                       na.value = "grey50") +
  labs(title = paste(chosen_goods_2, ": Change from previous year to", chosen_year_2),
       subtitle = "Origin", fill = "Difference") +
  theme_minimal() 

map4 <- ggplot(data = merged_dt_de) +
  geom_sf(aes(fill = Difference)) +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", 
                       midpoint = 0, limits = common_limits, 
                       na.value = "grey50") +
  labs(title = "", subtitle = "Destination", fill = "Difference") +
  theme_minimal()
#combine 2 maps into 1 - figure 3.8 in paper
fig3.8 <- grid.arrange(map3, map4, ncol = 2)
ggsave("fig3.8.png", plot = fig3.8, width = 10, height = 6, dpi = 300)

# Heat map plot - figure 3.9 in paper
# Specify the goods of interest
chosen_goods_3 <- "Coal and lignite"
# Prepare the data
df_filtered <- main_data %>%
  filter(Goods == chosen_goods_3, Tpt_perfm > 0) %>%  
  group_by(Years, Origin, Destination) %>%
  summarise(Count = n(), .groups = 'drop') %>%  # Count occurrences
  group_by(Origin, Destination) %>%
  summarise(TotalCount = sum(Count), .groups = 'drop')  # Sum occurrences over all years

# Arrange Origin and Destination alphabetically for the matrix
df_matrix <- df_filtered %>% arrange(Origin, Destination) %>%
  pivot_wider(names_from = Destination, values_from = TotalCount, values_fill = list(TotalCount = 0)) 

# Convert to long format for ggplot2
df_long <- melt(df_matrix, id.vars = 'Origin', variable.name = 'Destination', value.name = 'Count')

# Plotting the data as a heatmap
fig3.9 <- ggplot(data = df_long, aes(x = Destination, y = Origin, fill = Count)) +
  geom_tile() +  # Create tiles for each origin-destination pair
  scale_fill_gradient(low = "white", high = "blue") +  
  labs(title = paste("Active Transport Routes for", chosen_goods_3, "Across 13 Years"),
       x = "Destination", y = "Origin", fill = "Route Count") +
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))  

# Plot heatmap as figure 3.9 in paper
print(fig3.9)
ggsave("fig3.9.png", plot = fig3.9, width = 10, height = 10, dpi = 300)


### 4. Developing machine learning model
## 4.1 Data preprocessing
# filter only routes that consistently operate all 13 years
maindata1 <- main_data %>% group_by(Goods, Origin, Destination) %>%
  mutate(Years_with_Volume = sum(Tpt_perfm > 0, na.rm = TRUE))

maindata1 <- maindata1 %>% filter(Years_with_Volume == 13)

#delete some irrelevant and correlated features
maindata1 <- subset(maindata1, select = -Years_with_Volume)
maindata1 <- subset(maindata1, select = -Region_NUTS_1.x)
maindata1 <- subset(maindata1, select = -Region_NUTS_1.y)
maindata1 <- subset(maindata1, select = -pro_ind_energy) #correlated feature

#delete missing values
maindata1  =  na.omit(maindata1)

# split train data (before 2022), and test data (2022 and 2023)
data_train_01 <- maindata1 %>% mutate(Years = as.numeric(Years)) %>%
  filter(Years <= 2021) 
data_test_01 <- maindata1 %>% mutate(Years = as.numeric(Years)) %>%
  filter(Years >= 2022) 

# Adjust categories column to type factor
categorical_variables <- c("Origin", "Destination", "Goods")
data_train_01[categorical_variables] <- lapply(data_train_01[categorical_variables], factor)
data_test_01[categorical_variables] <- lapply(data_test_01[categorical_variables], factor)

# One-hot encode categorical variables and preprocess column names
encode_and_clean_names <- function(data) {
  data %>%
    model.matrix(~ . - 1, data = .) %>%
    as.data.frame() %>%
    rename_all(~ gsub(" ", "_", .)) %>% # Replace spaces with underscores
    rename_all(~ gsub("[^[:alnum:]_]", "", .)) %>% 
    rename_all(~ gsub("ü", "ue", .)) %>% #some Germany regions have umlaut in names
    rename_all(~ gsub("ä", "ae", .)) %>%
    rename_all(~ gsub("ö", "oe", .)) %>%
    rename_all(~ gsub("ß", "ss", .))}

# seperate train-test sets for one-hot encoding categorical values 
data_train_encoded <- data_train_01 %>% encode_and_clean_names()
data_test_encoded <- data_test_01 %>% encode_and_clean_names()


## 4.2. Preliminary model selection and validation
#create tasks
#keep seperate tasks for easier interpretation with each type of learners
task_train_01 = TaskRegr$new(id = "freight_volume_train", backend = data_train_01, target = "Tpt_perfm")
task_test_01 = TaskRegr$new(id = "freight_volume_test", backend = data_test_01, target = "Tpt_perfm")
#to serve separate learners which only work with numerical values
task_train_encoded = TaskRegr$new(id = "freight_volume_train", backend = data_train_encoded, target = "Tpt_perfm")
task_test_encoded = TaskRegr$new(id = "freight_volume_test", backend = data_test_encoded, target = "Tpt_perfm")

# set of measure performance metrics
measures = msrs(c("regr.rmse", "regr.mae", "regr.mape", "regr.rsq"))

#create list of learners
learner_lr = lrn("regr.lm") 
learner_dt = lrn("regr.rpart")
learner_rf = lrn("regr.ranger", importance = "permutation")

learner_svm = lrn("regr.svm") #numerical value only
learner_nn = lrn("regr.nnet", size = 10, decay = 1e-4, maxit = 100) #numerical value only
learner_gb = lrn("regr.xgboost", nrounds = 100, eta = 0.3, max_depth = 9) #numerical value only

#train learners
learner_lr$train(task_train_01)
learner_dt$train(task_train_01)
learner_rf$train(task_train_01)

learner_svm$train(task_train_encoded)
learner_nn$train(task_train_encoded)
learner_gb$train(task_train_encoded)

#predict test data
prediction_lr = learner_lr$predict(task_test_01)
prediction_dt = learner_dt$predict(task_test_01)
prediction_rf = learner_rf$predict(task_test_01)

prediction_svm = learner_svm$predict(task_test_encoded)
prediction_nn = learner_nn$predict(task_test_encoded)
prediction_gb = learner_gb$predict(task_test_encoded)

#measure performances ~ table 4.1 in paper
prediction_lr$score(measures)
prediction_dt$score(measures)
prediction_rf$score(measures)

prediction_svm$score(measures)
prediction_nn$score(measures)
prediction_gb$score(measures)


### 4.3. Nested resampling and hyperparameter tuning
# check hyperparameters of 2 learners
as.data.table(learner_rf$param_set)[, .(id, class, lower, upper, nlevels)] #check param set
as.data.table(learner_gb$param_set)[, .(id, class, lower, upper, nlevels)] #check param set

# Define the common outer resampling strategy 
rsmp_cv3 = rsmp("cv", folds = 3L) 

## Nested resampling and hyperparameter tuning ~ random forest 
#Create auto tuner
at_rf = auto_tuner(
  tuner = tnr("random_search"),
  learner = lrn("regr.ranger", importance = "permutation",
                num.trees = to_tune(100, 1000), #table 4.2 in paper
                mtry = to_tune(3, 10),
                min.node.size = to_tune(1, 20),
                max.depth  = to_tune(1, 30)),
  resampling = rsmp("cv", folds = 3L),
  measure = msr("regr.mae"),
  terminator = trm("run_time", secs = 600))

#Execute nested resampling
set.seed(123) #Reproducibility for random_search
rr_tuning_rf = resample(task = task_train_01, at_rf, rsmp_cv3, store_models = TRUE)

#show performance for each outer iterations
rr_tuning_rf$score(msr("regr.mae"))

#Extract tuning results and find best configuration
inner_archives_rf = extract_inner_tuning_archives(rr_tuning_rf)
best_param_rf <- inner_archives_rf[order(regr.mae)][1, .(num.trees = x_domain_num.trees, mtry = x_domain_mtry, min.node.size = x_domain_min.node.size, max.depth = x_domain_max.depth)]

#Print the best hyperparameter configuration for Random Forest
print(best_param_rf) #table 4.2 in paper

#Create a new Random Forest learner with the optimal configuration.
tuned_rf <- lrn("regr.ranger")
tuned_rf$param_set$values <- as.list(best_param_rf)

##Nested tuning takes 40 minutes, use code line 449 -> 453 (result of optimal configuration) to check results fast and run later plot if needed
# tuned_rf <- lrn("regr.ranger")
# tuned_rf$param_set$values$num.trees <- 140
# tuned_rf$param_set$values$mtry <- 8
# tuned_rf$param_set$values$min.node.size <- 2
# tuned_rf$param_set$values$max.depth <- 22

#Train model with tuned learners
set.seed(123)
tuned_rf$train(task_train_01)
#Predict and score predictions 
pred_rf_tuned = tuned_rf$predict(task_test_01)
pred_rf_tuned$score(measures) #table 4.3 in paper


## Nested resampling and hyperparameter tuning ~ XGB
#Create auto tuner
at_gb = auto_tuner(
  tuner = tnr("random_search"),
  learner = lrn("regr.xgboost",
                nrounds = to_tune(100, 1000), #table 4.2 in paper
                eta = to_tune(0.01, 0.3),
                max_depth = to_tune(3, 10),
                subsample = to_tune(0.5, 1),
                colsample_bytree = to_tune(0.3, 1)),
  resampling = rsmp("cv", folds = 3L),
  measure = msr("regr.mae"),
  terminator = trm("run_time", secs = 600))

#Execute nested resampling
set.seed(123)
rr_tuning_gb = resample(task = task_train_encoded, at_gb, rsmp_cv3, store_models = TRUE)

#show performance for each outer iterations
rr_tuning_gb$score(msr("regr.mae"))
#Extract tuning results and find best configuration
inner_archives_gb = extract_inner_tuning_archives(rr_tuning_gb)
print(colnames(inner_archives_gb))
best_param_gb <- inner_archives_gb[order(regr.mae)][1, .(nrounds = x_domain_nrounds, eta = x_domain_eta, max_depth = x_domain_max_depth,
                                                         subsample = subsample, colsample_bytree = colsample_bytree)]

#Print the best hyperparameter configuration for GB
print(best_param_gb) #table 4.2 in paper

#Create a new XGB learner with the optimal configuration.
tuned_gb <- lrn("regr.xgboost")
tuned_gb$param_set$values <- as.list(best_param_gb)
tuned_gb$param_set$values$early_stopping_set <- "none"

#Nested tuning take 40 minutes, use code line 498 -> 503 (result of optimal configuration) to check results fast and run later plot if needed
# tuned_gb <- lrn("regr.xgboost")
# tuned_gb$param_set$values$nrounds <- 319
# tuned_gb$param_set$values$eta <- 0.116596
# tuned_gb$param_set$values$max_depth <- 9
# tuned_gb$param_set$values$subsample <- 0.8763066
# tuned_gb$param_set$values$colsample_bytree <- 0.778358

#Train the tuned XGB learners
set.seed(123)
tuned_gb$train(task_train_encoded)

#Predict and score predictions ~ table 4.2 in paper
pred_gb_tuned = tuned_gb$predict(task_test_encoded)
pred_gb_tuned$score(measures) #table 4.3 in paper


### 4.4. Nested resampling and feature selection - XGB 
Create an AutoFSelector for feature selection using a GB learner
auto_fs_gb = AutoFSelector$new(
  learner = tuned_gb,
  resampling = rsmp("cv", folds = 3L),
  measure = msr("regr.mae"),
  fselector = fs("sequential"),
  terminator = trm("run_time", secs = 600))

# Perform nested resampling & feature selection
set.seed(123)
rr_fselect_gb = resample(task_train_encoded, auto_fs_gb, rsmp_cv3, store_models = TRUE)

# Extract performance of each iteration
rr_fselect_gb$score(msr("regr.mae"))

# Extract set of features selected in each iteration of the feature selection process.
fselect_results_gb = lapply(rr_fselect_gb$learners, function(auto_fsel) {
  auto_fsel$model$fselect_instance$result_feature_set})
fselect_results_gb

# Display the selected features from the fold with lowest mae
selected_features_gb <- fselect_results_gb[[2]]
selected_features_gb

#Nested fselection take 40 minutes, use code line 540 (result of optimal subset) to check results and run later plot if needed
# selected_features_gb <- c("OriginSaarland", "total_vol_ton" )

#Create new tasks with new sets of features
task_train_fs_gb = TaskRegr$new(id = "freight_volume_train", backend = data_train_encoded, target = "Tpt_perfm")
task_test_fs_gb = TaskRegr$new(id = "freight_volume_train", backend = data_test_encoded, target = "Tpt_perfm")
task_train_fs_gb$select(selected_features_gb)

#train and predict with new sets of features
set.seed(123)
tuned_gb_2 <- tuned_gb$clone()
tuned_gb_2$train(task_train_fs_gb)
pred_fs_gb = tuned_gb_2$predict(task_test_fs_gb)
pred_fs_gb$score(measures) #table 4.4 in paper


## 4.5.Model inspection and interpretation
## Plot results of 2 best performing model: RF & XGB after tuning
results_tuned_rf = data.frame(actual = pred_rf_tuned$response, predicted = data_test_01$Tpt_perfm)
results_tuned_gb = data.frame(actual = pred_gb_tuned$response, predicted = data_test_encoded$Tpt_perfm)

# Scatter plot of actual vs. predicted values - rf
p1 <- ggplot(results_tuned_rf, aes(x = actual, y = predicted)) +
  geom_point(color = 'blue') +
  geom_abline(slope = 1, intercept = 0, color = 'red', linetype = "dashed") +
  labs(title = "Scatter Plot of Actual vs. Predicted Values - RF",
       x = "Actual Values",
       y = "Predicted Values") +
  theme_minimal()

# Scatter plot of actual vs. predicted values - gb
p2 <- ggplot(results_tuned_gb, aes(x = actual, y = predicted)) +
  geom_point(color = 'blue') +
  geom_abline(slope = 1, intercept = 0, color = 'red', linetype = "dashed") +
  labs(title = "Scatter Plot of Actual vs. Predicted Values - XGB",
       x = "Actual Values",
       y = "Predicted Values") +
  theme_minimal()

#combine 2 plot into 1 image ~ figure 4.1 in paper
fig4.1 <- grid.arrange(p1, p2, ncol = 2)
ggsave("fig4.1.png", plot = fig4.1, width = 20, height = 10, dpi = 300)


## Plot Feature importance of RF ~ figure 4.2 in paper
features_rf = task_test_01$data(cols = task_test_01$feature_names)
target_rf = task_test_01$data(cols = task_test_01$target_names)
predictor_rf = Predictor$new(tuned_rf, data = features_rf, y = target_rf)
set.seed(123)
importance_rf = FeatureImp$new(predictor_rf, loss = "mae", n.repetitions = 5)
fig4.2 <- importance_rf$plot()
ggsave("fig4.2.png", plot = fig4.2, width = 10, height = 5, dpi = 300)

## Plot Feature importance of XGB 
features_gb = task_test_encoded$data(cols = task_test_encoded$feature_names)
target_gb = task_test_encoded$data(cols = task_test_encoded$target_names)
predictor_gb = Predictor$new(tuned_gb, data = features_gb, y = target_gb)
set.seed(123)
importance_gb = FeatureImp$new(predictor_gb, loss = "mae", n.repetitions = 5)

# Extract the feature importance results
feature_importance_df <- importance_gb$results

# Select top 5 most and 5 least important features because there's too many feature to plot
top_5_highest <- feature_importance_df[order(-feature_importance_df$importance), ][1:5, ]
top_5_lowest <- feature_importance_df[order(feature_importance_df$importance), ][1:5, ]

# Combine the top and bottom features into one data frame
combined_features = rbind(top_5_highest, top_5_lowest)

# Create 2 importance plot with same scale
p3 <- ggplot(top_5_highest, aes(x = reorder(feature, importance), y = importance)) +
  geom_col() +
  coord_flip() +
  xlab("Feature") +
  ylab("Importance") +
  ggtitle("Top 5 most important features") +
  ylim(0, max(feature_importance_df$importance))

p4 <- ggplot(top_5_lowest, aes(x = reorder(feature, importance), y = importance)) +
  geom_col() +
  coord_flip() +
  xlab("Feature") +
  ylab("Importance") +
  ggtitle("Top 5 least important features") +
  ylim(0, max(feature_importance_df$importance)) 

# figure 4.3 in paper
fig4.3 <- grid.arrange(p3, p4, nrow = 2)
ggsave("fig4.3.png", plot = fig4.3, width = 10, height = 5, dpi = 300)

## Plot Feature effects
effect_rf = FeatureEffect$new(predictor_rf, feature = "total_vol_ton", method = "pdp+ice")
p5 <- effect_rf$plot() + ggtitle("Volume in tons Effect in Random forest")
effect_gb = FeatureEffect$new(predictor_gb, feature = "total_vol_ton", method = "pdp+ice")
p6 <- effect_gb$plot() + ggtitle("Volume in tons Effect in X. Gradient Boosting")

# figure 4.4 in paper
fig4.4 <- grid.arrange(p5, p6, ncol = 2)
ggsave("fig4.4.png", plot = fig4.4, width = 10, height = 5, dpi = 300)