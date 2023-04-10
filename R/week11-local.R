# Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
require(haven)
library(tidyverse)
library(caret)
library(parallel)
library(doParallel)
library(tictoc)

set.seed(2023)



# Data Import and Cleaning
gss_tbl = read_sav("../data/GSS2016.sav") %>% 
  mutate(across(everything(), as.numeric)) %>% 
  drop_na(MOSTHRS) %>% 
  rename(workhours = MOSTHRS) %>% 
  select(-HRS1, HRS2) %>%
  select(where(function(x) (sum(is.na(x))/nrow(.)) < 0.75))

mod_vec = c("lm", "glmnet", "ranger", "xgbTree")




# Visualization
gss_tbl %>% ggplot(aes(x = workhours)) + 
  geom_histogram(bins = 50) + 
  labs(x = "workhours", y = "Frequency",
       title = "Univariate distribution of workhours")




# Analysis
index = createDataPartition(gss_tbl$workhours, p = 0.75, list = FALSE)
gss_tbl_train = gss_tbl[index,]
gss_tbl_test = gss_tbl[-index,]

training_folds = createFolds(gss_tbl_train$workhours, 10)

reuseControl = trainControl( method = "cv", number = 10, search = "grid", 
                             indexOut = training_folds, verboseIter = TRUE)


## Creating a list of objects for storing models and time elapsed within for loops
mod_ls_original = list()
mod_ls_parallel = list()
original_time = rep(0,4) # create vector to store time output
parallel_time = rep(0,4) # create vector to store time output


## 1. [Original] Training the first four models without parallelization

for(i in 1:length(mod_vec)){
  
  method = mod_vec[i]
  
  if(method == "lm" | method == "glmnet"){
    pre_process = c("center", "scale", "nzv", "medianImpute")
  }else{
    pre_process = "medianImpute"
  }
  
  # Added tic(), toc() to capture time needed to train a model 
  tic()
  mod = train(workhours ~ .,
              data = gss_tbl_train,
              method = method,
              metric = "Rsquared",
              na.action = na.pass,
              trControl = reuseControl,
              preProcess = pre_process)
  
  time_store = toc()
  time_elapsed = time_store$toc - time_store$tic # computing seconds elapsed
  
  original_time[i] = time_elapsed # storing seconds elapsed 
  mod_ls_original[[i]] = mod
  
}

## 2. [Parallel] Training the other four models with parallelization

### Turning on parallel processing
#### Using # of cores minus 1 for cluster to allow other processes on computer
local_cluster = makeCluster(detectCores() - 1)   
registerDoParallel(local_cluster)

for(i in 1:length(mod_vec)){
  
  method = mod_vec[i]
  
  if(method == "lm" | method == "glmnet"){
    pre_process = c("center", "scale", "nzv", "medianImpute")
  }else{
    pre_process = "medianImpute"
  }
  
  # Added tic(), toc() to capture time needed to train a model 
  tic()
  mod = train(workhours ~ .,
              data = gss_tbl_train,
              method = method,
              metric = "Rsquared",
              na.action = na.pass,
              trControl = reuseControl,
              preProcess = pre_process)
  
  time_store = toc()
  time_elapsed = time_store$toc - time_store$tic # computing seconds elapsed
  
  parallel_time[i] = time_elapsed # storing seconds elapsed
  mod_ls_parallel[[i]] = mod
  
}
### Turning off parallel processing
stopCluster(local_cluster)
registerDoSEQ()




# Publication

## Answers to Questions:
### 1. Which models benefited most from parallelization and why?
#### Based on my computation of percentage change in run-time, xgbTree 
#### benefited most from parallelization (78% reduction), followed by elastic 
#### net (69% reduction). This is likely because [BLAH BLAH BLAH]

### 2. How big was the difference between the fastest and slowest parallelized model? Why?
#### The fastest model was glmnet at 1.86s and the slowest model was 
#### random forest at 32.98s, their difference difference is 31.12s. This is 
#### likely because there are more dependency in computations for random forests
#### compared to elastic net. For elastic net, all models could theoretically be 
#### trained and predictions can be computed in parallel at the same time. 
#### Whereas for random forest, the predictions at the end will be computed 
#### based on individual trees (a dependency), making parallelization not 
#### possible at some point of the model training process. Therefore, 
#### training random forest took a lot more time than elastic net even though
#### parallelization still improved model training time for random forest by a little.  

### 3. If your supervisor asked you to pick a model for use in a production model, which would you recommend and why? Consider both Table 1 and Table 2 when providing an answer.
#### For this specific problem, I would recommend XGBoost. Based on Table 1, 
#### xgboost had the highest R-squared values from both 10-fold cross-validation
#### and holdout test validation. The computation time for XGBoost is also 
#### mangeable based on Table 2 where the model training time for XGBoost 
#### is about 130s without parallelization and 30s with parallelization on my local laptop. 



results = function(train_mod){
  algo = train_mod$method
  cv_rsq = str_remove(format(round(max(train_mod$results$Rsquared), 2), nsmall = 2), "^\\d")
  preds = predict(train_mod, gss_tbl_test, na.action = na.pass)
  ho_rsq = str_remove(format(round(cor(preds, gss_tbl_test$workhours)^2, 2), nsmall = 2), "^\\d")
  return(c("algo" = algo, "cv_rsq" = cv_rsq, "ho_sq" = ho_rsq))
}

table1_tbl = as_tibble(t(sapply(mod_ls_original, results))) 
### Output from table1_tbl for later comparison
# algo    cv_rsq ho_sq
# 1 lm      .16    .04  
# 2 glmnet  .86    .55  
# 3 ranger  .93    .57  
# 4 xgbTree .95    .65 

## New tibble with training time for all eight models 
table2_tbl = tibble(algorithm = mod_vec,
                    original = original_time,
                    parallelized = parallel_time)

### Outuput from table2_tbl for later comparison
# algorithm original parallelized
# 1        lm    2.930        2.883
# 2    glmnet    5.991        1.857
# 3    ranger   36.737       32.978
# 4   xgbTree  128.934       28.810


## Q1: Compare run-time across models by calculating percentage change in run time 
table2_tbl %>% group_by(algorithm) %>% 
  summarize(reduction = ((parallelized-original)/original) * 100)

### Percentage change in run time 
# algorithm reduction
# 1 glmnet       -69.0 
# 2 lm            -1.60
# 3 ranger       -10.2 
# 4 xgbTree      -77.7 

## Q2: Difference between fastest and slowest parallelized model
diff(range(table2_tbl$parallelized)) # 31.121s between random forest and glmnet