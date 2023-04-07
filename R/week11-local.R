# Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
require(haven)
library(tidyverse)
library(caret)
library(parallel)
library(doParallel)
library(tictoc)

## Setting seed for reproducibility
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
local_cluster = makeCluster(7) # My computer has 8 cores  
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




# Publication- NEED TO ADD STUFF 

## Answers to Questions:
### 1. Which models benefited most from parallelization and why?
#### Based on my computation of percentage change in run-time, xgbTree 
#### benefited most from parallelization, followed by elastic net. This is 
#### likely because [BLAH BLAH BLAH]

### 2. How big was the difference between the fastest and slowest parallelized model? Why?
#### The fastest model was glmnet at 1.86s and the slowest model was 
#### random forest at 32.98s, their difference difference is 31.12s. This is 
#### likely because ????

### 3. If your supervisor asked you to pick a model for use in a production model, which would you recommend and why? Consider both Table 1 and Table 2 when providing an answer.

results = function(train_mod){
  algo = train_mod$method
  cv_rsq = str_remove(format(round(max(train_mod$results$Rsquared), 2), nsmall = 2), "^\\d")
  preds = predict(train_mod, gss_tbl_test, na.action = na.pass)
  ho_rsq = str_remove(format(round(cor(preds, gss_tbl_test$workhours)^2, 2), nsmall = 2), "^\\d")
  return(c("algo" = algo, "cv_rsq" = cv_rsq, "ho_sq" = ho_rsq))
}

table1_tbl = as_tibble(t(sapply(mod_ls, results))) # NEED TO FIGURE OUT IF USING ORIGINAL OR NEW MODEL LIST

## New tibble with training time for all eight models 
table2_tbl = tibble(algorithm = mod_vec,
                    original = original_time,
                    parallelized = parallel_time)

### Outuput from table2_tbl
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


