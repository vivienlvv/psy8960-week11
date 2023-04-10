# Script Settings and Resources
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




# Analysis
index = createDataPartition(gss_tbl$workhours, p = 0.75, list = FALSE)
gss_tbl_train = gss_tbl[index,]
gss_tbl_test = gss_tbl[-index,]

training_folds = createFolds(gss_tbl_train$workhours, 10)

reuseControl = trainControl( method = "cv", number = 10, search = "grid", 
                             indexOut = training_folds, verboseIter = TRUE)


mod_ls_original = list()
mod_ls_parallel = list()
original_time = rep(0,4) 
parallel_time = rep(0,4) 


for(i in 1:length(mod_vec)){
  
  method = mod_vec[i]
  
  if(method == "lm" | method == "glmnet"){
    pre_process = c("center", "scale", "nzv", "medianImpute")
  }else{
    pre_process = "medianImpute"
  }
  
  tic()
  mod = train(workhours ~ .,
              data = gss_tbl_train,
              method = method,
              metric = "Rsquared",
              na.action = na.pass,
              trControl = reuseControl,
              preProcess = pre_process)
  
  time_store = toc()
  time_elapsed = time_store$toc - time_store$tic 
  
  original_time[i] = time_elapsed 
  mod_ls_original[[i]] = mod
  
}

## amdsmall partition has 128 cores available per node and each job 
## has a maximum of 1 node. I will double the # of cores I used in my local 
## laptop when using MSI to avoid taking up too much resources and long queue for my job
local_cluster = makeCluster(14)   
registerDoParallel(local_cluster)

for(i in 1:length(mod_vec)){
  
  method = mod_vec[i]
  
  if(method == "lm" | method == "glmnet"){
    pre_process = c("center", "scale", "nzv", "medianImpute")
  }else{
    pre_process = "medianImpute"
  }
  
  tic()
  mod = train(workhours ~ .,
              data = gss_tbl_train,
              method = method,
              metric = "Rsquared",
              na.action = na.pass,
              trControl = reuseControl,
              preProcess = pre_process)
  
  time_store = toc()
  time_elapsed = time_store$toc - time_store$tic 
  
  parallel_time[i] = time_elapsed 
  mod_ls_parallel[[i]] = mod
  
}

stopCluster(local_cluster)
registerDoSEQ()




# Publication
## 1. Which models benefited most from moving to the supercomputer and why?
### When looking at the % reduction in training time for parallel vs sequential 
### processing, random forest benefited the most moving from local laptop to 
### MSI supercomputer in that its training time reduced by 33% on MSI compared to
### no negligible change on my local laptop when parallelization is turned on. 
### This is likely because the supercomputer used allows double the number of 
### cores needed for more efficient parallel processing for random forest. 
### Similarly, training time reduced by 90% and 78% for XGboost and elastic 
### net with parallelization on MSI compared to only 78% and 70% on my local laptop.  
### However, when just looking at the raw seconds taken to train the models,
### moving to supercomputing doesn't seem to make a huge difference compared to my 
### local laptop, especially when parallelization is not enabled. 


## 2. What is the relationship between time and the number of cores used?
### By comparing tables 2 and 4, I observed that when parallel processing is used, 
### increasing the number of cores (from 7 to 14) generally reduced the number 
### of seconds needed to train all four models. 


## 3. If your supervisor asked you to pick a model for use in a production model, would you recommend using the supercomputer and why? Consider all four tables when providing an answer.
### Similar to my previous response, I will again choose XGBoost because of its
### greatest R-squared values from 10-fold cross-validation and holdout test validation.
### Taking all tables into consideration, I will likely NOT recommend using 
### the supercomputer not because of performance but because of its minimal 
### improvement in raw model training time. Based on tables 1 & 3, it is observed
### that model performance from my local desktop and MSI did not vary by much.
### Both tables 1 & 3 agree with each other than XGBoost should be chosen because 
### it has greatest R-Squared values from both cross-validation and holdout test validation. 
### Based on tables 2 & 4, although there is overall a greater % reduction in 
### training time for parallelization vs. no parallelization when supercomputer
### is used as opposed to a local desktop like mine, the raw training times (in s)
### needed did not show great improvement. Take the fastest and slowest training 
### models as an example , without parallelization, it took the supercomputer 
### close to 9s to train OLS and 220s to train XGBoost, but it only took my local 
### computer about 3s to train OLS and 128s to train XGBoost.
### Indeed, the raw training time in seconds improved slightly for random forest
### and XGBoost such that training time for random forest decreased from 35s to 33s
### and that for XGBoost decreased from 28s to 22s. However, this decrease, in my opinion, was quite minimal. 
### Taken together, for this given problem, supercomputer doesn't seem to yield much
### better performance (there isn't a reason to believe it would anyway!), 
### its raw training time is not significantly faster, and the additional steps 
### needed to work with shell and allocate the right amount of resources, I 
### would not recommend using a supercomputer for this specific problem! But 
### I think it is important to benchmark training times on my local computer 
### against other local laptops as there may by idiosyncracies across laptops
### (because they may have different computer powers). 




results = function(train_mod){
  algo = train_mod$method
  cv_rsq = str_remove(format(round(max(train_mod$results$Rsquared), 2), nsmall = 2), "^\\d")
  preds = predict(train_mod, gss_tbl_test, na.action = na.pass)
  ho_rsq = str_remove(format(round(cor(preds, gss_tbl_test$workhours)^2, 2), nsmall = 2), "^\\d")
  return(c("algo" = algo, "cv_rsq" = cv_rsq, "ho_sq" = ho_rsq))
}

table1_tbl = as_tibble(t(sapply(mod_ls_original, results))) 

table2_tbl = tibble(algorithm = mod_vec,
                    supercomputer = original_time,
                    "supercomputer-14" = parallel_time)

## Saving tables as Tables 3 and 4 
write_csv(table1_tbl, "../out/table3.csv")
write_csv(table2_tbl, "../out/table4.csv")

