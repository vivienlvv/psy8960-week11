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
## has a maximum of 1 node. I will double the # of cores used in MSI to avoid
## taking up too much resources and long
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

