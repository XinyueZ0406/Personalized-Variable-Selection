library(dplyr)
library(MASS)
library(Matrix)
library(randomForest)
library(randomForestSRC)
library(reshape2)
library(caret)
library(mclust)

compute_ELIE_ave_reg <- function(total_trees,mrf_model,test_df,rf_model,block_1_names,imp_names){
  pred_test_list = list()
  total_trees <- mrf_model$ntree
  for (imputation_idx in 1:total_trees) {
    pred_mrf <- predict.rfsrc(mrf_model, get.tree = imputation_idx, test_df[,block_1_names])
    pred_list <- lapply(imp_names, function(feature) {
      pred_mrf$regrOutput[[feature]]$predicted})
    pp <- do.call(cbind, pred_list)
    new_test_df=cbind(test_df[,block_1_names],pp)
    colnames(new_test_df)[(length(block_1_names)+1):(length(block_1_names)+length(imp_names))]=imp_names
    pred_test_list[[imputation_idx]] = predict(rf_model, new_test_df)
  }
  y_mrf_matrix = do.call(cbind, pred_test_list)
  ELIE = apply(y_mrf_matrix, 1, var)
  
  return(ELIE)
}

compute_ELIE_nns_reg <- function(total_trees,mrf_model,test_df,train_df,rf_model,block_1_names,imp_names){
  pred_test_list = list()
  total_trees <- mrf_model$ntree
  train_nodes <- mrf_model$membership
  test_nodes <- predict(mrf_model, newdata = test_df[,block_1_names], membership = TRUE)$membership
  for (j in 1:total_trees) {
    new_test_x2<- list()
    for (i in 1:nrow(test_df)) {
      test_node <- test_nodes[i, j]
      train_inds <- which(train_nodes[, j] == test_node)
      if (length(train_inds) > 0) {
        sampled_k <- sample(train_inds, 1)
        new_x2 <- train_df[sampled_k, imp_names]
        new_test_x2[[length(new_test_x2) + 1]] <- new_x2 
      }
    }
    new_test_x2_df <- do.call(rbind, new_test_x2)
    new_test_df=cbind(test_df[,block_1_names],new_test_x2_df)
    pred_test_list[[j]] = predict(rf_model, new_test_df)
  }
  y_mrf_matrix = do.call(cbind, pred_test_list)
  ELIE = apply(y_mrf_matrix, 1, var)
  
  return(ELIE)
}

MRF_imputation_step_reg <- function(y_name,block_1_names,imp_names,train_df,test_df,
                                    full_tree_size=10,red_tree_size=30,total_trees=1000){
  # train full model with X and Z
  rf_model <- randomForest(train_df[,y_name] ~ ., data=train_df[, names(train_df) != y_name], ntree=full_tree_size)
  predicted_y_test=predict(rf_model,test_df)
  squared_errors_full <- (test_df[,y_name] - predicted_y_test)^2
  
  #fit reduced model only with X
  rf_model_5 <- randomForest(train_df[,y_name] ~ ., data=train_df[,block_1_names], ntree=red_tree_size)
  predicted_y_test_5=predict(rf_model_5,test_df[,block_1_names])
  squared_errors_5 <- (test_df[,y_name] - predicted_y_test_5)^2
  
  #mrf
  lhs <- paste(imp_names, sep = "", collapse = ", ")
  rhs <- paste(block_1_names, sep = "", collapse = " + ")
  formula_str <- paste("Multivar(", lhs, ") ~ ", rhs)
  formula_obj <- as.formula(formula_str)
  mrf_model=rfsrc(formula_obj,  ntree = total_trees,membership = TRUE,forest=TRUE,data =train_df)
  ELIE_ave=compute_ELIE_ave_reg(total_trees=total_trees,mrf_model=mrf_model,
                                test_df=test_df,rf_model=rf_model,
                                block_1_names=block_1_names,imp_names=imp_names)
  ELIE_nns=compute_ELIE_nns_reg(total_trees=total_trees,mrf_model=mrf_model,
                                test_df=test_df,train_df=train_df,rf_model=rf_model,
                                block_1_names=block_1_names,imp_names=imp_names)
  
  return(list("ELIE_ave"=ELIE_ave,
              "ELIE_nns"=ELIE_nns,
              "squared_errors_full"=squared_errors_full,
              "squared_errors_5"=squared_errors_5))
  
}
