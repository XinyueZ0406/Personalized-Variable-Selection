library(dplyr)
library(MASS)
library(Matrix)
library(randomForest)
library(randomForestSRC)
library(reshape2)
library(caret)
library(mclust)

calibrate_isotonic <- function(pred_valid, y_valid,pred_test,groups) {
  # Check if labels are factors, if not convert them
  if (!is.factor(y_valid)) {
    y_valid <- as.factor(y_valid)
  }
  calibrated_probs_test <- matrix(0, nrow = nrow(pred_test), ncol = groups)
  for (class in 1:groups) {
    # Extract predicted probabilities for the current class from validation data
    probs_class_valid <- pred_valid[, class]
    y_binary_valid <- as.numeric(y_valid == class)
    # Fit isotonic regression on the validation data for the current class
    iso_fit <- isoreg(probs_class_valid, y_binary_valid)
    step_fun <- as.stepfun(iso_fit)
    probs_class_test <- pred_test[, class]
    min_x <- min(iso_fit$x)
    max_x <- max(iso_fit$x)
    probs_class_test_clamped <- pmax(pmin(probs_class_test, max_x), min_x)
    calibrated_probs_test[, class] <- step_fun(probs_class_test_clamped)
  }
  zero_sum_rows <- which(rowSums(calibrated_probs_test) == 0)
  if (length(zero_sum_rows) > 0) {
    calibrated_probs_test[zero_sum_rows, ] <- 1 / groups  # Uniform probabilities
  }
  calibrated_probs_test[calibrated_probs_test == 0] <- 1e-10
  calibrated_probs_test <- sweep(calibrated_probs_test, 1, rowSums(calibrated_probs_test), "/")
  return(calibrated_probs_test)
}

calculate_entropy <- function(prob_matrix) {
  #prob_matrix <- prob_matrix[prob_matrix > 0]  # Avoid log(0)
  return(-sum(prob_matrix * log(prob_matrix)))
}

compute_ELIE_ave_cls <- function(total_trees,mrf_model,test_df,rf_model,groups,y_valid,pred_valid,block_1_names,imp_names) {
  predicted_y_test_true_cal_imp <- array(NA, dim = c(total_trees, nrow(test_df), groups))
  for (imputation_idx in 1:total_trees) {
    pred_mrf <- predict.rfsrc(mrf_model, get.tree = imputation_idx, test_df[,block_1_names])
    pred_list <- lapply(imp_names, function(feature) {
      pred_mrf$regrOutput[[feature]]$predicted})
    pp <- do.call(cbind, pred_list)
    new_test_df=cbind(test_df[,block_1_names],pp)
    colnames(new_test_df)[(length(block_1_names)+1):(length(block_1_names)+length(imp_names))]=imp_names
    predicted_test_array=predict(rf_model, new_test_df,type = "prob")
    predicted_y_test_true_cal_imp[imputation_idx,,]=calibrate_isotonic(pred_valid,y_valid,
                                                                       pred_test=predicted_test_array,
                                                                       groups) 
  }
  moe <- numeric(nrow(test_df))
  for (i in 1:nrow(test_df)) {
    prob_matrix <- predicted_y_test_true_cal_imp[, i, ]
    entropy_for_i <- apply(prob_matrix, 1, calculate_entropy)
    moe[i] <- mean(entropy_for_i)
  }
  
  mm=apply(predicted_y_test_true_cal_imp,c(2,3),mean)
  eom <- apply(mm, 1, calculate_entropy)
  ELIE=eom-moe
  return(ELIE)
}

compute_ELIE_nns_cls <- function(total_trees,mrf_model,test_df,train_df,rf_model,groups,y_valid,pred_valid,block_1_names,imp_names) {
  predicted_y_test_true_cal_imp <- array(NA, dim = c(total_trees, nrow(test_df), groups))
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
    predicted_test_array=predict(rf_model, new_test_df,type = "prob")
    predicted_y_test_true_cal_imp[j,,]=calibrate_isotonic(pred_valid,y_valid,
                                                          pred_test=predicted_test_array,
                                                          groups=groups)
  }
  moe <- numeric(nrow(test_df))
  for (i in 1:nrow(test_df)) {
    prob_matrix <- predicted_y_test_true_cal_imp[, i, ]
    entropy_for_i <- apply(prob_matrix, 1, calculate_entropy)
    moe[i] <- mean(entropy_for_i)
  }
  mm=apply(predicted_y_test_true_cal_imp,c(2,3),mean)
  eom <- apply(mm, 1, calculate_entropy)
  ELIE=eom-moe
  return(ELIE)
}


MRF_imputation_step_cls <- function(y_name,block_1_names,imp_names,train_df,test_df,valid_df,groups,
                                    full_tree_size=10,red_tree_size=30,total_trees=1000){
  # train full model with X and Z
  rf_model <- randomForest(train_df[,y_name] ~ ., data=train_df[, names(train_df) != y_name], ntree=full_tree_size)
  predicted_y_test_true=predict(rf_model, test_df,type = "prob")
  predicted_y_valid_true=predict(rf_model, valid_df,type = "prob")
  predicted_y_test_true_cal=calibrate_isotonic(pred_valid=predicted_y_valid_true, 
                                               y_valid=valid_df[,y_name],
                                               pred_test=predicted_y_test_true,
                                               groups=groups) 
  log_probs <- mapply(function(prob_row, true_class) {
    log(prob_row[true_class])
  }, split(predicted_y_test_true_cal, seq(nrow(predicted_y_test_true_cal))), test_df[,y_name])
  log_probs_full <- as.numeric(log_probs)
  #fit reduced model only with X
  rf_model_5 <- randomForest(train_df[,y_name] ~ ., data=train_df[,block_1_names], ntree=red_tree_size)
  predicted_y_test_true_5=predict(rf_model_5, test_df,type = "prob")
  predicted_y_valid_true_5=predict(rf_model_5, valid_df,type = "prob")
  predicted_y_test_true_cal_5 = calibrate_isotonic(pred_valid=predicted_y_valid_true_5, 
                                                   y_valid=valid_df[,y_name],
                                                   pred_test=predicted_y_test_true_5,
                                                   groups=groups) 
  log_probs_5 <- mapply(function(prob_row, true_class) {
    log(prob_row[true_class])
  }, split(predicted_y_test_true_cal_5, seq(nrow(predicted_y_test_true_cal_5))), test_df[,y_name])
  log_probs_5=as.numeric(log_probs_5)
  #mrf
  lhs <- paste(imp_names, sep = "", collapse = ", ")
  rhs <- paste(block_1_names, sep = "", collapse = " + ")
  formula_str <- paste("Multivar(", lhs, ") ~ ", rhs)
  formula_obj <- as.formula(formula_str)
  mrf_model=rfsrc(formula_obj,  ntree = total_trees,membership = TRUE,forest=TRUE,data =train_df)
  ELIE_ave=compute_ELIE_ave_cls(total_trees,mrf_model,test_df,rf_model,groups,y_valid=valid_df[,y_name],
                                pred_valid=predicted_y_valid_true,block_1_names,imp_names)
  ELIE_nns=compute_ELIE_nns_cls(total_trees,mrf_model,test_df,rf_model,groups,y_valid=valid_df[,y_name],
                                pred_valid=predicted_y_valid_true,block_1_names,imp_names)
  
  return(list("ELIE_ave"=ELIE_ave,
              "ELIE_nns"=ELIE_nns,
              "log_probs_full"=log_probs_full,
              "log_probs_5"=log_probs_5))
  
}


