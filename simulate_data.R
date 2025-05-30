library(dplyr)
library(MASS)
library(Matrix)
library(randomForest)
library(randomForestSRC)
library(reshape2)
library(caret)
library(mclust)

add_sparse_noise <- function(mean_vector, perturb_dims, sd_perturb = 3) {
  perturbation <- rep(0, length(mean_vector))
  perturbation[perturb_dims] <- rnorm(length(perturb_dims), mean = 0, sd = sd_perturb)
  return(mean_vector + perturbation)
}

simulate_data_cls <- function(n, train_n, test_n,p=100,groups=6){
  # n is total number of observations 
  #p is total number of predictors
  # groups is the number of classes
  overall_mean <- runif(p, -1, 1) 
  group_means <- vector("list", groups)
  #tree structure to generate group mean
  
  dim_list=sample(6:100,25)
  l1_mean <- add_sparse_noise(overall_mean, perturb_dims = 1:5)
  l2_mean <- add_sparse_noise(overall_mean, perturb_dims = dim_list[1:5])
  l3_mean <- add_sparse_noise(overall_mean, perturb_dims = dim_list[6:10])
  
  group_means[[1]] <- add_sparse_noise(l1_mean, perturb_dims = 1:5)  # Noise in the first 5 dimensions
  group_means[[2]] <- add_sparse_noise(l1_mean, perturb_dims = dim_list[11:15])
  
  group_means[[3]] <- add_sparse_noise(l2_mean, perturb_dims = 1:5)
  group_means[[4]] <- add_sparse_noise(l2_mean, perturb_dims = dim_list[16:20])
  
  group_means[[5]] <- add_sparse_noise(l3_mean, perturb_dims = 1:5)
  group_means[[6]] <- add_sparse_noise(l3_mean, perturb_dims = dim_list[21:25])
  
  sigma <- diag(runif(p, min = 1, max = 1))
  corr_mat <- diag(1, p)
  corr_mat[upper.tri(corr_mat)] <- runif(p * (p - 1) / 2, min = -0.5, max = 0.5)
  corr_mat <- corr_mat + t(corr_mat) - diag(diag(corr_mat))
  corr_mat <- nearPD(corr_mat, corr = TRUE)$mat
  cov_mat <- sigma %*% corr_mat %*% sigma
  
  
  simulated_data <- do.call(rbind, lapply(1:groups, function(g) {
    # Generate data for each group
    matrix( mvrnorm(n = n / groups, mu = group_means[[g]], Sigma = cov_mat), 
            nrow = n / groups, ncol = p)
  }))
  
  group_labels <- rep(1:groups, each = n / groups)
  simulated_data <- data.frame(simulated_data)
  names(simulated_data) = paste0("v", 1:p)
  simulated_data$label <- factor(group_labels)
  
  train_index = sample(length(group_labels), train_n)
  train_df = simulated_data[train_index, ]
  test_valid_df = simulated_data[-train_index, ]
  test_index  = sample(nrow(test_valid_df ), test_n)
  test_df = test_valid_df[test_index,]
  valid_df = test_valid_df[-test_index,]
  return(list("train_df"=train_df,
              "test_df"=test_df,
              "valid_df"=valid_df))
  
}


simulate_data_reg <- function(n, train_n, test_n,p=100,groups=6){
  overall_mean <- runif(p, -1, 1) 
  group_means <- vector("list", groups)
  dim_list=sample(6:100,25)
  l1_mean <- add_sparse_noise(overall_mean, perturb_dims = 1:5)
  l2_mean <- add_sparse_noise(overall_mean, perturb_dims = dim_list[1:5])
  l3_mean <- add_sparse_noise(overall_mean, perturb_dims = dim_list[6:10])
  
  group_means[[1]] <- add_sparse_noise(l1_mean, perturb_dims = 1:5)  # Noise in the first 5 dimensions
  group_means[[2]] <- add_sparse_noise(l1_mean, perturb_dims = dim_list[11:15])
  
  group_means[[3]] <- add_sparse_noise(l2_mean, perturb_dims = 1:5)
  group_means[[4]] <- add_sparse_noise(l2_mean, perturb_dims = dim_list[16:20])
  
  group_means[[5]] <- add_sparse_noise(l3_mean, perturb_dims = 1:5)
  group_means[[6]] <- add_sparse_noise(l3_mean, perturb_dims = dim_list[21:25])
  
  sigma <- diag(runif(p, min = 1, max = 1))
  corr_mat <- diag(1, p)
  corr_mat[upper.tri(corr_mat)] <- runif(p * (p - 1) / 2, min = -0.5, max = 0.5)
  corr_mat <- corr_mat + t(corr_mat) - diag(diag(corr_mat))
  corr_mat <- nearPD(corr_mat, corr = TRUE)$mat
  cov_mat <- sigma %*% corr_mat %*% sigma
  
  coefficients_vecs=matrix(sample(c(0.5, -0.5), groups * p, replace = TRUE), 
                           nrow = groups, ncol = p)
  
  
  X_list <- list()
  coefficients_matrix=c()
  resY = c()
  for (g in 1:groups){
    X=mvrnorm(n = n / groups, mu = group_means[[g]], Sigma = cov_mat)
    zero_coeffs_95=which(group_means[[g]]!=overall_mean)
    coefficients <- coefficients_vecs[g,]
    zero_coeffs<- c(sample(1:5, 6-g),sample(setdiff(zero_coeffs_95,1:5), g-1))
    print(zero_coeffs)
    coefficients[-zero_coeffs] <- 0
    coefficients_matrix= cbind(coefficients_matrix,coefficients)
    y <- X %*% coefficients + rnorm(1, mean = 0, sd = 1)
    resY = c(resY, y)
    X_list[[g]] <- X
  }
  
  X = do.call(rbind, X_list)
  combined_df = as.data.frame(X)
  names(combined_df) = paste0("v", 1:p)
  combined_df$Y = resY+rnorm(n, mean = 0, sd = 1)
  group_labels <- rep(1:groups, each = n / groups)
  
  train_index = sample(length(resY), train_n)
  train_df = combined_df[train_index, ]
  test_df = combined_df[-train_index, ]
  return(list("train_df"=train_df,
              "test_df"=test_df))
  
}




