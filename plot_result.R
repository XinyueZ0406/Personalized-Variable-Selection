library(dplyr)
library(MASS)
library(Matrix)
library(randomForest)
library(randomForestSRC)
library(reshape2)
library(caret)
library(mclust)

compute_curve <- function(df, thresholds = seq(0, 100, by = 1)) {
  # Rank by decreasing ELLIE
  df <- df[order(df$ELIE,decreasing = TRUE), ]
  total_points <- nrow(df)
  results1 <- data.frame(
    cumsum_ELIE = numeric(length(thresholds)),
    total_loglik = numeric(length(thresholds))
  )
  for (i in seq_along(thresholds)) {
    perc <- thresholds[i]
    n_top <- ceiling(total_points * perc / 100)
    
    # Top group uses full model, rest uses reduced model 
    hybrid_loss <- c(df$loss_full[1:n_top], df$loss_5[(n_top+1):total_points])
    
    # Cumulative sum of ELLIE for the top group = x-axis value
    x_cumsum <- sum(df$ELIE[1:n_top], na.rm = TRUE)

    total_loss <- sum(hybrid_loss, na.rm = TRUE)
    
    results1$cumsum_ELIE[i] <- x_cumsum
    results1$total_loss[i] <- total_loss
  }
  return(results1)
}


data=simulate_data_cls(n=18000, train_n=8000, test_n=9000,p=100,groups=6)

ELIE_result_cls=MRF_imputation_step_cls(y_name='label',block_1_names=colnames(data$train_df)[1:5],
                                        imp_names=colnames(data$train_df)[6:100],train_df=data$train_df,
                                        test_df=data$test_df,valid_df=data$valid_df,groups=6,
                                        full_tree_size=10,red_tree_size=30,total_trees=1000)

data=simulate_data_reg(n=18000, train_n=9000, test_n=9000,p=100,groups=6)
ELIE_result_reg=MRF_imputation_step_reg(y_name='Y',block_1_names=colnames(data$train_df)[1:5],
                                        imp_names=colnames(data$train_df)[6:100],train_df=data$train_df,test_df=data$test_df,
                                        full_tree_size=10,red_tree_size=30,total_trees=1000)


# use regression example to show the results in the paper
#Histogram of ELIE
ELIE_result_reg=data.frame(ELIE_result_reg)
ggplot(data=ELIE_result_reg,aes(x=ELIE_ave))+
  geom_histogram(fill = "blue", color = "black", alpha = 0.7)+
  labs(x = "ELIE_avs")

ggplot(data=ELIE_result_reg,aes(x=ELIE_nns))+
  geom_histogram(fill = "blue", color = "black", alpha = 0.7)+
  labs(x = "ELIE_nns")

#also can do plots for ELIE_nns
df=data.frame(ELIE=ELIE_result_reg$ELIE_ave,loss_full=ELIE_result_reg$log_probs_full,loss_5=ELIE_result_reg$log_probs_5)
plot_result_ave=compute_curve(df)
plot_result_ave$percent=seq(0,100,1)
df=data.frame(ELIE=ELIE_result_reg$ELIE_nns,loss_full=ELIE_result_reg$log_probs_full,loss_5=ELIE_result_reg$log_probs_5)
plot_result_nns=compute_curve(df)
plot_result_nns$percent=seq(0,100,1)
plot_result_ave$Method <- "average"
plot_result_nns$Method <- "sample"   
plot_combined <- bind_rows(plot_result_ave,plot_result_nns)

# SSE vs cumulative ELIE for observations with Z measured
ggplot(plot_combined , aes(x =cumsum_ELIE , y = total_loss,color = Method)) +
  geom_line(size = 1.2)+
  labs(x = "Cumulative Sum of ELIE_ave", y = "Sum of Squared Error")+
  geom_abline(slope = -1, intercept =sum(ELIE_result_reg$squared_error_5), color = "black", linetype = "dashed")

# MSE vs percentage of observation with Z measured
ggplot(plot_combined , aes(x =percent , y = total_loss/nrow(data$test_df),color = Method)) +
  geom_line(size = 1.2)+
  labs(x = "Percentage of observations", y = "Mean Squared Error")
