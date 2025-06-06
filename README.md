# Personalized-Variable-Selection
## Introduction
This is an implementation of the paper "Personalized Variable Selection". This method proposed a personalized variable selection method designed to decide whether additional variables should be measured for each data point based on the variables already collected. We propose a novel metric called Expected Loss Improvement Estimate (ELIE), which quantifies the expected improvement in the sum of squared error (SSE) for regression or negative total log-likelihood for classification by collecting additional variables. Our method uses the multivariate random forest to generate multiple imputed values for uncollected variables using available features. These imputed datasets are then used to produce multiple predictions through a pre-trained model. The core idea of our method is to measure the variability of these predictions as ELIE. Large ELLIE values indicate the variability in predictions across imputed datasets, suggesting that collecting the true values of uncollected variables can significantly improve model accuracy. Conversely, small ELLIE values suggest that imputed values produce stable predictions, thus additional data collection is unnecessary. 
## Usage
Here, I gave an example of simulating data in the paper.
For regression,
```
run data=simulate_data_reg(n=18000, train_n=9000, test_n=9000,p=100,groups=6)
MRF_imputation_step_reg(y_name='Y',block_1_names=colnames(data$train_df)[1:5],
                                        imp_names=colnames(data$train_df)[6:100],train_df=data$train_df,test_df=data$test_df,
                                        full_tree_size=10,red_tree_size=30,total_trees=1000)

```
For classification,
```
run data=simulate_data_cls(n=18000, train_n=8000, test_n=9000,p=100,groups=6)
MRF_imputation_step_cls(y_name='label',block_1_names=colnames(data$train_df)[1:5],
                                        imp_names=colnames(data$train_df)[6:100],train_df=data$train_df,
                                        test_df=data$test_df,valid_df=data$valid_df,groups=6,
                                        full_tree_size=10,red_tree_size=30,total_trees=1000)

```
Note: 
1. y_name is the name of response variable
2. block_1_names is the name list of all observed variables
3. imp_names is the name list of all missing or uncollected variables
4. full_tree_size is the the number of trees for random forest using all variables including observed variables and uncollected variables
5. red_tree_size is the the number of trees for random forest only using all observed variables
6. total_trees is the number of trees for multivariate random forest, which is the number of imputations
7. groups is the number of classes for classification problem.
   
