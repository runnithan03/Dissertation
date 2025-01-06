install.packages("glmnet")
install.packages("caret")
install.packages("ggplot2")
install.packages("grpreg") 

library(glmnet)
library(caret)
library(ggplot2)
library(grpreg)

data <- read.csv("C:\\Users\\rajiu\\OneDrive\\Documents\\4th Year\\Dissertation\\Code\\Datasets\\final.csv")
str(data)

# Frequency encode the categorical variables
rating_counts <- table(data$rating)
data$rating_encoded <- as.numeric(rating_counts[data$rating])

risk_rating_counts <- table(data$risk_rating)
risk_rating_counts
data$risk_rating_encoded <- as.numeric(risk_rating_counts[data$risk_rating])

category_counts <- table(data$category)
category_counts
data$category_encoded <- as.numeric(category_counts[data$category])

data$holdings_n_stock <- as.numeric(data$holdings_n_stock)

# Drop original categorical variables
data <- data[, setdiff(names(data), c("rating", "risk_rating", "category"))]
str(data)

response_vars <- c("roe", "sustainability_score")  
predictor_vars <- setdiff(names(data), response_vars)  

# Split data into training and test sets
set.seed(40)
train_index <- sample(seq_len(nrow(data)), size = 0.8 * nrow(data))
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Standardise predictors and responses
X_train <- scale(as.matrix(train_data[, predictor_vars]))
Y_train <- as.matrix(train_data[, response_vars])
X_test <- scale(as.matrix(test_data[, predictor_vars]), center = attr(X_train, "scaled:center"), scale = attr(X_train, "scaled:scale"))
Y_test <- as.matrix(test_data[, response_vars])  

# Function to calculate NRMSE for multivariate responses
calculate_nrmse <- function(predictions, actual_data) {
  residuals <- actual_data - predictions
  rmse_per_response <- apply(residuals, 2, function(residuals_col) {
    sqrt(mean(residuals_col^2))
  })
  std_devs <- apply(actual_data, 2, sd, na.rm = TRUE)
  nrmse <- mean(rmse_per_response / std_devs)  # Average NRMSE across all responses
  return(nrmse)
}

# Function to fit models and calculate NRMSE for multivariate regression
fit_and_evaluate <- function(alpha_value, x, y) {
  cv_model <- cv.glmnet(x = x, y = y, family = "mgaussian", alpha = alpha_value)
  lambda_min <- cv_model$lambda.min
  predictions <- predict(cv_model, s = lambda_min, newx = x)[,, 1]
  nrmse <- calculate_nrmse(predictions, y)
  return(list(lambda = lambda_min, nrmse = nrmse, model = cv_model))
}

# Lasso Regression (alpha = 1)
cat("Running Lasso Regression...\n")
lasso_results <- fit_and_evaluate(alpha_value = 1, x = X_train, y = Y_train)
cat("Lasso NRMSE (Train):", lasso_results$nrmse, "\n")
lasso_test_predictions <- predict(lasso_results$model, s = lasso_results$lambda, newx = X_test)[,, 1]
cat("Lasso NRMSE (Test):", calculate_nrmse(lasso_test_predictions, Y_test), "\n")

# Ridge Regression (alpha = 0)
cat("Running Ridge Regression...\n")
ridge_results <- fit_and_evaluate(alpha_value = 0, x = X_train, y = Y_train)
cat("Ridge NRMSE (Train):", ridge_results$nrmse, "\n")
ridge_test_predictions <- predict(ridge_results$model, s = ridge_results$lambda, newx = X_test)[,, 1]
cat("Ridge NRMSE (Test):", calculate_nrmse(ridge_test_predictions, Y_test), "\n")

# Elastic Net Regression (alpha tuning)
cat("Tuning Elastic Net...\n")
alpha_values <- seq(0.1, 0.9, by = 0.1)  # More granular alpha values
elastic_results <- data.frame(alpha = numeric(), lambda = numeric(), train_nrmse = numeric(), test_nrmse = numeric())

for (alpha in alpha_values) {
  result <- fit_and_evaluate(alpha_value = alpha, x = X_train, y = Y_train)
  elastic_test_predictions <- predict(result$model, s = result$lambda, newx = X_test)[,, 1]
  elastic_results <- rbind(elastic_results, data.frame(
    alpha = alpha,
    lambda = result$lambda,
    train_nrmse = result$nrmse,
    test_nrmse = calculate_nrmse(elastic_test_predictions, Y_test)
  ))
}

# Find the best alpha
best_elastic <- elastic_results[which.min(elastic_results$test_nrmse), ]
cat("Best Elastic Net Alpha:", best_elastic$alpha, "\n")
cat("Best Elastic Net Lambda:", best_elastic$lambda, "\n")
cat("Best Elastic Net NRMSE (Train):", best_elastic$train_nrmse, "\n")
cat("Best Elastic Net NRMSE (Test):", best_elastic$test_nrmse, "\n")

# Plot Elastic Net Results
ggplot(elastic_results, aes(x = alpha, y = test_nrmse)) +
  geom_line() +
  geom_point() +
  labs(title = "Elastic Net Test NRMSE vs Alpha", x = "Alpha", y = "Test NRMSE") +
  theme_minimal()

# Group Lasso
cat("Running Group Lasso...\n")

# Define groups (e.g., group predictors by domain knowledge or categories)
groups <- rep(1:length(predictor_vars), each = 1)  # Example: Each predictor is its own group
group_lasso_model <- cv.grpreg(X_train, Y_train, group = groups, penalty = "grLasso")

# Best lambda
group_lambda_min <- group_lasso_model$lambda.min

# Predict and evaluate
group_lasso_predictions <- predict(group_lasso_model, X_test, lambda = group_lambda_min)
group_lasso_nrmse <- calculate_nrmse(group_lasso_predictions, Y_test)
cat("Group Lasso NRMSE (Test):", group_lasso_nrmse, "\n")

# Model Comparison
cat("Model Comparison:\n")
cat("Lasso NRMSE (Test):", calculate_nrmse(lasso_test_predictions, Y_test), "\n")
cat("Ridge NRMSE (Test):", calculate_nrmse(ridge_test_predictions, Y_test), "\n")
cat("Elastic Net NRMSE (Test):", best_elastic$test_nrmse, "\n")
cat("Group Lasso NRMSE (Test):", group_lasso_nrmse, "\n")
