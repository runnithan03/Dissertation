install.packages("MRCE", dependencies = TRUE)
install.packages("ggplot2")
install.packages("caret")
install.packages("Matrix")  
install.packages("rrpack")  

library(MRCE)
library(ggplot2)
library(caret)
library(Matrix)
library(rrpack)

data <- read.csv("C:\\Users\\raulu\\OneDrive\\Documents\\4th Year\\Dissertation\\Code\\clean.csv")
dim(data)
cor(data$rating, data$risk_rating, method = "spearman")

# Assuming your data is stored in a data frame called 'data' with columns 'roe' and 'sustainability'

# Create the scatter plot with semi-transparent black points
plot(data$roe, data$sustainability,
     main = "ROE vs Sustainability Score",
     xlab = "Return on Equity (ROE)",
     ylab = "Sustainability Score",
     pch = 1,
     col = "black")  # Black with 50% opacity

# Fit a linear model
model <- lm(sustainability_score ~ roe, data = data)

# Add the regression line to the plot in purple
abline(model, col = "red", lwd = 2)

# Compute the correlation matrix for the two responses
response_cor <- cor(data[c("roe", "sustainability_score")])
print(response_cor)

# Compute the covariance matrix for the two responses
response_cov <- cov(data[c("roe", "sustainability_score")])
print(response_cov)

dim(data)
print(length(unique(data$category)))

## Data Pre-Preparation
# Encode categorical variables
encode_categorical <- function(df, col_name) {
  counts <- table(df[[col_name]])
  df[[paste0(col_name, "")]] <- as.numeric(counts[df[[col_name]]])
  return(df)
}

categorical_vars <- c("rating", "risk_rating", "category")
for (var in categorical_vars) {
  data <- encode_categorical(data, var)
}

# Convert numeric variables & drop original categorical columns
data$holdings_n_stock <- as.numeric(data$holdings_n_stock)
data <- data[, setdiff(names(data), categorical_vars)]

# Define predictors and responses
response_vars <- c("roe", "sustainability_score")  
predictor_vars <- setdiff(names(data), response_vars)  

## Separate training and test datasets
set.seed(1)
train_index <- sample(seq_len(nrow(data)), size = 0.8 * nrow(data))
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

X_train <- scale(as.matrix(train_data[, predictor_vars]))
Y_train <- scale(as.matrix(train_data[, response_vars]))
X_test <- scale(as.matrix(test_data[, predictor_vars]), center = attr(X_train, "scaled:center"), scale = attr(X_train, "scaled:scale"))
Y_test <- scale(as.matrix(test_data[, response_vars]), center = attr(Y_train, "scaled:center"), scale = attr(Y_train, "scaled:scale"))


## Key Functions for this dataset
calculate_nrmse <- function(predictions, actual) {
  residuals <- actual - predictions
  rmse <- apply(residuals, 2, function(col) sqrt(mean(col^2)))
  nrmse <- mean(rmse / apply(actual, 2, sd, na.rm = TRUE))
  return(nrmse)
}
?mrce
# MRCE with cross-validation
fit_mrce <- function(X_train, Y_train, X_test, Y_test, method = "single") {
  k_folds <- 5
  folds <- sample(1:k_folds, nrow(X_train), replace = TRUE)
  
  best_nrmse <- Inf
  best_mrce_model <- NULL
  lambda_grid <- seq(0.01, 0.2, length.out = 5)
  
  for (lam1 in lambda_grid) {
    for (lam2 in lambda_grid) {
      cv_nrmse <- c()
      
      for (k in 1:k_folds) {
        train_idx <- folds != k
        val_idx <- folds == k
        
        model <- mrce(X_train[train_idx, ], Y_train[train_idx, ], lam1 = as.numeric(lam1), lam2 = as.numeric(lam2), method = method)
        predictions <- X_train[val_idx, ] %*% model$B
        cv_nrmse <- c(cv_nrmse, calculate_nrmse(predictions, Y_train[val_idx, ]))
      }
      
      avg_nrmse <- mean(cv_nrmse)
      if (avg_nrmse < best_nrmse) {
        best_nrmse <- avg_nrmse
        best_mrce_model <- model
      }
    }
  }
  
  test_predictions <- X_test %*% best_mrce_model$B
  return(calculate_nrmse(test_predictions, Y_test))
}

?rrr.fit
# Reduced-rank ridge regression
fit_rrridge <- function(X_train, Y_train, X_test, Y_test, rank = 2, lambda = 1) {
  X_train <- as.matrix(X_train)
  Y_train <- as.matrix(Y_train)
  X_test <- as.matrix(X_test)
  Y_test <- as.matrix(Y_test)
  
  # Compute Ridge-penalised estimate
  ridge_coef <- solve(t(X_train) %*% X_train + lambda * diag(ncol(X_train))) %*% t(X_train) %*% Y_train
  
  # Compute SVD of ridge solution
  svd_res <- svd(ridge_coef)
  
  # Keep only the top 'rank' singular values
  B_rrridge <- svd_res$u[, 1:rank] %*% diag(svd_res$d[1:rank]) %*% t(svd_res$v[, 1:rank])
  
  # Predictions
  predictions <- X_test %*% B_rrridge
  
  return(calculate_nrmse(predictions, Y_test))
}



mrce_nrmse <- fit_mrce(X_train, Y_train, X_test, Y_test, method = "single")
rrridge_nrmse <- fit_rrridge(X_train, Y_train, X_test, Y_test, rank = 2)

# Include Polynomial Features
X_train_poly <- poly(X_train, degree = 2, raw = TRUE)
X_test_poly <- poly(X_test, degree = 2, raw = TRUE)

mrce_poly_nrmse <- fit_mrce(X_train_poly, Y_train, X_test_poly, Y_test, method = "single")
rrridge_poly_nrmse <- fit_rrridge(X_train_poly, Y_train, X_test_poly, Y_test, rank = 2)

# Include Interaction Features
X_train_interact <- model.matrix(~ .^2, data = as.data.frame(X_train))[, -1]
X_test_interact <- model.matrix(~ .^2, data = as.data.frame(X_test))[, -1]

mrce_interact_nrmse <- fit_mrce(X_train_interact, Y_train, X_test_interact, Y_test, method = "single")
rrridge_interact_nrmse <- fit_rrridge(X_train_interact, Y_train, X_test_interact, Y_test, rank = 2)

results <- data.frame(
  Model = c("MRCE", "RRRidge",
            "MRCE (Poly)", "RRRidge (Poly)",
            "MRCE (Interact)", "RRRidge (Interact)"),
  NRMSE = c(mrce_nrmse, rrridge_nrmse,
            mrce_poly_nrmse, rrridge_poly_nrmse,
            mrce_interact_nrmse, rrridge_interact_nrmse)
)

print(results)

ggplot(results, aes(x = Model, y = NRMSE, fill = Model)) +
  geom_bar(stat = "identity") +
  labs(title = "NRMSE Comparison (MRCE and RRRidge)", y = "NRMSE", x = "Model") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

