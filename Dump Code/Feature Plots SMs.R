# Install required packages
install.packages("MRCE", dependencies = TRUE)
install.packages("ggplot2")
install.packages("caret")
install.packages("Matrix")  
install.packages("rrpack")  
install.packages("reshape2")

# Load libraries
library(MRCE)
library(ggplot2)
library(caret)
library(Matrix)
library(rrpack)
library(reshape2)

# Load dataset
data <- read.csv("C:\\Users\\raulu\\OneDrive\\Documents\\4th Year\\Dissertation\\Code\\clean.csv")
dim(data)

## Data Pre-Processing
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

# Split dataset into training and test sets
set.seed(1)
train_index <- sample(seq_len(nrow(data)), size = 0.8 * nrow(data))
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

X_train <- scale(as.matrix(train_data[, predictor_vars]))
Y_train <- scale(as.matrix(train_data[, response_vars]))
X_test <- scale(as.matrix(test_data[, predictor_vars]), center = attr(X_train, "scaled:center"), scale = attr(X_train, "scaled:scale"))
Y_test <- scale(as.matrix(test_data[, response_vars]), center = attr(Y_train, "scaled:center"), scale = attr(Y_train, "scaled:scale"))

# Function to calculate NRMSE
calculate_nrmse <- function(predictions, actual) {
  residuals <- actual - predictions
  rmse <- apply(residuals, 2, function(col) sqrt(mean(col^2)))
  nrmse <- mean(rmse / apply(actual, 2, sd, na.rm = TRUE))
  return(nrmse)
}

fit_mrce <- function(X_train, Y_train, X_test, Y_test, method = "single") {
  k_folds <- 5
  folds <- sample(1:k_folds, nrow(X_train), replace = TRUE)
  
  best_nrmse <- Inf
  best_mrce_model <- NULL
  best_lam1 <- NA
  best_lam2 <- NA
  lambda_grid <- seq(0.01, 0.2, length.out = 5)
  
  for (lam1 in lambda_grid) {
    for (lam2 in lambda_grid) {
      cv_nrmse <- c()
      
      for (k in 1:k_folds) {
        train_idx <- folds != k
        val_idx <- folds == k
        
        model <- mrce(X_train[train_idx, ], Y_train[train_idx, ], 
                      lam1 = lam1, lam2 = lam2, method = method)
        predictions <- X_train[val_idx, ] %*% model$B
        cv_nrmse <- c(cv_nrmse, calculate_nrmse(predictions, Y_train[val_idx, ]))
      }
      
      avg_nrmse <- mean(cv_nrmse)
      if (avg_nrmse < best_nrmse) {
        best_nrmse <- avg_nrmse
        best_mrce_model <- model
        best_lam1 <- lam1
        best_lam2 <- lam2
      }
    }
  }
  
  test_predictions <- X_test %*% best_mrce_model$B
  
  return(list(
    model = best_mrce_model,
    nrmse = calculate_nrmse(test_predictions, Y_test),
    best_lam1 = best_lam1,
    best_lam2 = best_lam2
  ))
}


# Reduced-rank ridge regression
fit_rrridge <- function(X_train, Y_train, X_test, Y_test, rank = 2) {
  # Fit Reduced-Rank Regression with Ridge Regularisation
  model <- rrr.fit(X = X_train, Y = Y_train, nrank = rank)
  
  predictions <- X_test %*% model$coef
  return(list(model = model, nrmse = calculate_nrmse(predictions, Y_test)))
}

# Fit MRCE and RRRidge
mrce_result <- fit_mrce(X_train, Y_train, X_test, Y_test, method = "single")
best_mrce_model <- mrce_result$model
mrce_result$best_lam1
mrce_result$best_lam2

mrce_nrmse <- mrce_result$nrmse

# Identify predictors that were excluded (i.e., all coefficients = 0 across responses)
excluded_mrce <- rowSums(best_mrce_model$B == 0) == ncol(best_mrce_model$B)
excluded_predictors <- predictor_vars[excluded_mrce]

# Print excluded predictors
cat("Predictors excluded (all-zero coefficients in MRCE):\n")
print(excluded_predictors)

rrridge_result <- fit_rrridge(X_train, Y_train, X_test, Y_test, rank = 2)
best_rrridge_model <- rrridge_result$model
rrridge_nrmse <- rrridge_result$nrmse

length(predictor_vars)

# Polynomial Features
X_train_poly <- poly(X_train, degree = 2, raw = TRUE)
X_test_poly <- poly(X_test, degree = 2, raw = TRUE)

mrce_poly_result <- fit_mrce(X_train_poly, Y_train, X_test_poly, Y_test, method = "single")
rrridge_poly_result <- fit_rrridge(X_train_poly, Y_train, X_test_poly, Y_test, rank = 2)

# Interaction Features
X_train_interact <- model.matrix(~ .^2, data = as.data.frame(X_train))[, -1]
X_test_interact <- model.matrix(~ .^2, data = as.data.frame(X_test))[, -1]

mrce_interact_result <- fit_mrce(X_train_interact, Y_train, X_test_interact, Y_test, method = "single")
rrridge_interact_result <- fit_rrridge(X_train_interact, Y_train, X_test_interact, Y_test, rank = 2)

# Combine results
results <- data.frame(
  Model = c("MRCE", "RRRidge", "MRCE (Poly)", "RRRidge (Poly)", "MRCE (Interact)", "RRRidge (Interact)"),
  NRMSE = c(mrce_nrmse, rrridge_nrmse, mrce_poly_result$nrmse, rrridge_poly_result$nrmse, mrce_interact_result$nrmse, rrridge_interact_result$nrmse)
)

# Print results
print(results)

# Extract MRCE coefficients
coefficients_mrce <- as.data.frame(best_mrce_model$B)
colnames(coefficients_mrce) <- response_vars
coefficients_mrce$Feature <- predictor_vars
coefficients_mrce$Model <- "MRCE"

# Extract RRRidge coefficients
coefficients_rrridge <- as.data.frame(best_rrridge_model$coef)
colnames(coefficients_rrridge) <- response_vars
coefficients_rrridge$Feature <- predictor_vars
coefficients_rrridge$Model <- "RRRidge"

# Combine both coefficient dataframes
coefficients_combined <- rbind(coefficients_mrce, coefficients_rrridge)
dim(coefficients_combined)

# Reshape for ggplot
library(reshape2)
coefficients_long <- melt(coefficients_combined, id.vars = c("Feature", "Model"), variable.name = "Response", value.name = "Coefficient")

# Plot feature importance
library(ggplot2)
ggplot(coefficients_long, aes(x = Feature, y = Coefficient, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  coord_flip() +
  facet_wrap(~Response, scales = "free") +
  labs(title = "Feature Coefficients (MRCE vs. RRRidge)", y = "Coefficient Value", x = "Feature") +
  theme_minimal()


