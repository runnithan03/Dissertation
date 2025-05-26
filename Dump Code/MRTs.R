# Install necessary packages if not already installed
install.packages(c("randomForestSRC", "Rcpp", "caret", "MASS"))

# Load required libraries
library(randomForestSRC)
library(caret)
library(MASS)

# Define the covariance regression function directly (paste the function you provided here)

# ---------------------------- #
# STEP 1: Load and Preprocess Data
# ---------------------------- #

# Load dataset
data <- read.csv("C:\\Users\\raulu\\OneDrive\\Documents\\4th Year\\Dissertation\\Code\\clean.csv")

# Frequency encode categorical variables
for (col in c("rating", "category", "risk_rating")) {
  freq <- table(data[[col]]) / nrow(data)
  data[[col]] <- freq[data[[col]]]
}

# Define features (X) and response variables (Y)
X <- data[, !names(data) %in% c("roe", "sustainability_score")]
Y <- data[, c("roe", "sustainability_score")]

# Train-test split
set.seed(42)
train_idx <- sample(seq_len(nrow(X)), size = 0.8 * nrow(X))
X_train <- X[train_idx, ]
X_test  <- X[-train_idx, ]
Y_train <- Y[train_idx, ]
Y_test  <- Y[-train_idx, ]

# ---------------------------- #
# STEP 2: Train CovRegRF Model (Directly Using covregrf Function)
# ---------------------------- #

# Define hyperparameters manually since caret doesn't support CovRegRF
params.rfsrc <- list(ntree = 100, mtry = ceiling(ncol(X) / 3), nsplit = max(round(nrow(X) / 50), 10))

# Train model using the custom covregrf function
covreg_model <- covregrf(formula = "roe + sustainability_score ~ .", data = data, 
                         params.rfsrc = params.rfsrc)

# ---------------------------- #
# STEP 3: Make Predictions
# ---------------------------- #

# Predict response covariance matrices on the test set (use the custom model)
pred_Y <- predict(covreg_model, X_test)  # Modify based on your actual prediction method

# ---------------------------- #
# STEP 4: Compute ARSR = (NRMSE / SD)
# ---------------------------- #

# Compute sample covariance matrix
compute_covariance <- function(Y_subset) {
  if (nrow(Y_subset) > 1) {
    return(cov(Y_subset))
  } else {
    return(matrix(0, ncol = ncol(Y_subset), nrow = ncol(Y_subset)))
  }
}

# Compute Mean Squared Error (MSE)
mse_cov <- function(true_cov, estimated_cov) {
  return(mean((true_cov - estimated_cov)^2))
}

# Compute NRMSE
compute_nrmse <- function(true_covs, estimated_covs) {
  mse_values <- sapply(seq_len(nrow(true_covs)), function(i) mse_cov(true_covs[[i]], estimated_covs[[i]]))
  rmse <- sqrt(mean(mse_values))
  range_cov <- max(sapply(true_covs, function(cov) max(cov))) - min(sapply(true_covs, function(cov) min(cov)))
  return(rmse / range_cov)
}

# Compute Standard Deviation (SD) of true covariances
compute_sd_cov <- function(true_covs) {
  return(sd(sapply(true_covs, function(cov) sum(cov))))
}

# Compute true covariance matrices
true_test_covs <- lapply(seq_len(nrow(Y_test)), function(i) compute_covariance(Y_test[i, , drop = FALSE]))

# Compute NRMSE and SD
nrmse_value <- compute_nrmse(true_test_covs, pred_Y)
sd_value <- compute_sd_cov(true_test_covs)

# Compute ARSR metric
arsr <- nrmse_value / sd_value
print(paste("ARSR Metric:", arsr))
