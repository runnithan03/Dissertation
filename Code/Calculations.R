math_scores <- c(78, 85, 88, 65, 92)
science_scores <- c(80,79,88,70,74)

cor(math_scores, science_scores)

# Define matrix Y (response variables)
Y <- matrix(c(78, 85, 88, 65, 92,
              80,79,88,70,74), 
            nrow = 5, ncol = 2, byrow = FALSE)

# Standard deviations
std_devs <- apply(Y, 2, sd)

# Correlation matrix
cor_matrix <- cor(Y)

# Diagonal matrix of standard deviations
D <- diag(std_devs)

# Covariance matrix: Sigma = D * R * D
cov_matrix_manual <- D %*% cor_matrix %*% D

# Compare with built-in cov()
cov_builtin <- cov(Y)

# Print both to compare
print("Manual Covariance Matrix:")
print(cov_matrix_manual)

print("Built-in Covariance Matrix:")
print(cov_builtin)


# Define matrix X (design matrix with intercept and one predictor)
X1 <- matrix(c(1, 1, 1, 1, 1,  # Intercept column
              5, 7, 8, 3, 10), # Predictor column
            nrow = 5, ncol = 2, byrow = FALSE)

## Multiple Response Linear Regression - MANOVA 
# Compute OLS estimate B-hat
B_hat <- solve(t(X1) %*% X1) %*% t(X1) %*% Y

# Print the result
print(B_hat)

Y_hat <- X1 %*% B_hat
Y_bar <- matrix(rep(colMeans(Y), each = nrow(Y)), nrow = nrow(Y))
Y_bar

H = t(Y_hat - Y_bar)%*%(Y_hat - Y_bar)
W1 <- t(Y - Y_hat)%*%(Y-Y_hat)
T <- t(Y - Y_bar)%*%(Y - Y_bar)
T  
W = T-H
det(W)
det(H)
det(W1)

det(T)
det(W)/det(T)

1 - det(W)/det(T) # Wilks' Lambda Difference

# ANRMSE Calculation on a set of predictions
# Function to compute ANRMSE
compute_anrmse <- function(Y_true, Y_pred) {
  # Y_true: n x m matrix of actual responses
  # Y_pred: n x m matrix of predicted responses
  
  n <- nrow(Y_true)
  m <- ncol(Y_true)
  
  # Pre-allocate vector for normalised RMSE per response
  nrmse_per_response <- numeric(m)
  
  for (j in 1:m) {
    y_j <- Y_true[, j]
    y_hat_j <- Y_pred[, j]
    
    # RMSE for response j
    rmse_j <- sqrt(mean((y_j - y_hat_j)^2))
    
    # Standard deviation of y_j
    s_j <- sd(y_j)
    
    # Normalised RMSE
    nrmse_per_response[j] <- rmse_j / s_j
  }
  
  # Average normalised RMSE
  anrmse <- mean(nrmse_per_response)
  return(nrmse_per_response)
}
compute_anrmse(Y, Y_hat)
Y - Y_hat
sd(Y[,2])

# RRRR
sqrt(0.01)
# Define matrix Y (response variables)
Y <- matrix(c(78, 85, 88, 65, 92,
              80,79,88,70,74), 
            nrow = 5, ncol = 2, byrow = FALSE)

# Define matrix X (design matrix with intercept and one predictor)
X <- matrix(c(5, 7, 8, 3, 10,
              2,3,4,1,5),
             nrow = 5, ncol = 2, byrow = FALSE)

# Compute column means
X_mean <- colMeans(X)
Y_mean <- colMeans(Y)

# Mean-center the matrices
X_centered <- scale(X, center = X_mean, scale = FALSE)
Y_centered <- scale(Y, center = Y_mean, scale = FALSE)

# Print the results
print("Mean-Centered X:")
print(X_centered)

print("Mean-Centered Y:")
print(Y_centered)

# Define lambda (leave as a variable for flexibility)
lambda <- 0.01

# Compute sqrt(lambda)
sqrt_lambda <- sqrt(lambda)

# Identity matrix I of size p x p, where p is the number of predictors (columns of X)
p <- ncol(X_centered)
I_p <- diag(sqrt_lambda, p, p)  # Creates a p x p diagonal matrix with sqrt(lambda)

# Augment X*
X_star <- rbind(X_centered, I_p)

# Augment Y* (adding p rows of zeros)
Y_star <- rbind(Y_centered, matrix(0, nrow = p, ncol = ncol(Y_centered)))

# Print the results
print("Augmented X*:")
print(X_star)

print("Augmented Y*:")
print(Y_star)

B_ridge_hat <- solve(t(X_star) %*% X_star) %*% t(X_star) %*% Y_star

# Print the computed ridge-regularized coefficient matrix
print("Estimated B_hat_R^* (Ridge-Regularized Coefficients):")
print(B_ridge_hat)

Y_ridge_hat <- X_star %*% B_ridge_hat
Y_ridge_hat

Gram <- t(Y_ridge_hat) %*% Y_ridge_hat
Gram
eigen(Gram)
sqrt(eigen(Gram)$values)

svd <- svd(Y_ridge_hat)

# Extract first singular value, left singular vector, and right singular vector
d1 <- svd$d[1]  # Largest singular value
u1 <- svd$u[,1]  # First column of U
v1 <- svd$v[,1]  # First column of V

test_Y_ridge_hat <- svd$u %*% diag(svd$d) %*% t(svd$v)
test_Y_ridge_hat

# Compute rank-one approximation
rank_one_approx <- d1 * (u1 %*% t(v1))
rank_one_approx

proj <- v1 %*% t(v1)
proj

rrrr_estimator = B_ridge_hat %*% proj
rrrr_estimator

y_pred <- X_centered %*% rrrr_estimator
y_pred

Y_mean <- colMeans(Y)

# Step 2: Expand the mean vector to match the dimensions of Y_hat
Y_mean_expanded <- matrix(rep(Y_mean, each = nrow(Y_hat)), nrow = nrow(Y_hat), byrow = TRUE)

# Step 3: Un-mean center the predictions
y_pred <- Y_hat + Y_mean_expanded
y_pred

# MRCE
install.packages("glasso")  # Install if not already installed
library(glasso)

# Set regularisation parameters (you may tune these)
lambda1 <- 0.1  # Graphical Lasso penalty (for Omega)
lambda2 <- 0.1  # Ridge penalty (for B)

# Get dimensions
n <- nrow(X_centered)  # Number of observations
p <- ncol(X_centered)  # Number of predictors
m <- ncol(Y_centered)  # Number of responses

B_Ridge <- solve(t(X_centered) %*% X_centered + lambda2*diag(1,2,2)) %*% t(X_centered) %*% Y_centered
B_Ridge
sum(abs(B_Ridge))

1.95/10.92

# STEP 0: INITIALISE B AND OMEGA
# ----------------------------------------------
# Initialise B (set to zero matrix)
B <- matrix(0, nrow = p, ncol = m)

# Compute initial residuals
E <- Y_centered - X_centered %*% B

# Compute initial error covariance N#_E
Sigma_E <- (t(E) %*% E) / n  

# Initialise Omega as the inverse of Sigma_E with regularisation
Omega <- solve(Sigma_E + lambda1 * diag(m))
Omega

# SINGLE ITERATION OF MRCE

# STEP 1: UPDATE B USING ALGORITHM 1 (Equation 4.4)
# --------------------------------------------------
# Compute H = X^T Y Omega
H <- t(X_centered) %*% Y_centered %*% Omega

# Compute S = X^T X
S <- t(X_centered) %*% X_centered

# Coordinate-wise update for each (r, c) entry of B
B_old <- B  # Store old B for comparison

for (r in 1:p) {
  for (c in 1:m) {
    # Compute update for B_rc using coordinate-wise soft-thresholding
    U_rc <- sum(S[r, ] %*% B %*% Omega[, c])  # FIXED LINE
    B[r, c] <- sign(B[r, c] + (H[r, c] - U_rc) / (S[r, r] * Omega[c, c])) * 
      max(abs(B[r, c] + (H[r, c] - U_rc) / (S[r, r] * Omega[c, c])) - 
            (n * lambda2) / (S[r, r] * Omega[c, c]), 0)
  }
}
B_old
B

# STEP 2: UPDATE OMEGA USING GRAPHICAL LASSO (Equation 4.5)
# -----------------------------------------------------------
# Compute residuals E using updated B
E <- Y_centered - X_centered %*% B

# Compute new error covariance N#_E
Sigma_E <- (t(E) %*% E) / n

# Apply Graphical Lasso to estimate sparse precision matrix Omega
glasso_fit <- glasso(Sigma_E, rho = lambda1)
Omega <- glasso_fit$wi  # 'wi' stores the estimated precision matrix

# Create empty matrix for Omega update
# Manually compute the first iteration of Graphical Lasso
Omega_new <- solve(Sigma_E + lambda1 * diag(m))  # Regularized inverse
Omega_new

# j = 1
j = 1
# Extract submatrix Sigma_E[-j, -j] and corresponding off-diagonal values
Sigma_jj <- Sigma_E[-j, -j]
Sigma_j <- Sigma_E[-j, j]

# Solve for the inverse of Sigma_jj (needed for beta computation)
Omega_jj <- solve(Sigma_jj)

# Compute beta_j (Lasso regression coefficient estimate)
beta_j <- -Omega_jj %*% solve(Sigma_j)

# Apply soft-thresholding to enforce sparsity
beta_j <- sign(beta_j) * pmax(abs(beta_j) - lambda1, 0)

# Update off-diagonal elements in Omega using the estimated beta_j
Omega_new[j, -j] <- beta_j
Omega_new[-j, j] <- beta_j

# **Diagonal Element Update (Following Algorithm 4.5)**
# Instead of using a naive inverse, use the structured update:
diag_update <- Sigma_E[j, j] - sum(Sigma_j * beta_j)

# Ensure positive definiteness of Omega
if (diag_update > 0) {
  Omega_new[j, j] <- 1 / diag_update  # Update diagonal
} else {
  Omega_new[j, j] <- Omega[j, j]  # Retain old value to avoid instability
}

for (j in 1:q) {
  # Extract submatrix Sigma_E[-j, -j] and corresponding off-diagonal values
  Sigma_jj <- Sigma_E[-j, -j]
  Sigma_j <- Sigma_E[-j, j]
  
  # Solve for the inverse of Sigma_jj (needed for beta computation)
  Omega_jj <- solve(Sigma_jj)
  
  # Compute beta_j (Lasso regression coefficient estimate)
  beta_j <- -Omega_jj %*% Sigma_j
  
  # Apply soft-thresholding to enforce sparsity
  beta_j <- sign(beta_j) * pmax(abs(beta_j) - lambda1, 0)
  
  # Update off-diagonal elements in Omega using the estimated beta_j
  Omega_new[j, -j] <- beta_j
  Omega_new[-j, j] <- beta_j
  
  # **Diagonal Element Update (Following Algorithm 4.5)**
  # Instead of using a naive inverse, use the structured update:
  diag_update <- Sigma_E[j, j] - sum(Sigma_j * beta_j)
  
  # Ensure positive definiteness of Omega
  if (diag_update > 0) {
    Omega_new[j, j] <- 1 / diag_update  # Update diagonal
  } else {
    Omega_new[j, j] <- Omega[j, j]  # Retain old value to avoid instability
  }
}

# Print updated Omega after one iteration
print("Updated Omega after one iteration:")
print(Omega_new)

all.equal(Omega, Omega_new)

# FINAL OUTPUTS AFTER ONE ITERATION
# ----------------------------------------------
print("Updated B after 1 iteration:")
print(B)

print("Updated Omega after 1 iteration:")
print(Omega)

# Compute final predictions for this iteration
Y_pred <- X_centered %*% B

print("Predicted Y (Mean-Centered) after 1 iteration:")
print(Y_pred)

# If needed: un-mean center Y_pred using column means of original Y
Y_mean <- colMeans(Y_centered)
Y_pred_final <- Y_pred + matrix(rep(Y_mean, each = nrow(Y_pred)), nrow = nrow(Y_pred), byrow = TRUE)

print("Final un-mean centered predictions after 1 iteration:")
print(Y_pred_final)

# CRRFs
# Define the Left Node Matrix (X_L) for X1 < 7
X_L <- matrix(c(5, 2, 78, 80,
                3, 1, 65, 70), 
              nrow = 2, byrow = TRUE)

# Define the Right Node Matrix (X_R) for X1 >= 7
X_R <- matrix(c(7, 3, 85, 79,
                8, 4, 88, 88,
                10, 5, 92, 74), 
              nrow = 3, byrow = TRUE)

# Print the matrices
print("X_L:")
print(X_L)

print("X_R:")
print(X_R)

# Extract the last two columns from X_L
Y_L <- X_L[, 3:4]

# Extract the last two columns from X_R
Y_R <- X_R[, 3:4]

Y_bar_L <- colMeans(Y_L)
Y_bar_R <- colMeans(Y_R)

Y_bar_L
Y_bar_R

n_L = 2
n_R = 3

# Compute covariance matrices using the given formula
Sigma_L <- (1 / (n_L - 1)) * t(Y_L - Y_bar_L) %*% (Y_L - Y_bar_L)
Sigma_R <- (1 / (n_R - 1)) * t(Y_R - Y_bar_R) %*% (Y_R - Y_bar_R)

Sigma_L
Sigma_R

# Compute the Euclidean distance between Sigma_L and Sigma_R
euclidean_distance <- function(Sigma_L, Sigma_R) {
  # Extract upper triangular part (including diagonal)
  diff_matrix <- Sigma_L - Sigma_R
  distance <- sqrt(sum(diff_matrix^2))
  return(distance)
}

# Apply the function to compute the distance
distance_value <- sqrt(2*3)*euclidean_distance(Sigma_L, Sigma_R)
distance_value

distance_value/sqrt(6)

# Cholesky-Decomposed XGBoost

# Sample dataset (from your table)
data <- data.frame(
  X1 = c(5, 7, 8, 3, 10),  # Hours Studied
  X2 = c(2, 3, 4, 1, 5),    # Time Spent on Papers
  Y1 = c(78, 85, 88, 65, 92),  # Math Scores
  Y2 = c(80, 79, 88, 70, 74)   # Science Scores
)

# Extract the response variables (Y1 and Y2)
Y <- as.matrix(data[, c("Y1", "Y2")])
colMeans(Y)

# Compute the empirical covariance matrix
Sigma_Y <- cov(Y)

# Perform Cholesky decomposition
L <- chol(Sigma_Y)

# Print results
print("Covariance Matrix of Y:")
print(Sigma_Y)

print("Cholesky Decomposition (Lower Triangular Matrix):")
print(t(L))  # Transposing since chol() returns upper-triangular

# Compute the inverse of L
L_inv <- solve(t(L))  # Cholesky returns upper triangular, so transpose it

# Load necessary libraries
library(xgboost)

# Define predictor variables
X <- as.matrix(data[, c("X1", "X2")])

# Transform Y using inverse Cholesky
Y_transformed <- t(L_inv %*% t(Y))  # Ensure correct row format

# Check dimensions
print(dim(X))  # Should be (5,2)
print(dim(Y_transformed))  # Should be (5,2)

# Convert transformed Y into individual response vectors
Y_transformed_1 <- Y_transformed[,1]  # First response variable
Y_transformed_2 <- Y_transformed[,2]  # Second response variable

# 1 Manual XGBoost Iteration
Y_tilde <- Y_transformed
colMeans(Y_tilde)
# Compute residuals
residuals <- (Y_tilde - matrix(rep(colMeans(Y_tilde), each = nrow(Y_tilde)), nrow = nrow(Y_tilde), byrow = FALSE))[,1] # XGBoost for 1 response
print(residuals)

g1 <- t(t(2*residuals))
h1 <- matrix(rep(2, 5), nrow = 5, ncol = 1)

score1 <- (sum(g1))^2/(sum(h1)+0.1)
score1

# Create the dataset
data <- data.frame(cbind(X, Y_transformed))
data$g1 <- 2*t(t(residuals))
data$h1 <- 2

# Create two subsets
data_L <- subset(data, X1 < 7)    # Rows where X1 < 7
data_R <- subset(data, X1 >= 7)  # Rows where X1 >= 7
lambda = 0.1

G_L = sum(data_L$g1)
H_L = sum(data_L$h1)
S_L = G_L^2/(H_L + lambda)
S_L

G_R = sum(data_R$g1)
H_R = sum(data_R$h1)
S_R = G_R^2/(H_R + lambda)
S_R

S_L+S_R

w_L = -G_L/(H_L + lambda)
w_R = -G_R/(H_R + lambda)
data_L$Initial <- mean(data$Y1)
data_L$w_L = w_L

data_R$Initial <- mean(data$Y1)
data_R$w_R = w_R

eta = 0.3

# Compute the initial transformed response means
Initial <- colMeans(Y_tilde)

# Compute new transformed predictions using the update rule
data_L$New <- data_L$Initial + eta * data_L$w_L
data_R$New <- data_R$Initial + eta * data_R$w_R

# Ensure data_L$New and data_R$New are matrices with correct dimensions
New_L <- rbind(data_L$New, rep(0, length(data_L$New)))  # Add placeholder for Y2
New_R <- rbind(data_R$New, rep(0, length(data_R$New)))  # Add placeholder for Y2

# Apply Cholesky matrix L to revert predictions to original space
Y_hat_L <- L %*% New_L
Y_hat_R <- L %*% New_R

# Store predictions in the data frames
data_L$predict <- t(Y_hat_L)[,1]  # Convert back to column format
data_R$predict <- t(Y_hat_R)[,1]  # Convert back to column format
