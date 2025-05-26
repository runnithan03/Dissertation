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
Omega

# Step 1: Initial precision estimate
Omega_new <- solve(Sigma_E + lambda1 * diag(m))

# Step 2: Partition for column j
j <- 1
not_j <- setdiff(1:m, j)
Omega_sub <- Omega_new[not_j, not_j]
Omega_sub^-0.5

s_sub <- Sigma_E[not_j, j]

# Step 3: Eigen-decomposition for whitening
e <- eigen(Omega_sub)
if (length(e$values) == 1) {
  Omega_sqrt <- sqrt(Omega_sub)
  Omega_inv_sqrt <- 1 / sqrt(Omega_sub)
} else {
  Omega_sqrt <- e$vectors %*% diag(sqrt(e$values)) %*% t(e$vectors)
  Omega_inv_sqrt <- e$vectors %*% diag(1 / sqrt(e$values)) %*% t(e$vectors)
}

# Step 4: Create Lasso problem
A <- Omega_sqrt
b <- Omega_inv_sqrt %*% s_sub

# Step 5: Manual coordinate descent (1 step)
beta <- rep(0, length(b))
k <- 1

if (length(b) == 1) {
  # Scalar case: residual is just b
  a_k <- A
  residual <- b
} else {
  a_k <- A[, k]
  residual <- b - A[, -k, drop = FALSE] %*% beta[-k]
}

z_k <- sum(a_k * residual)
beta[k] <- sign(z_k) * max(abs(z_k) - lambda1, 0)



# Step 6: Recover precision sub-column
omega_minus_j_j <- Omega_sub %*% beta

Omega_sub
omega_minus_j_j

# Step 2: Partition for column j
j <- 2
not_j <- setdiff(1:m, j)
Omega_sub <- Omega_new[not_j, not_j]
Omega_sub^-0.5

s_sub <- Sigma_E[not_j, j]

# Step 3: Eigen-decomposition for whitening
e <- eigen(Omega_sub)
if (length(e$values) == 1) {
  Omega_sqrt <- sqrt(Omega_sub)
  Omega_inv_sqrt <- 1 / sqrt(Omega_sub)
} else {
  Omega_sqrt <- e$vectors %*% diag(sqrt(e$values)) %*% t(e$vectors)
  Omega_inv_sqrt <- e$vectors %*% diag(1 / sqrt(e$values)) %*% t(e$vectors)
}

# Step 4: Create Lasso problem
A <- Omega_sqrt
b <- Omega_inv_sqrt %*% s_sub

# Step 5: Manual coordinate descent (1 step)
beta <- rep(0, length(b))
k <- 1

if (length(b) == 1) {
  # Scalar case: residual is just b
  a_k <- A
  residual <- b
} else {
  a_k <- A[, k]
  residual <- b - A[, -k, drop = FALSE] %*% beta[-k]
}

z_k <- sum(a_k * residual)
beta[k] <- sign(z_k) * max(abs(z_k) - lambda1, 0)
beta


# Step 6: Recover precision sub-column
omega_minus_j_j <- Omega_sub %*% beta

Omega_sub
omega_minus_j_j

