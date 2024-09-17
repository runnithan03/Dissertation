import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("Generating sample dataset...")

# Generate a sample dataset
X_train, y_train = make_classification(n_samples=100, n_features=20, n_informative=2, n_redundant=10, random_state=42)

# Print the shapes of X_train and y_train to verify
print("Shapes of X_train and y_train:")
print(X_train.shape)  # Should output (100, 20)
print(y_train.shape)  # Should output (100,)

print("Fitting Lasso (L1) Regularization model...")

# Lasso (L1) Regularization
model_lasso = LogisticRegression(penalty='l1', solver='saga', max_iter=10000).fit(X_train, y_train)
print("Lasso model coefficients:", model_lasso.coef_)
print("Lasso model accuracy on training data:", accuracy_score(y_train, model_lasso.predict(X_train)))

print("Fitting Ridge (L2) Regularization model...")

# Ridge (L2) Regularization
model_ridge = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=10000).fit(X_train, y_train)
print("Ridge model coefficients:", model_ridge.coef_)
print("Ridge model accuracy on training data:", accuracy_score(y_train, model_ridge.predict(X_train)))

print("Script completed.")

