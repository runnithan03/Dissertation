# Multiple Response Regression - Dissertation Project  
**Author**: Raul Unnithan
**Supervisor**: Ric Crossman
**Institution**: Durham University  
**Course**: [Mathematics and Statistics  
**Date**: April 2025  

---

## Project Overview  
This project focuses on applying **Multiple Response Regression (MRR)** techniques using methods such as **Lasso, Neural Networks and Random Forests** to predict multiple continuous response variables. The goal is to build predictive models for financial and sustainability metrics, more specifically **Return on Equity (ROE)** and **Sustainability Score**.

---

## Objectives  
- Develop predictive models for **multiple continuous response variables**.  
- Apply these methods to prevent overfitting and improve model interpretability.  
- Compare the performance on equity fund multivariate data.  

---

## Dataset  
- **File**: `clean.csv`  
- **Description**: This dataset contains financial and categorical information about equity funds, including factors that influence **ROE** and **Sustainability Score**.  
- **Key Features**:  
  - `rating` – Categorical rating of the company.  
  - `risk_rating` – Risk classification.  
  - `equity_category` – Type of equity fund classification.  
  - `holdings_n_stock` – Number of stocks held.  
  - `roe` – Return on Equity (Response Variable).  
  - `sustainability_score` – Sustainability performance (Response Variable).  

---

## Methodology  
### Some of the Techniques Used  
- **Lasso Regression** (`alpha = 1`) – Performs variable selection and regularization.  
- **Ridge Regression** (`alpha = 0`) – Shrinks coefficients but retains all predictors.  
- **Elastic Net** (`0 < alpha < 1`) – Combines Lasso and Ridge properties.  
- **Group Lasso** – Regularizes groups of predictors collectively.

### Key Steps  
1. **Preprocessing**:  
   - Frequency encoding for categorical variables.  
   - Standardisation of predictor and response variables.  

2. **Model Development**:  
   - Split data into training (80%) and test (20%) sets.  
   - Scale the predictors and responses.  

3. **Model Evaluation**:  
   - Use **Normalised Root Mean Square Error (NRMSE)** to evaluate model performance.  
   - Compare results across the model methods.
