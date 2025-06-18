import numpy as np
from sklearn.linear_model import LinearRegression

class MultiRegression:
    def __init__(self):
        # Initialize parameters (will be set in fit)
        self.params = None

    def fit(self, X, y):
        # Add a bias (intercept) column of ones to the input features
        bias = np.ones(len(X))
        X_bias = np.c_[bias, X]
        # Compute (X^T X)
        inner_part = np.transpose(X_bias) @ X_bias
        # Compute the inverse of (X^T X)
        inverse = np.linalg.inv(inner_part)
        # Compute (X^T X)^-1 X^T
        X_part = inverse @ np.transpose(X_bias)
        # Compute the least squares estimate: (X^T X)^-1 X^T y
        lse = X_part @ y
        # Store the parameters
        self.params = lse
        return self.params

    def predict(self, Xi):
        # Add a bias (intercept) column of ones to the test features
        bias_test = np.ones(len(Xi))
        X_test = np.c_[bias_test, Xi]
        # Predict using the learned parameters
        y_hat = X_test @ self.params
        return y_hat

if __name__ == '__main__':
    # Example training data (4 samples, 3 features)
    X = np.array([
        [1, 4, 6],
        [2, 5, 10],
        [3, 8, 11],
        [4, 2, 13]
    ])

    # Target values for training data
    y = np.array([1, 6, 8, 12])

    # Create and train the custom MultiRegression model
    lr = MultiRegression()
    b_hat = lr.fit(X, y)
    # print(b_hat)  # Uncomment to see learned parameters

    # Example test data (1 sample, 3 features)
    X_test = np.array([
        [5, 3, 33]
    ])

    # Predict using the custom model
    y_hat = lr.predict(X_test)
    print(f'Hardcoded model : {y_hat}')

    # Compare with scikit-learn's LinearRegression
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X_test)
    print(f'Sklearn model : {y_pred}')