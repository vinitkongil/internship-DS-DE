import numpy as np

class LinearRegression:
    def __init__(self):
        self.b0, self.b1 = 0, 0

    def fit(self, x, y):
        n = len(x)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        ssxy, ssx = 0, 0
        for i in range(n):
            ssxy += (x[i] - x_mean) * (y[i] - y_mean)
            ssx += (x[i] - x_mean) ** 2
        self.b1 = ssxy / ssx
        self.b0 = y_mean - self.b1 * x_mean  # intercept
        return self.b0, self.b1

    def predict(self, xi):
        return self.b0 + self.b1 * xi

if __name__ == "__main__":
    height = np.array([160, 171, 182, 180, 154])
    weight = np.array([72, 76, 77, 83, 76])
    lr = LinearRegression()
    b0, b1 = lr.fit(x=height, y=weight)
    print(f"the value of b0 is {b0} and the value of b1 is {b1}")
    xi = np.array([170])
    y_hat = lr.predict(xi)
    print(f'the weight for the height {xi[0]} is {y_hat[0]}')