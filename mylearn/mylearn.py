"""
mylearn: my own machine-learning library, written from scratch.

Author:  Anish Lal Manandhar
Started: 13-14 November 2026 (ML final prep, Code It Yourself workbook, levels 0-4)

Every function here was written by me, line by line, and checked against
numpy / sklearn. The first model: linear regression trained by batch
gradient descent (batch_gd), finished at 07:21 on 14 Nov 2026.
"""
import numpy as np


# ---------------------------------------------------------------- level 0
def add_intercept(X):
    """(m, n) -> (m, n+1) with a column of 1s on the LEFT."""
    intercept_col = np.ones((X.shape[0], 1))          # (m, 1): a COLUMN, not a flat (m,)
    return np.concatenate((intercept_col, X), axis=1)


def predict(X, theta):
    """(m, n) @ (n,) -> (m,). Each prediction = one ROW of X dotted with theta (sum across)."""
    return X @ theta


def grad_loop(X, y, theta):
    """Gradient of J = 1/2 * sum((yhat - y)^2), with plain for-loops."""
    m, n = X.shape
    grad = np.zeros(n)
    e = X @ theta - y              # all errors, shape (m,)
    for j in range(n):             # each theta_j = each COLUMN
        for i in range(m):         # every sample = go DOWN the column
            grad[j] += X[i, j] * e[i]
    return grad


def grad_vec(X, y, theta):
    """The same gradient in one line: X.T @ e IS the double for-loop above."""
    return X.T @ (X @ theta - y)


# ---------------------------------------------------------------- level 1
def standardize(X_train, X_test):
    """Learn mean and std from TRAIN only, apply them to both (no leakage)."""
    mu = X_train.mean(axis=0)      # axis=0: collapse the rows -> one value per column
    sd = X_train.std(axis=0)
    return (X_train - mu) / sd, (X_test - mu) / sd


# ---------------------------------------------------------------- level 2
def mse(yhat, y):
    return ((yhat - y) ** 2).sum() / (y.shape[0])


def r2(yhat, y):
    ss_res = ((y - yhat) ** 2).sum()          # my model's error
    ss_tot = ((y - y.mean()) ** 2).sum()      # the "always predict the mean" error
    return 1 - (ss_res / ss_tot)


# ---------------------------------------------------------------- level 3
def closed_form(X, y):
    return np.linalg.inv(X.transpose() @ X) @ (X.transpose() @ y)


def ridge_closed_form(X, y, lam):
    n = X.shape[1]
    return np.linalg.inv((X.transpose() @ X) + lam * np.eye(n)) @ (X.transpose() @ y)


# ---------------------------------------------------------------- level 4
def batch_gd(X, y, lr=0.1, max_iter=1000):
    """My first model: linear regression by batch gradient descent.
    Each iteration: predict -> store the loss -> gradient from the ERRORS -> one update."""
    theta = np.zeros(X.shape[1])
    losses = []
    m = X.shape[0]
    for i in range(max_iter):
        pred = X @ theta
        losses.append(mse(pred, y))                 # loss: 1 number, stored
        gradient = ((X.T @ (pred - y)) / m)         # errors: m numbers, used
        theta = theta - (lr * gradient)
    return theta, losses


# ---------------------------------------------------------------- demo
if __name__ == '__main__':
    # Train my model on the professor's diabetes data and compare with sklearn.
    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    X, y = load_diabetes(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
    X_tr, X_te = standardize(X_tr, X_te)
    X_tr, X_te = add_intercept(X_tr), add_intercept(X_te)

    theta, losses = batch_gd(X_tr, y_tr, lr=0.1, max_iter=5000)
    print(f'my model   test MSE {mse(X_te @ theta, y_te):8.2f} · R² {r2(X_te @ theta, y_te):.4f}')
    sk = LinearRegression().fit(X_tr[:, 1:], y_tr)
    print(f'sklearn    test MSE {mse(sk.predict(X_te[:, 1:]), y_te):8.2f} · R² {r2(sk.predict(X_te[:, 1:]), y_te):.4f}')

    np.save('my_first_theta.npy', theta)
    print('saved the trained model (theta) to my_first_theta.npy')
