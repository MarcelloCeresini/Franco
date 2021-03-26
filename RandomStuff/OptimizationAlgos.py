import numpy as np

f = lambda x: x ** 2
Df = lambda x: 2 * x
x0 = 3
eps = 1e-5
max_iter = 50


def newton(f, Df, x0, eps, max_iter):
    xn = x0
    for n in range(0, max_iter):
        fxn = f(xn)
        if abs(fxn) < eps:
            print("Found solution after {0} iterations.".format(n))
            return xn
        Dfxn = Df(xn)
        if Dfxn == 0:
            print("Zero derivative, no solution found")
            return None
        xn = xn - fxn / Dfxn
    print("Exceeded maximum iteration, no solution found")
    return None


lr = 0.05
dim = 6
n = 700
X = np.zeros(dim, n)
y = np.zeros(dim)
loss = lambda theta, Xl, yl: (1 / (2 * n)) * ((theta[0] + theta[1] * Xl) - yl).sum()
d_loss = lambda theta, Xl, yl: (1 / n) * (((theta[0] + theta[1] * Xl) - yl) * Xl).sum()


def grad_descent(X, y, loss, d_loss, theta, iter=100, lr=0.01):
    xn = x0
    theta_history = np.zeros(len(theta), iter)
    cost_history = np.zeros(len(theta))
    for i in range(0, iter):
        theta = theta - d_loss(theta, X, y) * lr
        theta_history[i] = theta
        cost_history[i] = loss(theta, X, y)
    return xn
