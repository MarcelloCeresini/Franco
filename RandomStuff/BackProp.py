import numpy as np
import math


def sigmoid(x): return 1 / (1 + math.exp(-x))


def sigderiv(x): return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    if x >= 0:
        return x
    else:
        return 0


def reluderiv(x):
    if x >= 0:
        return 1
    else:
        return 0


def activate(x): return sigmoid(x)


def actderiv(x): return sigderiv(x)


dim = [8, 3, 8]  # neurons in each layer

l = len(dim)
w, b = [], []

for i in range(1, l):
    w.append(np.random.rand(dim[i - 1], dim[i]))
    b.append(np.random.rand(dim[i]))

mu = 1e-3

z, a, d = [], [], []

for i in range(0, l):
    a.append(np.zeros(dim[i]))

for i in range(1, l):
    z.append(np.zeros(dim[i]))
    d.append(np.zeros(dim[i]))


def update(x, y):
    # input
    a[0] = x
    # feed forward
    for i in range(0, l-1):
        z[i] = np.dot(a[i], w[i]) + b[i]
        a[i+1] = np.vectorize(activate)(z[i])
    # output error
    d[l-2] = (y - a[l-1])*np.vectorize(actderiv)(z[l-2])
    # backpropagation
    for i in range(l-3, -1, -1):
        d[i] = np.dot(w[i+1], d[i+1]) * np.vectorize(actderiv)(z[i+1])
    # updating
    for i in range(0, l-1):
        for k in range(0, dim[i+1]):
            for j in range(0, dim[i]):
                w[i][j, k] = w[i][j, k] + mu * a[i][j] * d[i][k]
            b[i][k] = b[i][k] + mu * d[i][k]
    return np.sum((y-a[l-1])**2)


def epoch(data):
    e = 0
    for (x, y) in data:
        e += update(x, y)
    return e


X = np.eye(8)
data = zip(X, X)

final_error = .003
dist = epoch(data)

while dist > final_error:
    print("Distance: ", dist)
    dist = epoch(data)
