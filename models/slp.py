# slp.py

import math
import random

from utils.math_functions import sigmoid

class SLP:
    def __init__(self, input_size, output_size):
        self.W = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(output_size)]
        self.b = [random.uniform(-1, 1) for _ in range(output_size)]

    def forward(self, x):
        y = []
        for i in range(len(self.W)):
            z = sum(self.W[i][j] * x[j] for j in range(len(x))) + self.b[i]
            y.append(sigmoid(z))
        return y

    def backward(self, x, y_true, lr):
        y_pred = self.forward(x)

        for i in range(len(self.W)):
            dy_dz = y_pred[i] * (1 - y_pred[i])

            error = y_pred[i] - y_true[i]

            for j in range(len(self.W[i])):
                grad_w = error * dy_dz * x[j]
                self.W[i][j] -= lr * grad_w

            grad_b = error * dy_dz
            self.b[i] -= lr * grad_b

