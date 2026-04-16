# mlp.py

import math
import random

from utils.math_functions import sigmoid

class MLP:
    def __init__(self, layer_sizes):
        self.W = []
        self.b = []

        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            output_size = layer_sizes[i+1]
            W_layer = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(output_size)]
            b_layer = [random.uniform(-1, 1) for _ in range(output_size)]
            self.W.append(W_layer)
            self.b.append(b_layer)

    def forward(self, x):
        ys = [x]

        for layer in range(len(self.W)):
            y = []
            for i in range(len(self.W[layer])):
                z = sum(self.W[layer][i][j] * x[j] for j in range(len(x))) + self.b[layer][i]
                y.append(sigmoid(z))
            x = y
            ys.append(y)

        return ys

    def backward(self, x, y_true, lr):
        ys = self.forward(x)
        y_pred = ys[-1]

        ds = []
        for i in range(len(y_pred)):
            dy_dz = y_pred[i] * (1 - y_pred[i])
            error = y_pred[i] - y_true[i]
            ds.append(error * dy_dz)

        for layer in reversed(range(len(self.W))):
            ys_prev = ys[layer]
            d_next = [0] * len(ys_prev)

            for i in range(len(self.W[layer])):
                for j in range(len(self.W[layer][i])):
                    grad_w = ds[i] * ys_prev[j]
                    self.W[layer][i][j] -= lr * grad_w
                    d_next[j] += ds[i] * self.W[layer][i][j]

                grad_b = ds[i]
                self.b[layer][i] -= lr * grad_b

            if layer > 0:
                for i in range(len(d_next)):
                    d_next[j] *= ys_prev[j] * (1 - ys_prev[j])

            ds = d_next

