# single_neuron.py

import math
import random

from utils.math_functions import sigmoid

class SingleNeuron:
    def __init__(self):
        self.w = random.uniform(-1, 1)
        self.b = random.uniform(-1, 1)
    
    def forward(self, x):
        z = self.w * x + self.b
        y = sigmoid(z)
        return y

    def backward(self, x, y_true, lr):
        y_pred = self.forward(x)

        dy_dz = y_pred * (1 - y_pred)
        
        error = y_pred - y_true

        grad_w = error * dy_dz * x
        grad_b = error * dy_dz

        self.w -= lr * grad_w
        self.b -= lr * grad_b

