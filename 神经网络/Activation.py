import numpy as np

class activation:
    activation_support = ['tanh', 'relu', 'sigmoid']

    def __init__(self, name):
        assert name in self.activation_support, 'the {} activation unsupported in this neural network.'.format(name)
        self.name = name

    def calculate_activation(self, x, forward=True):
        if forward:
            if self.name == 'tanh':
                return np.tanh(x)
            elif self.name == 'relu':
                return np.maximum(0, x)
            elif self.name == 'sigmoid':
                return 1 / (1 + np.exp(-x))
        else:
            if self.name == 'tanh':
                return 1 - x ** 2
            elif self.name == 'relu':
                x[x > 0] = 1
                return x
            elif self.name == 'sigmoid':
                return x * (1 - x)
