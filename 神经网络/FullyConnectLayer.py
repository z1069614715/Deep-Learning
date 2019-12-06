import numpy as np

class FullyConnectLayer:
    param_count = []
    param = []
    forward_res = []
    temp = []
    activation = []
    back_res = []

    # init function must have next paramters:
    # x => input_x
    # y => input_y
    # lr => learning_rate (default 0.001)
    def __init__(self, x, y, lr=0.001):
        self.x = x
        self.y = y
        self.shape = self.x.shape[1]
        self.lr = lr

    # add dense layer to net
    def add_layer(self, w_num, use_bias=False, activation=None):
        w = np.random.normal(size=(self.shape, w_num))
        self.param_count.append(self.shape * w_num)
        self.shape = w_num
        self.param.append([w])
        self.activation.append(activation)
        if use_bias:
            self.param[-1].append(np.array([1.0] * w_num))
            self.param_count[-1] += w_num

    # forward function
    def forward(self):
        self.forward_res = []
        for x in self.x:
            temp = [x]
            for w in self.param:
                x = np.dot(x, w[0])
                if len(w) == 2:
                    x += w[1]
                temp.append(x)
            self.forward_res.append(temp)

    # 'mse' backforward function
    def mse_back_forward(self, pred, label):
        return -(label - pred)

    # update weights and bias
    def back_forward(self):
        # default using SGD
        idx = np.random.randint(0, self.x.shape[0])

        pred, y, layers = self.forward_res[idx], self.y[idx], len(self.param)
        gradient = [np.array([self.mse_back_forward(i, y) for i in pred[-1]])]

        # Computed network gradient
        for i in range(layers - 1, 0, -1):
            temp = gradient[-1].reshape((-1, 1))
            gradient.append(np.dot(self.param[i][0], temp))
        gradient.reverse()

        # 更新权重
        for i in range(layers - 1, -1, -1):
            for j in range(self.param[i][0].shape[1]):
                if self.activation[i] is not None:
                    pass

            # 更新weights
            for j in range(self.param[i][0].shape[1]):
                for k in range(self.param[i][0].shape[0]):
                    self.param[i][0][k][j] = self.param[i][0][k][j] - self.lr * gradient[i][j] * pred[i][k]

            # 更新bias
            for j in range(self.param[i][1].shape[0]):
                self.param[i][1][j] = self.param[i][1][j] - self.lr * gradient[i][j]

    # calculate loss function
    # default loss function => 'mean square error'
    def cal_loss(self, lossF='mse'):
        if lossF == 'mse':
            return np.mean(0.5 * (self.y - np.array(self.forward_res)[:, -1]) ** 2)

    # using to debug
    def print_param(self):
        for i in self.param:
            print(i)

    # using to debug
    def print_forward(self):
        for i in self.temp:
            print(i)
        print('\n\n')
        for i in self.forward_res:
            print(i)

    # using to debug
    def print_param_count(self):
        print(self.param_count)
        print(sum(self.param_count))
