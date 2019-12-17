import numpy as np
import matplotlib.pyplot as plt
import time


class FullyConnectLayer:
    # record the model layer nums
    layer_num = 0
    # save the param
    param = {}
    # using to save the model fit information
    history = {'loss': []}
    # save the model forward result
    forward_res = []

    def __init__(self, x, y, loss):
        """
        class init function
        :param x: your train_x data, the shape must => (data_len, input_shape) (np.array)
        :param y: your train_y data, the shape must => (data_len, output_shape) (np.array)
        :param loss: your loss function to this model (string)
        """
        self.x = x
        self.y = y
        self.input_shape = self.x.shape[1]
        self.output_shape = self.y.shape[1]
        self.loss = loss

    def cal_loss(self, pred, y):
        """
        Compute Loss
        :param pred: model predict data (np.array)
        :return: loss value (float)
        """
        if self.loss == 'mse':
            return self.mse(pred, y)
        elif self.loss == 'CrossEntropy':
            return self.cross_entropy(pred, y)

    def mse(self, pred, y, forward=True):
        """
        MeanSquareErrorFunction
        :param pred: model predict data (np.array)
        :param y: the true data (np.array)
        :param forward: judge is forward or backward, exp: forward=True => forward (default:True) (bool)
        :return: forward result or backward result (np.array)
        """
        if forward:
            return np.mean((y - pred) ** 2)
        else:
            return pred - y

    def cross_entropy(self, pred, y, forward=True):
        """
        CrossEntropyFunction
        :param pred: model predict data (np.array)
        :param y: the true data (np.array)
        :param forward: judge is forward or backward, exp: forward=True => forward (default:True) (bool)
        :return: forward result or backward result (np.array)
        """
        if forward:
            return np.mean(-np.sum(y * np.log(np.clip(pred, 1e-15, 1.0)), axis=1))
        else:
            return pred - y

    def add_layer(self, hiddle_num, use_bias=True, activation=None, weight_init=np.random.normal, bias_init=np.ones):
        """
        add layer to the model
        :param hiddle_num: the hiddle num (int>0)
        :param use_bias: when the use_bias is True, the model will use bias to compute (default:True) (bool)
        :param activation: the activation of this layer (default:None(mean tha activation layer is not use)) (instance see Activation)
        :param weight_init: the weight init function (default:np.random.normal) (instance)
        :param bias_init: the bias init function (default:np.ones) (instance)
        """
        # update layer_num
        self.layer_num = self.layer_num + 1

        # init weights
        weights = weight_init(size=(self.input_shape, hiddle_num))
        # update the next layer input_shape
        self.input_shape = hiddle_num

        # update param(weights, bias, activation)
        self.param['w' + str(self.layer_num)] = weights
        self.param['a' + str(self.layer_num)] = activation
        if use_bias:
            self.param['b' + str(self.layer_num)] = bias_init(shape=(hiddle_num,))

    def fit(self, optimizer, epoch, valid_data=None, epoch_shuffle=False, learning_rate=0.001, display_epoch=0,
            callback=[]):
        """
        train model
        :param optimizer: optimizer (string)
        :param epoch: number of times to train the model (int>0)
        :param valid_data: use to valid the model (default:None) (tuple)
        :param epoch_shuffle: if True, it will shuffle data every epoch (default:False) (bool)
        :param learning_rate: learning_rate (default:0.001) (float>0.0)
        :param display_epoch: Training things how many epochs to show (default:0(means not show)) (int>0)
        :param callback: callback list function (default:empty list) (list[string])
        :return: history information (dict)
        """
        begin = time.time()
        self.lr = learning_rate
        for i in range(1, epoch + 1):
            if epoch_shuffle:
                self.shuffle_data()
            if optimizer == 'gd':
                self.gradient_descent()
            elif optimizer == 'sgd':
                self.stochastic_gradient_descent()
            elif optimizer == 'momentum':
                if i == 1:
                    self.pre_gradient = {}
                    for i in range(self.layer_num, 0, -1):
                        self.pre_gradient['dw' + str(i)] = np.zeros_like(self.param['w' + str(i)])
                        if 'b' + str(i) in self.param:
                            self.pre_gradient['db' + str(i)] = np.zeros_like(self.param['b' + str(i)])
                self.stochastic_gradient_descent_momentum()
            elif optimizer == 'adam':
                if i == 1:
                    self.pre_m = {}
                    self.pre_v = {}
                    for i in range(self.layer_num, 0, -1):
                        self.pre_m['dw' + str(i)] = np.zeros_like(self.param['w' + str(i)])
                        self.pre_v['dw' + str(i)] = np.zeros_like(self.param['w' + str(i)])
                        if 'b' + str(i) in self.param:
                            self.pre_m['db' + str(i)] = np.zeros_like(self.param['b' + str(i)])
                            self.pre_v['db' + str(i)] = np.zeros_like(self.param['b' + str(i)])
                self.adam(i)

            self.history['loss'].append(self.cal_loss(self.forward_res, self.y))

            if 'SaveBest' in callback:
                self.save_best()

            if i % display_epoch == 0 and display_epoch != 0:
                print('epoch:{}, loss:{}, time:{}'.format(i, self.history['loss'][-1], time.time() - begin), end='')
                if 'accuracy' in callback:
                    print(', test_accuracy:{}'.format(self.accuracy(valid_data[0], valid_data[1])), end='')
                print()
                begin = time.time()

        return self.history

    def gradient_descent(self):
        """
        gradient descent optimizer
        """
        self.forward_res = []
        pred = [self.x]
        temp = self.x
        for j in range(1, self.layer_num + 1):
            temp = np.dot(temp, self.param['w' + str(j)])
            if 'b' + str(j) in self.param:
                temp = temp + np.tile(self.param['b' + str(j)], (temp.shape[0], 1))
            if self.param['a' + str(j)] is not None:
                temp = self.param['a' + str(j)].calculate_activation(temp)
            pred.append(temp)
        self.forward_res = pred[-1]

        gradient = {}

        for j in range(self.layer_num, 0, -1):
            if j == self.layer_num:
                if self.loss == 'mse':
                    gradient_temp = self.mse(pred[-1], self.y, forward=False)
                    if self.param['a' + str(j)] is not None:
                        gradient_temp = gradient_temp * self.param['a' + str(j)].calculate_activation(pred[j],
                                                                                                      forward=False)
                elif self.loss == 'CrossEntropy':
                    gradient_temp = self.cross_entropy(pred[-1], self.y, forward=False)
            else:
                gradient_temp = np.dot(gradient_temp, self.param['w' + str(j + 1)].T)
                if self.param['a' + str(j)] is not None:
                    gradient_temp = gradient_temp * self.param['a' + str(j)].calculate_activation(
                        pred[j],
                        forward=False)

            gradient['dw' + str(j)] = self.lr * np.dot(pred[j - 1].T, gradient_temp) / self.x.shape[0]
            if 'b' + str(j) in self.param:
                gradient['db' + str(j)] = self.lr * np.mean(gradient_temp, axis=0)

        for j in range(self.layer_num, 0, -1):
            self.param['w' + str(j)] = self.param['w' + str(j)] - gradient['dw' + str(j)]
            if 'b' + str(j) in self.param:
                self.param['b' + str(j)] = self.param['b' + str(j)] - gradient['db' + str(j)]

    def stochastic_gradient_descent(self):
        """
        stochastic gradient descent
        """
        self.forward_res = []
        for i in range(self.x.shape[0]):
            temp = self.x[i].reshape(1, -1)
            pred = [temp]
            y = self.y[i].reshape((1, -1))
            for j in range(1, self.layer_num + 1):
                temp = np.dot(temp, self.param['w' + str(j)])
                if 'b' + str(j) in self.param:
                    temp = temp + np.tile(self.param['b' + str(j)], (temp.shape[0], 1))
                if self.param['a' + str(j)] is not None:
                    temp = self.param['a' + str(j)].calculate_activation(temp)
                pred.append(temp)
            self.forward_res.append(pred[-1].reshape((-1)))

            gradient = {}

            for j in range(self.layer_num, 0, -1):
                if j == self.layer_num:
                    if self.loss == 'mse':
                        gradient_temp = self.mse(pred[-1], y, forward=False)
                        if self.param['a' + str(j)] is not None:
                            gradient_temp = gradient_temp * self.param['a' + str(j)].calculate_activation(
                                pred[j],
                                forward=False)
                    elif self.loss == 'CrossEntropy':
                        gradient_temp = self.cross_entropy(pred[-1], y, forward=False)
                else:
                    gradient_temp = np.dot(gradient_temp, self.param['w' + str(j + 1)].T)
                    if self.param['a' + str(j)] is not None:
                        gradient_temp = gradient_temp * self.param['a' + str(j)].calculate_activation(
                            pred[j],
                            forward=False)

                gradient['dw' + str(j)] = self.lr * np.dot(pred[j - 1].T, gradient_temp)
                if 'b' + str(j) in self.param:
                    gradient['db' + str(j)] = self.lr * gradient_temp

            for j in range(self.layer_num, 0, -1):
                self.param['w' + str(j)] = self.param['w' + str(j)] - gradient['dw' + str(j)]
                if 'b' + str(j) in self.param:
                    self.param['b' + str(j)] = self.param['b' + str(j)] - gradient['db' + str(j)]

    def stochastic_gradient_descent_momentum(self, momentum=0.9):
        """
        stochastic gradient descent with momentum
        :param momentum: parameter that accelerates SGD in the relevant direction and dampens oscillations (default:0.9) (float 0.0<momentum<=1.0)
        """
        self.forward_res = []
        for i in range(self.x.shape[0]):
            temp = self.x[i].reshape(1, -1)
            pred = [temp]
            y = self.y[i].reshape((1, -1))
            for j in range(1, self.layer_num + 1):
                temp = np.dot(temp, self.param['w' + str(j)])
                if 'b' + str(j) in self.param:
                    temp = temp + np.tile(self.param['b' + str(j)], (temp.shape[0], 1))
                if self.param['a' + str(j)] is not None:
                    temp = self.param['a' + str(j)].calculate_activation(temp)
                pred.append(temp)
            self.forward_res.append(pred[-1].reshape((-1)))

            gradient = {}

            for j in range(self.layer_num, 0, -1):
                if j == self.layer_num:
                    if self.loss == 'mse':
                        gradient_temp = self.mse(pred[-1], y, forward=False)
                        if self.param['a' + str(j)] is not None:
                            gradient_temp = gradient_temp * self.param['a' + str(j)].calculate_activation(
                                pred[j],
                                forward=False)
                    elif self.loss == 'CrossEntropy':
                        gradient_temp = self.cross_entropy(pred[-1], y, forward=False)
                else:
                    gradient_temp = np.dot(gradient_temp, self.param['w' + str(j + 1)].T)
                    if self.param['a' + str(j)] is not None:
                        gradient_temp = gradient_temp * self.param['a' + str(j)].calculate_activation(
                            pred[j],
                            forward=False)

                gradient['dw' + str(j)] = momentum * self.pre_gradient['dw' + str(j)] + self.lr * np.dot(pred[j - 1].T,
                                                                                                         gradient_temp)
                if 'b' + str(j) in self.param:
                    gradient['db' + str(j)] = momentum * self.pre_gradient['db' + str(j)] + self.lr * gradient_temp

            for j in range(self.layer_num, 0, -1):
                self.param['w' + str(j)] = self.param['w' + str(j)] - gradient['dw' + str(j)]
                if 'b' + str(j) in self.param:
                    self.param['b' + str(j)] = self.param['b' + str(j)] - gradient['db' + str(j)]

            self.pre_gradient = gradient.copy()

    def adam(self, epoch, b1=0.9, b2=0.999, e=0.00000001):
        """
        adam optimizer
        :param epoch: epoch (int)
        :param b1: b1 param (default:0.9) (float 0.0<=b1<=1.0)
        :param b2: b2 param (default:0.999) (float 0.0<=b2<=1.0)
        :param e: e param (default:0.00000001) (float)
        """
        self.forward_res = []
        for i in range(self.x.shape[0]):
            temp = self.x[i].reshape(1, -1)
            pred = [temp]
            y = self.y[i].reshape((1, -1))
            for j in range(1, self.layer_num + 1):
                temp = np.dot(temp, self.param['w' + str(j)])
                if 'b' + str(j) in self.param:
                    temp = temp + np.tile(self.param['b' + str(j)], (temp.shape[0], 1))
                if self.param['a' + str(j)] is not None:
                    temp = self.param['a' + str(j)].calculate_activation(temp)
                pred.append(temp)
            self.forward_res.append(pred[-1].reshape((-1)))

            gradient = {}

            for j in range(self.layer_num, 0, -1):
                if j == self.layer_num:
                    if self.loss == 'mse':
                        gradient_temp = self.mse(pred[-1], y, forward=False)
                        if self.param['a' + str(j)] is not None:
                            gradient_temp = gradient_temp * self.param['a' + str(j)].calculate_activation(
                                pred[j],
                                forward=False)
                    elif self.loss == 'CrossEntropy':
                        gradient_temp = self.cross_entropy(pred[-1], y, forward=False)
                else:
                    gradient_temp = np.dot(gradient_temp, self.param['w' + str(j + 1)].T)
                    if self.param['a' + str(j)] is not None:
                        gradient_temp = gradient_temp * self.param['a' + str(j)].calculate_activation(
                            pred[j],
                            forward=False)

                g = np.dot(pred[j - 1].T, gradient_temp)
                m = b1 * self.pre_m['dw' + str(j)] + (1 - b1) * g
                v = b2 * self.pre_v['dw' + str(j)] + (1 - b2) * (g ** 2)
                mt = m / (1 - b1 ** epoch)
                vt = v / (1 - b2 ** epoch)
                self.pre_m['dw' + str(j)] = m
                self.pre_v['dw' + str(j)] = v
                gradient['dw' + str(j)] = self.lr * mt / (np.sqrt(vt) + e)

                if 'b' + str(j) in self.param:
                    m = b1 * self.pre_m['db' + str(j)] + (1 - b1) * gradient_temp
                    v = b2 * self.pre_v['db' + str(j)] + (1 - b2) * (gradient_temp ** 2)
                    mt = m / (1 - b1 ** epoch)
                    vt = v / (1 - b2 ** epoch)
                    self.pre_m['db' + str(j)] = m
                    self.pre_v['db' + str(j)] = v
                    gradient['db' + str(j)] = self.lr * mt / (np.sqrt(vt) + e)

            for j in range(self.layer_num, 0, -1):
                self.param['w' + str(j)] = self.param['w' + str(j)] - gradient['dw' + str(j)]
                if 'b' + str(j) in self.param:
                    self.param['b' + str(j)] = self.param['b' + str(j)] - gradient['db' + str(j)]

    def predict(self, x):
        """
        predict function
        :param x: input_data (np.array)
        :return: output_data (np.array)
        """
        temp = x
        for j in range(1, self.layer_num + 1):
            temp = np.dot(temp, self.param['w' + str(j)])
            if 'b' + str(j) in self.param:
                temp = temp + np.tile(self.param['b' + str(j)], (temp.shape[0], 1))
            if self.param['a' + str(j)] is not None:
                temp = self.param['a' + str(j)].calculate_activation(temp)
        return temp

    def accuracy(self, x, y):
        """
        when the loss function is cross entropy, accuracy can be a metrics
        :param x: data like self.x
        :param y: data like self.y
        :return: accuracy metrics
        """
        pred = self.predict(x)
        return np.sum(np.equal(np.argmax(pred, axis=1), np.argmax(y, axis=1))) / x.shape[0]

    def save_best(self):
        """
        Save_Best function
        """
        if 'best_loss' not in self.history:
            self.history['best_loss'] = 0x7fffffff
            self.history['best_param'] = self.param
        if self.history['best_loss'] > self.history['loss'][-1]:
            self.history['best_loss'] = self.history['loss'][-1]
            self.history['best_param'] = self.param

    def plt_loss(self):
        """
        plot the loss image
        """
        plt.figure()
        plt.plot(range(1, len(self.history['loss']) + 1), self.history['loss'])
        plt.title('loss')
        plt.show()

    def shuffle_data(self):
        """
        shuffle self.x and self.y
        """
        idx = np.arange(self.x.shape[0])
        np.random.shuffle(idx)
        self.x = self.x[idx]
        self.y = self.y[idx]
