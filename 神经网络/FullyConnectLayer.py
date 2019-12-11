import numpy as np
import matplotlib.pyplot as plt
import time, sys, math


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

    def cal_loss(self, pred):
        """
        Compute Loss
        :param pred: model predict data (np.array)
        :return: loss value (float)
        """
        if self.loss == 'mse':
            return self.mse(pred)

    def mse(self, pred, forward=True):
        """
        MeanSquareErrorFunction
        :param pred: model predict data (np.array)
        :param forward: judge is forward or backward, exp: forward=True => forward (default:True) (bool)
        :return: forward result or backward result (np.array)
        """
        if forward:
            return np.mean((self.y - pred) ** 2)
        else:
            return pred - self.y

    # unfinish
    def cross_entropy(self, pred, forward=True):
        if forward:
            return -(np.sum(self.y * np.log(pred)))
        else:
            return -(self.y / pred)

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

    def fit(self, optimizer, epoch, learning_rate=0.001, display_epoch=0, callback=[]):
        """
        train model
        :param optimizer: optimizer (string)
        :param epoch: number of times to train the model (int>0)
        :param learning_rate: learning_rate (default:0.001) (float>0.0)
        :param display_epoch: Training things how many epochs to show (default:0(means not show)) (int>0)
        :param callback: callback list function (default:empty list) (list[string])
        :return: history information (dict)
        """
        begin = time.time()
        for i in range(1, epoch + 1):
            self.forward_res = [self.x]
            temp = self.x
            for j in range(1, self.layer_num + 1):
                temp = np.dot(temp, self.param['w' + str(j)])
                if 'b' + str(j) in self.param:
                    temp = temp + np.tile(self.param['b' + str(j)], (temp.shape[0], 1))
                if self.param['a' + str(j)] is not None:
                    temp = self.param['a' + str(j)].calculate_activation(temp)
                self.forward_res.append(temp)

            gradient = {}

            for j in range(self.layer_num, 0, -1):
                if j == self.layer_num:
                    gradient_temp = self.mse(self.forward_res[-1], forward=False)
                    if self.param['a' + str(j)] is not None:
                        gradient_temp = gradient_temp * self.param['a' + str(j)].calculate_activation(
                            self.forward_res[j],
                            forward=False)
                else:
                    gradient_temp = np.dot(gradient_temp, self.param['w' + str(j + 1)].T)
                    if self.param['a' + str(j)] is not None:
                        gradient_temp = gradient_temp * self.param['a' + str(j)].calculate_activation(
                            self.forward_res[j],
                            forward=False)

                gradient['dw' + str(j)] = np.dot(self.forward_res[j - 1].T, gradient_temp) / self.x.shape[0]
                if 'b' + str(j) in self.param:
                    gradient['db' + str(j)] = np.mean(gradient_temp, axis=0)

            for j in range(self.layer_num, 0, -1):
                self.param['w' + str(j)] = self.param['w' + str(j)] - gradient['dw' + str(j)] * learning_rate
                if 'b' + str(j) in self.param:
                    self.param['b' + str(j)] = self.param['b' + str(j)] - gradient['db' + str(j)] * learning_rate

            self.history['loss'].append(self.cal_loss(self.forward_res[-1]))

            if 'SaveBest' in callback:
                self.save_best()

            if i % display_epoch == 0 and display_epoch != 0:
                print('epoch:{}, loss:{}, time:{}'.format(i, self.history['loss'][-1], time.time() - begin))
                begin = time.time()

        return self.history

    def predict(self, x):
        """
        predict funtion
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
