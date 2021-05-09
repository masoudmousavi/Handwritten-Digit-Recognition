import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from math import log as _log, exp
from scipy.optimize import minimize
import json


def log(num):
    if(num):
        return _log(num)
    return -25


class NeuralNetwork(object):
    def __init__(self, path, architecture, regularization_param):
        self.path_to_training_data_set = path
        self.architecture = architecture
        self.layers_num = len(self.architecture)
        self.learning_rate = 50
        self._iterations = 0
        self.regularization_param = regularization_param
        self.X, self.y, self.classes = self._extract_input_data(self.path_to_training_data_set)
        self.training_set_size = self.X.shape[0]
        self.weights = self._create_random_weights()
        self.activations = self._initialize_activations()

    @property
    def input_units_num(self):
        return self.architecture[0]


    @property
    def hidden_units_num(self):
        return self.architecture[1]


    @property
    def classes_num(self):
        return self.architecture[-1]


    def _extract_input_data(self, path):
        data_dict = sio.loadmat(path)
        labeled_y = data_dict['y']
        ret_y = np.zeros((labeled_y.shape[0], self.classes_num))
        for i in range(labeled_y.shape[0]):
            ret_y[i][labeled_y[i] % 10] = 1

        return data_dict['X'], ret_y, labeled_y


    def _create_random_weights(self):
        EPSILON = 0.4
        weights_1 = np.random.rand(self.hidden_units_num, self.input_units_num+1) * (2 * EPSILON) - EPSILON
        weights_2 = np.random.rand(self.classes_num, self.hidden_units_num+1) * (2 * EPSILON) - EPSILON
        return [weights_1, weights_2]


    def unroll_matrix(self, weights):
        return np.array(list(np.ravel(weights[0])) + list(np.ravel(weights[1])))


    def matricize(self, vector):

        vector = list(vector)
        _s0 = self.weights[0].shape
        _s1 = self.weights[1].shape
        _w0 = vector[:np.ravel(self.weights[0]).shape[0]]
        _w1 = vector[np.ravel(self.weights[0]).shape[0]:]
        return [np.reshape(_w0, _s0), np.reshape(_w1, _s1)]


    def _initialize_activations(self):
        activations = [
            None,
            None,
            None
        ]
        return activations


    def _activate_input_units(self, training_example):
        self.activations[0] = np.array([1] + list(self.X[training_example]))
        # for i in range(1, self.input_units_num + 1):
        #     self.activations[0][i] = 2


    def forward_prop(self, training_example):
        self._activate_input_units(training_example)
        # self.activations[0][0] = 1
        # self.activations[0][1:] = self.X[training_example]
        for l in range(1, self.layers_num):
            z = np.matmul(self.weights[l-1], self.activations[l-1])
            # if not l == self.layers_num - 1:
            #     _z = list(z)
            #     z = np.array([1.] + _z)
            self._activate(l, z)


    def _activate(self, layer, z):
        # sigmoid = lambda x: 1 / (1 + exp(-x))
        # k = 1
        # else:
        #     k = 0
        # for i in range(k, self.architecture[layer]+k):
        #     # if i == self.classes_num and layer == self.layers_num - 1:
        #     #     return
        #     self.activations[layer][i] = sigmoid(z[i-1])
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        vfunc = np.vectorize(sigmoid)
        if layer < self.layers_num - 1:

            self.activations[layer] = np.array([1] + list(vfunc(z)))

        else:
            self.activations[layer] = np.roll(vfunc(z), 1)


    #might have a proplem in the mult methods
    def compute_errors(self, training_example):
        output_layer_errors = self.activations[-1] - self.y[training_example]
        hidden_layer_errors = np.multiply(
            np.matmul(self.weights[1].T, output_layer_errors),
            np.multiply(self.activations[1], (1 - self.activations[1]))
        )
        _hle = list(hidden_layer_errors)
        _hle.pop(0)
        hidden_layer_errors = np.array(_hle)
        return [hidden_layer_errors, output_layer_errors]


    def cost_function(self, weights):
        _weights = self.matricize(weights)
        sum_term = 0.0
        for t in range(self.training_set_size):
            self.forward_prop(t)
            vfunc = np.vectorize(np.log)
            positive_term = np.dot(
                self.y[t],
                vfunc(self.activations[-1])
            )
            negative_term = np.dot(
                (1 - self.y[t]),
                vfunc(1 - self.activations[-1])
            )
            sum_term += (-positive_term - negative_term)

        regularization_sum = 0.0
        for l in range(2):
            regularization_sum += np.dot(
                self.unroll_matrix(_weights[l]),
                self.unroll_matrix(_weights[l])
            )
        cost = (1 / self.training_set_size) * sum_term + (self.regularization_param / (2*self.training_set_size)) * regularization_sum
        print(f'Iteration {self._iterations + 1}\'s cost: {cost}')
        return cost


    def backprop(self, weights):
        self._iterations += 1
        weights = list(weights)
        _s0 = self.weights[0].shape
        _s1 = self.weights[1].shape
        _w0 = weights[:np.ravel(self.weights[0]).shape[0]]
        _w1 = weights[np.ravel(self.weights[0]).shape[0]:]
        _weights = [np.reshape(_w0, _s0), np.reshape(_w1, _s1)]
        gradients = [np.copy(_weights[0]), np.copy(_weights[1])]
        delta_accumulators = [np.copy(_weights[0]), np.copy(_weights[1])]
        for t in range(self.training_set_size):
            if(t % 500 == 0):
                # print(f"Iteration {self._iterations}, Training example {t+1}")
                pass
            # self.activate_input_units(t)
            self.forward_prop(t)
            errors = self.compute_errors(t)
            # for l in range(2):
            #     delta_accumulators[l] += np.matmul([errors[l]], [self.activations[l]] )

            for layer in range(2):
                delta_accumulators[layer] += np.matmul(
                    errors[layer].reshape(1, errors[layer].shape[0]).T,
                    self.activations[layer].reshape((1, self.activations[layer].shape[0]))
                )

            # for l in range(len(delta_accumulators)):
            #     for i in range(delta_accumulators[l].shape[0]):
            #         for j in range(delta_accumulators[l].shape[1]):
            #             delta_accumulators[l][i][j] += self.activations[l][j] * errors[l][i]
        for layer in range(2):
            gradients[layer] = (1 / self.training_set_size) * delta_accumulators[layer] + (
                    self.regularization_param / self.training_set_size) * _weights[layer]

        # for l in range(len(delta_accumulators)):
        #     for i in range(delta_accumulators[l].shape[0]):
        #         for j in range(delta_accumulators[l].shape[1]):
        #             if j == 0:
        #                 gradients[l][i][j] = (1 / self.training_set_size) * delta_accumulators[l][i][j]
        #             else:
        #                 gradients[l][i][j] = (1 / self.training_set_size) * delta_accumulators[l][i][j] + (self.regularization_param / self.training_set_size) * _weights[l][i][j]

        # return gradients
        return np.array(list(np.ravel(gradients[0])) + list(np.ravel(gradients[1])))


    def gradient_checking(self):

        grad_approx = [np.copy(self.weights[0]), np.copy(self.weights[1])]
        EPSILON = 0.0001
        for l in range(len(grad_approx)):
            for i in range(grad_approx[l].shape[0]):
                for j in range(grad_approx[l].shape[1]):
                    Theta_plus = [np.copy(self.weights[0]), np.copy(self.weights[1])]
                    Theta_minus = [np.copy(self.weights[0]), np.copy(self.weights[1])]
                    Theta_minus[l][i][j] -= EPSILON
                    Theta_plus[l][i][j] += EPSILON
                    x1 = np.array(list(np.ravel([0])) + list(np.ravel([1])))
                    grad_approx[l][i][j] = (self.cost_function(self.unroll_matrix(Theta_plus)) - self.cost_function(self.unroll_matrix(Theta_minus))) / (2 * EPSILON)

        return grad_approx


    def optimize_params(self):
        res = minimize(self.cost_function,
                 x0=np.array(list(np.ravel(self.weights[0])) + list(np.ravel(self.weights[1]))),
                 jac=self.backprop,
                 options={'disp': True}
        )
        with open("weights.txt", "w") as fp:
            json.dump({
                'weights0': list(res.x)
            }, fp)
        return res.x

        # temp_weights = [np.copy(self.weights[0]), np.copy(self.weights[1])]
        # cost1 = self.cost_function(self.weights)
        # for _ in range(4000):
        #     gradients = self.backprop(self.weights, iter=_)
        #     for l in range(len(self.weights)):
        #         for i in range(self.weights[l].shape[0]):
        #             for j in range(self.weights[l].shape[1]):
        #                 self.weights[l][i][j] -= self.learning_rate * (1/self.training_set_size) * gradients[l][i][j]
        #     # self.learning_rate *= 0.9
        #     cost2 = self.cost_function(self.weights)
        #     print(f"Iteration {_+1} completed, {cost1 - cost2} decreased from the cost function, cost function is {cost2}")
        #     cost1 = cost2
        # return self.weights


    def find_class(self, training_example):
        return self.y[training_example]
        # for i in range(self.classes_num):
        #     # if(self.y[training_example][i]):
        #     # #     if i == 0:
        #     # #         return self.classes_num
        #     # #     return i
        #     #     return i


    def predict(self, features):
        with open('weights.txt') as fp:
            weights = json.load(fp)

        weights = list(weights['weights0'])
        _s0 = self.weights[0].shape
        _s1 = self.weights[1].shape
        _w0 = weights[:np.ravel(self.weights[0]).shape[0]]
        _w1 = weights[np.ravel(self.weights[0]).shape[0]:]
        self.weights = [np.reshape(_w0, _s0), np.reshape(_w1, _s1)]
        # self.weights = [np.ones_like(_w0, _s0), np.ones_like(_w1, _s1)]

        self._initialize_activations()
        for i in range(1, self.architecture[0]+1):
            self.activations[0][i] = features[i-1]

        self.forward_prop()
        return self.activations[-1]


    def gradient_decent(self):
        # temp_weights = [np.copy(self.weights[0]), np.copy(self.weights[1])]
        alpha = 500
        constant = 0.5
        for _ in range(1000):
            gradients = self.backprop(self.unroll_matrix(self.weights))
            for layer in range(2):
                self.weights[layer] -= constant * gradients[layer]
                # self.weights[layer] = temp_weights[layer]
            print(
                f'Iteration {self._iterations}\'s cost: {self.cost_function(self.unroll_matrix(self.weights))}')

            print('===================')


ANN = NeuralNetwork('ex4data1.mat', [400, 25, 10], 0.0)





# indx = 2354

dict = sio.loadmat('ex4weights.mat')
ANN.weights = [dict['Theta1'], dict['Theta2']]
# # ANN.activate_input_units(indx)
# # ANN.forward_prop()
print(ANN.cost_function(ANN.unroll_matrix(ANN.weights)))
# print(np.argmax(ANN.activations[-1]))
# plt.imshow(
#     np.reshape(
#         ANN.X[indx],
#         (20, 20)
#     ).T
# )
# plt.show()

# print(w)
# with open("weights.txt", "w") as fp:
#     json.dump({
#         'weights0': list(w)
#     }, fp)

# indx = 4567
# print(ANN.y[indx])
# print(ANN.classes[indx])
# plt.imshow(
#     np.reshape(
#         ANN.X[indx],
#         (20, 20)
#     ).T
# )
# plt.show()

print(ANN.optimize_params())

# print(ANN.optimize_params())
# for i in range(ANN.training_set_size):
#     plt.figure()
#     index = random.randint(0, ANN.training_set_size)
#     plt.imshow(np.reshape(ANN.X[index], (20, 20)).T, cmap='gray')
#     plt.title(ANN.classes[index] % 10)
#     # plt.colorbar()
#     plt.grid(False)
#     plt.show()
#     plt.close()
#     plt.ioff()
#     # time.sleep(5)