import numpy as np
import scipy.io as sio


class StandardNeuralNetwoek(object):
    def __init__(self, path, architecture, regularization_param):
        self.path_to_training_data_set = path
        self.architecture = architecture
        self.layers_num = len(self.architecture)
        self.learning_rate = 50
        self._iterations = 0
        self.regularization_param = regularization_param
        self.X, self.Y, self.classes = self._extract_input_data(self.path_to_training_data_set)
        self.training_set_size = self.X.shape[1]
        self.weights, self.bias = self._create_random_weights()
        # self.bias = self._create_random_bias()
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
        labeled_Y = data_dict['y']
        ret_Y = np.zeros((labeled_Y.shape[0], self.classes_num))
        for i in range(labeled_Y.shape[0]):
            ret_Y[i][labeled_Y[i] % 10] = 1
        X = data_dict['X'].T
        self.mean = np.mean(X)
        self.sigma = np.std(X)
        X = (X - self.mean) / self.sigma
        # np.random.shuffle(X)
        return X, ret_Y.T, labeled_Y


    def _create_random_weights(self):
        np.random.seed(1232)
        EPSILON = 0
        weights_1 = np.random.rand(self.hidden_units_num, self.input_units_num) * np.sqrt(1 / self.architecture[0])
        weights_2 = np.random.rand(self.classes_num, self.hidden_units_num) * np.sqrt(1 / self.architecture[2] )
        return [weights_1, weights_2], [
            np.random.rand(self.hidden_units_num, 1) * EPSILON,
            np.random.rand(self.classes_num, 1) * EPSILON
        ]


    def _initialize_activations(self):
        activations = [
            np.ones((self.architecture[0], self.training_set_size)),
            np.ones((self.architecture[1], self.training_set_size)),
            np.ones((self.architecture[2], self.training_set_size)),
        ]
        return activations


    def forward_propagation(self):
        Z1 = np.matmul(self.weights[0], self.X) + self.bias[0]
        self.activations[1] = np.tanh(Z1)
        Z2 = np.matmul(self.weights[1], self.activations[1]) + self.bias[1]
        sigmoid = np.vectorize(lambda z: 1 / (1 + np.exp(-z)))
        self.activations[2] = sigmoid(Z2)


    def backpropagation(self):
        dZ2 = self.activations[2] - self.Y
        dW2 = np.matmul(dZ2, self.activations[1].T) / self.training_set_size
        db2 = np.sum(dZ2, axis=1, keepdims=True) / self.training_set_size
        Z1 = np.matmul(self.weights[0], self.X) + self.bias[0]
        tanh_prime = np.vectorize(lambda z: 1 - np.power(np.tanh(z), 2))
        dZ1 = np.matmul(self.weights[1].T, dZ2) * tanh_prime(Z1)
        dW1 = np.matmul(dZ1, self.X.T) / self.training_set_size
        db1 = np.sum(dZ1, axis=1, keepdims=True)
        return db1, db2, dW1, dW2


    def cost_function(self):
        o_cost = np.sum(np.concatenate(
            self.Y * np.log(self.activations[-1]) +
            (1 - self.Y) * np.log(1 - self.activations[-1])
        ))
        cost = -o_cost / self.training_set_size
        if self._iterations % 10 == 0:
            print(f'Epoch {self._iterations + 1}, Cost {cost}')
            # self.eval_on_test_set()
            # self.eval_on_training_set()
            print('================================')
        return -o_cost / self.training_set_size


    def gradient_descent(self):
        alpha = 0.1
        for epoch in range(1, 3100):
            self.forward_propagation()
            db1, db2, dW1, dW2 = self.backpropagation()
            self.weights[0] -= alpha * dW1
            self.weights[1] -= alpha * dW2
            self.bias[0] -= alpha * db1
            self.bias[1] -= alpha * db2
            print(f'Epoch: {epoch}, Cost: {self.cost_function()}')


    def adam(self):
        VdW = [np.ones(self.weights[layer].shape) for layer in range(2)]
        SdW = [np.ones(self.weights[layer].shape) for layer in range(2)]
        Vdb = [np.ones(self.bias[layer].shape) for layer in range(2)]
        Sdb = [np.ones(self.bias[layer].shape) for layer in range(2)]
        alpha = 0.005
        beta1 = 0.9
        beta2 = 0.99
        epsilon = 0.00000001
        for epoch in range(1, 3000):
            self._iterations += 1
            self.forward_propagation()
            db1, db2, dW1, dW2 = self.backpropagation()
            dW = [dW1, dW2]
            db = [db1, db2]
            for layer in range(2):
                VdW[layer] = VdW[layer] * beta1 + dW[layer] * (1 - beta1)
                Vdb[layer] = Vdb[layer] * beta1 + db[layer] * (1 - beta1)
                SdW[layer] = SdW[layer] * beta2 + (dW[layer] ** 2) * (1 - beta2)
                Sdb[layer] = Sdb[layer] * beta2 + (db[layer] ** 2) * (1 - beta2)
                # VdW[layer] = VdW[layer] / (1 - beta1 ** epoch)
                # Vdb[layer] = Vdb[layer] / (1 - beta1 ** epoch)
                # SdW[layer] = SdW[layer] / (1 - beta2 ** epoch)
                # Sdb[layer] = Sdb[layer] / (1 - beta2 ** epoch)

                self.weights[layer] -= (VdW[layer] / (SdW[layer] ** 0.5 + epsilon)) * alpha
                self.bias[layer] -= (Vdb[layer] / (Sdb[layer] ** 0.5 + epsilon)) * alpha

            cost = self.cost_function()

            if epoch % 100 == 0:
                print(f'Epoch {epoch + 1}, Cost {cost}')
                # self.eval_on_test_set()
                # self.eval_on_training_set()
                print('================================')


SNN = StandardNeuralNetwoek('ex4data1.mat', [400, 25, 10], 0.)
# SNN.gradient_descent()
SNN.adam()