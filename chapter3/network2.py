import numpy
import random
import json
import sys


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        return numpy.sum(numpy.nan_to_num(
            -y * numpy.log(a) - (1 - y) * numpy.log(1 - a)
            ))
    
    @staticmethod
    def delta(z, a, y):
        return (a - y)
    
class QuadraticCost(object):
    @staticmethod
    def fn(a, y):
        return 0.5 * numpy.linalg.norm(a - y) ** 2
    
    @staticmethod
    def delta(z, a, y):
        return (a - y) * sigmoid_prime(z)

class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self):
        self.biases = [numpy.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [numpy.random.randn(y, x) / numpy.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        
    def large_weight_initializer(self):
        self.biases = [numpy.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [numpy.random.randn(y, x) 
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(numpy.dot(w, a) + b)

        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda=0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):

        if (evaluation_data):
            n_data = len(evaluation_data)
        
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))
            
            print(f'Epoch {j} training complete')
            
            if (monitor_training_cost):
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print(f'Cost on training data: {cost}')
            
            if (monitor_training_accuracy):
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print(f'Accuracy on training data: {accuracy} / {n}')
            
            if (monitor_evaluation_cost):
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost);
                print(f'Cost on evaluation: {cost}')

            if (monitor_evaluation_accuracy):
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print(f'Accuracy on evaluation data: {self.accuracy(evaluation_data)}, {n_data}')            
            
            print()

        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        self.weights = [(1 - eta * (lmbda / n)) * w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                        for b, nb in zip(self.biases, nabla_b)]
        
    def backprop(self, x, y):
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]

        #feedforward
        activation = x
        activations = [x]
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = numpy.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        #backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = numpy.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = numpy.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = numpy.dot(delta, activations[-l - 1].transpose())

        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        if (convert):
            results = [(numpy.argmax(self.feedforward(x)), numpy.argmax(y))
                        for (x, y) in data]
        else:
            results = [(numpy.argmax(self.feedforward(x)), y)
                        for (x, y) in data]

        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        cost = 0.0

        for x, y in data:
            a = self.feedforward(x)
            if (convert):
                y = vectorized_result(y)
            cost += self.cost.fn(a, y) / len(data)
        
        cost += 0.5 * (lmbda / len(data)) * sum(
            numpy.linalg.norm(w)**2 for w in self.weights
        )
        return cost

    def save(self, filename):
        data = {
            "sizes": self.sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "cost": str(self.cost.__name__)
        }
        f = open(filename, 'w')
        json.dump(data, f)
        f.close()

def load(filename):
    f = open(filename, 'r')
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data['cost'])
    net = Network(data["sizes"], cost=cost)
    net.weights = [numpy.array(w) for w in data['weights']]
    net.biases = [numpy.array(b) for b in data['biases']]
    return net

def vectorized_result(j):
    e = numpy.zeros((10, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    # z = numpy.clip(z, -500, 500)
    return 1.0 / (1.0 + numpy.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))
    