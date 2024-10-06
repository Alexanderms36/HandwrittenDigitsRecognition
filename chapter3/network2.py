import numpy


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
        
    