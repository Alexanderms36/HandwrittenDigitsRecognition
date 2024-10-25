import os
os.environ['AESARA_FLAGS'] = 'device=cpu,floatX=float64'
import aesara
aesara.config.cxx = ''
import numpy
import gzip
import pickle
import aesara.tensor.nnet as nnet
import aesara.tensor.signal as signal
# from aesara.tensor.signal import downsample
from aesara.tensor.signal import pool
from .dropout_layer import drp_lr
from aesara import printing
# from theano.tensor.nnet import softmax
import aesara.tensor as at

def linear(z):
    return z

def ReLU(z):
    return aesara.tensor.maximum(0.0, z)

def sigmoid(z):
    return 1 / (1 + aesara.tensor.exp(-z))

def load_data_shared(filename):
    f = gzip.open(filename, 'rb')
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    training_data, validation_data, test_data = u.load()
    f.close()
    def shared(data):
        shared_x = aesara.shared(
            numpy.asarray(data[0], dtype=aesara.config.floatX), borrow=True
        )
        shared_y = aesara.shared(
            numpy.asarray(data[1], dtype=aesara.config.floatX), borrow=True
        )
        return shared_x, aesara.tensor.cast(shared_y, "int32")
    return [shared(training_data), shared(validation_data), shared(test_data)]

print(f'aesara version: {aesara.__version__}')

class Network(object):
    def __init__(self, layers, mini_batch_size):
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = aesara.tensor.matrix("x")
        self.y = aesara.tensor.ivector("y")
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        for i in range(1, len(self.layers)):
            prev_layer, layer = self.layers[i-1], self.layers[i]
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout, self.mini_batch_size
            )
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, lmbda=0.0):
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data
        print("Check training data:", numpy.any(numpy.isnan(training_x.get_value())), 
              numpy.any(numpy.isinf(training_x.get_value())))
        print("Check validation data:", numpy.any(numpy.isnan(validation_x.get_value())), 
              numpy.any(numpy.isinf(validation_x.get_value())))

        print(f'size(training_data): {size(training_data)}')
        print(f'mini_batch_size: {mini_batch_size}')

        num_training_batches = int(size(training_data) / mini_batch_size)
        print(f'num_training_batches: {num_training_batches}')
        num_validation_batches = int(size(validation_data) / mini_batch_size) # = 0
        num_test_batches = int(size(test_data) / mini_batch_size) # = 0

        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])

        cost = self.layers[-1].cost(self) + \
              0.5 * lmbda * l2_norm_squared / num_training_batches
        #print(cost)
        grads = aesara.tensor.grad(cost, self.params)
        updates = [(param, (param - eta * grad).astype(aesara.config.floatX)) 
                   for param, grad in zip(self.params, grads)]


        i = aesara.tensor.lscalar()
        train_mb = aesara.function(
            [i],
            cost,
            updates=updates,
            givens= {
                self.x:
                training_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size],
                self.y:
                training_y[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            }
        )

        validate_mb_accuracy = aesara.function(
            [i],
            self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                validation_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size],
                self.y:
                validation_y[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            }
        )

        test_mb_accuracy = aesara.function(
            [i],
            self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                test_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size],
                self.y:
                test_y[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            }
        )

        self.test_mb_predictions = aesara.function(
            [i],
            self.layers[-1].y_out,
            givens={
                self.x:
                test_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            }
        )

        best_validation_accuracy = 0.0
        best_iteration = 0
        test_accuracy = 0.0
        assert not numpy.any(numpy.isnan(training_x.get_value())), "Training data contains NaN values!"
        assert not numpy.any(numpy.isinf(training_x.get_value())), "Training data contains infinite values!"   
        for epoch in range(epochs):
            print(f'epoch: {epoch}')
            for minibatch_index in range(num_training_batches):
                iteration = num_training_batches * epoch + minibatch_index
                if (iteration % 1000 == 0):
                    print(f"Training mini-batch number {iteration}")
                cost_ij = train_mb(minibatch_index)
                if (iteration + 1) % num_training_batches == 0:
                    validation_accuracy = numpy.mean(
                        [validate_mb_accuracy(j) for j in range(num_validation_batches)]
                    )
                    print(f"Epoch {epoch}: validation accuracy {validation_accuracy}")
                    if validation_accuracy >= best_validation_accuracy:
                        print("This is the best validation accuracy to date")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        if (test_data):
                            test_accuracy = numpy.mean(
                                [test_mb_accuracy(j) for j in range(num_test_batches)]
                            )
                            print(f"The corresponding test accuracy is {test_accuracy}")
        
        print("Finished training network")
        print(f"Best validation accuracy of {best_validation_accuracy} obtained at iteration {best_iteration}")
        print(f"Corresponding test accuracy of {test_accuracy}")


class ConvPoolLayer(object):
    def __init__(self, filter_shape, image_shape, poolsize=(2, 2), activation_fn=sigmoid):
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn = activation_fn

        n_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) / numpy.prod(poolsize))
        self.w = aesara.shared(
            numpy.asarray(
                numpy.random.normal(loc=0, scale=numpy.sqrt(1.0/n_out), size=filter_shape),
                dtype=aesara.config.floatX
            ),
            borrow=True
        )
        self.b = aesara.shared(
            numpy.asarray(
                numpy.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                dtype=aesara.config.floatX
            ),
            borrow=True
        )
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)

        conv_out = nnet.conv.conv2d(
            input = self.inpt,
            filters=self.w,
            filter_shape=self.filter_shape,
            image_shape=self.image_shape
        )

        pooled_out = pool.pool_2d(
            input=conv_out,
            ds=self.poolsize,
            ignore_border=True
        )

        self.output = self.activation_fn(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        )

        self.output_dropout = self.output


class FullyConnectedLayer(object):
    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout

        self.w = aesara.shared(
            numpy.asarray(
                numpy.random.normal(
                    loc=0.0, scale=numpy.sqrt(1.0/n_in), size=(n_in, n_out)
                ),
                dtype=aesara.config.floatX
            ),
            name='w',
            borrow=True
        )
        self.b = aesara.shared(
            numpy.asarray(
                numpy.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                dtype=aesara.config.floatX
            ),
            name='b',
            borrow=True
        )
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(
            (1 - self.p_dropout) * aesara.tensor.dot(self.inpt, self.w) + self.b
        )
        self.y_out = aesara.tensor.argmax(self.output, axis=1)

        #
        # srng = RandomStream(seed=42)
        # reshaped_input = inpt_dropout.reshape((mini_batch_size, self.n_in))
        # dropout_mask = srng.binomial(n=1, size=reshaped_input.shape, p=1 - self.p_dropout)
        # self.inpt_dropout = reshaped_input * dropout_mask / (1 - self.p_dropout)

        self.inpt_dropout = drp_lr(inpt_dropout, mini_batch_size, self.p_dropout, self.n_in)

        # self.inpt_dropout = aesara.tensor.nnet.dropout_layer(
        #     inpt_dropout.reshape((mini_batch_size, self.n_in)),
        #     self.p_dropout
        # )

        self.output_dropout = self.activation_fn(
            aesara.tensor.dot(self.inpt_dropout, self.w) + self.b
        )
    
    def accuracy(self, y):
        return aesara.tensor.mean(aesara.tensor.eq(y, self.y_out))
    

class SoftmaxLayer(object):
    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout

        self.w = aesara.shared(
            numpy.zeros((n_in, n_out), dtype=aesara.config.floatX),
            name='w',
            borrow=True
        )
        self.b = aesara.shared(
            numpy.zeros((n_out,), dtype=aesara.config.floatX),
            name='b',
            borrow=True
        )
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = at.nnet.softmax(
            (1 - self.p_dropout) * aesara.tensor.dot(self.inpt, self.w) + self.b,
            axis=-1
        )
        self.y_out = aesara.tensor.argmax(self.output, axis=1)

        self.inpt_dropout = drp_lr(inpt_dropout, mini_batch_size, self.p_dropout, self.n_in)

        # self.inpt_dropout = aesara.tensor.nnet.dropout_layer(
        #     inpt_dropout.reshape((mini_batch_size, self.n_in)),
        #     self.p_dropout
        # )
        self.output_dropout = aesara.tensor.nnet.softmax(
            aesara.tensor.dot(self.inpt_dropout, self.w) + self.b,
            axis=-1
        )

    def cost(self, net):
        return -aesara.tensor.mean(
            aesara.tensor.log(self.output_dropout)[aesara.tensor.arange(net.y.shape[0]),
                                                    net.y]
        )
    
    def accuracy(self, y):
        return aesara.tensor.mean(aesara.tensor.eq(y, self.y_out))
    

def size(data):
    return data[0].get_value(borrow=True).shape[0]

def dropout_layer(layer, p_dropout):
    srng = aesara.tensor.shared_randomstreams.RandomStreams(
        numpy.random.RandomState(0).randint(999999)
    )
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer * aesara.tensor.cast(mask, aesara.config.floatX)
