import theano
import numpy
import gzip
import pickle

import theano.tensor
import theano.tensor.shared_randomstreams


def linear(z):
    return z

def ReLU(z):
    return theano.tensor.maximum(0.0, z)

def load_data_shared(filename="../data/mnist.pkl.gz"):
    f = gzip.open(filename, 'rb')
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    training_data, validation_data, test_data = u.load()
    f.close()
    def shared(data):
        shared_x = theano.shared(
            numpy.asarray(data[0], dtype=theano.config.floatX), borrow=True
        )
        shared_y = theano.shared(
            numpy.asarray(data[0], dtype=theano.config.floatX), borrow=True
        )
        return shared_x, theano.tensor.cast(shared_y, "int32")
    return [shared(training_data), shared[validation_data], shared(test_data)]


class Network(object):
    def __init__(self, layers, mini_batch_size):
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = theano.tensor.matrix("x")
        self.y = theano.tensor.ivector("y")
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

        num_training_batches = numpy.size(training_data) / mini_batch_size
        num_validation_batches = numpy.size(validation_data) / mini_batch_size
        num_test_batches = numpy.size(test_data) / mini_batch_size

        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self) + \
              0.5 * lmbda * l2_norm_squared / num_training_batches
        grads = theano.tensor.grad(cost, self.params)
        updates = [(param, param - eta * grad)
                   for param, grad in zip(self.params, grads)]
        
        i = theano.tensor.lscalar()
        train_mb = theano.function(
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

        validate_mb_accuracy = theano.function(
            [i],
            self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                validation_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size],
                self.y:
                validation_y[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            }
        )

        test_mb_accuracy = theano.function(
            [i],
            self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                test_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size],
                self.y:
                test_y[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            }
        )

        self.test_mb_predictions = theano.function(
            [i],
            self.layers[-1].y_out(),
            givens={
                self.x:
                test_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            }
        )

        best_validation_accuracy = 0.0
        for epoch in range(epochs):
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
        self.w = theano.shared(
            numpy.asarray(
                numpy.random.normal(loc=0, scale=numpy.sqrt(1.0/n_out), size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        self.b = theano.shared(
            numpy.asarray(
                numpy.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)

        conv_out = theano.tensor.nnet.conv.conv2d(
            input = self.inpt,
            filters=self.w,
            filter_shape=self.filter_shape,
            image_shape=self.image_shape
        )

        pooled_out = theano.tensor.signal.downsample.max_pool_2d(
            input=conv_out,
            ds=self.poolsize,
            ignore_border=True
        )

        self.output = self.activation_fn(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        )

        self.output_dropout = self.output


class FullyConnectedLayer(object):
    def __init__(self, n_in, n_out, activation_fn=theano.tensor.nnet.sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout

        self.w = theano.shared(
            numpy.asarray(
                numpy.random.normal(
                    loc=0.0, scale=numpy.sqrt(1.0/n_out), size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            ),
            name='w',
            borrow=True
        )
        self.b = theano.shared(
            numpy.asarray(
                numpy.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(
            (1 - self.p_dropout) * theano.tensor.dot(self.inpt, self.w) + self.b
        )
        self.y_out = theano.tensor.argmax(self.output, axis=1)
        self.inpt_dropout = theano.tensor.nnet.dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)),
            self.p_dropout
        )
        self.output_dropout = self.activation_fn(
            theano.tensor.dot(self.inpt_dropout, self.w) + self.b
        )
    
    def accuracy(self, y):
        return theano.tensor.mean(theano.tensor.eq(y, self.y_out))
    

class SoftmaxLayer(object):
    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout

        self.w = theano.shared(
            numpy.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='w',
            borrow=True
        )
        self.b = theano.shared(
            numpy.zeros((n_out,), dtype=theano.config.floatX),
            name='b',
            borrow=True
        )
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = theano.tensor.nnet.softmax(
            (1 - self.p_dropout) * theano.tensor.dot(self.inpt, self.w) + self.b
        )
        self.y_out = theano.tensor.argmax(self.output, axis=1)
        self.inpt_dropout = theano.tensor.nnet.dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)),
            self.p_dropout
        )
        self.output_dropout = theano.tensor.nnet.softmax(
            theano.tensor.dot(self.inpt_dropout, self.w) + self.b
        )

    def cost(self, net):
        return -theano.tensor.mean(
            theano.tensor.log(self.output_dropout)[theano.tensor.arrange(net.y.shape[0]),
                                                    net.y]
        )
    
    def accuracy(self, y):
        return theano.tensor.mean(theano.tensor.eq(y, self.y_out))
    

def size(data):
    return data[0].get_value(borrow=True).shape[0]

def dropout_layer(layer, p_dropout):
    srng = theano.tensor.shared_randomstreams.RandomStreams(
        numpy.random.RandomState(0).randint(999999)
    )
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer * theano.tensor.cast(mask, theano.config.floatX)
