import chapter6.network3 as cnn
import mnist_loader


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
mini_batch_size = 10

net = cnn.Network([
        cnn.ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                          filter_shape=(20, 1, 5, 5), 
                          poolsize=(2, 2), 
                          activation_fn=cnn.ReLU
                          ),
        cnn.ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), 
                          filter_shape=(40, 20, 5, 5), 
                          poolsize=(2, 2), 
                          activation_fn=cnn.ReLU
                          ),
        cnn.FullyConnectedLayer(n_in=40*4*4,
                                n_out=1000, 
                                activation_fn=cnn.ReLU, 
                                p_dropout=0.5
                                ),
        cnn.FullyConnectedLayer(n_in=1000, 
                                n_out=1000, 
                                activation_fn=cnn.ReLU, 
                                p_dropout=0.5
                                ),
        cnn.SoftmaxLayer(n_in=1000, 
                         n_out=10, 
                         p_dropout=0.5
                         )], 
        mini_batch_size)

net.SGD(training_data, 40, mini_batch_size, 0.03, 
            validation_data, test_data)