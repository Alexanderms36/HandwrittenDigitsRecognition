import chapter6.network3 as cnn
import mnist_loader


mini_batch_size = 10

#заюзать расширенный датасет
training_data, validation_data, test_data = cnn.load_data_shared("./data/mnist_expanded.pkl.gz")
# ./data/mnist.pkl.gz

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
# print(training_data)
net.SGD(training_data, 40, mini_batch_size, 0.03, validation_data, test_data)
