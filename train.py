import chapter1.network as network
import mnist_loader


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network.Network([784, 100, 10])
net.SGD(training_data, 20, 10, 0.001, test_data=test_data)
print(net.weights)
