import chapter1.network as network
import chapter3.network2 as network2
import mnist_loader


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network.Network([784, 100, 10])
net.SGD(training_data, 20, 10, 0.001, test_data=test_data)
print(net.weights)

# >>> import mnist_loader
# >>> training_data, validation_data, test_data = \
# ... mnist_loader.load_data_wrapper()
# >>> import network2
# >>> net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
# >>> net.large_weight_initializer()
# >>> net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data,
# ... monitor_evaluation_accuracy=True)

# >>> import mnist_loader 
# >>> training_data, validation_data, test_data = \
# ... mnist_loader.load_data_wrapper() 
# >>> import network2 
# >>> net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
# >>> net.large_weight_initializer()
# >>> net.SGD(training_data[:1000], 400, 10, 0.5,
# ... evaluation_data=test_data, lmbda = 0.1,
# ... monitor_evaluation_cost=True, monitor_evaluation_accuracy=True,
# ... monitor_training_cost=True, monitor_training_accuracy=True)

# >>> import mnist_loader
# >>> training_data, validation_data, test_data = \
# ... mnist_loader.load_data_wrapper()
# >>> import network2
# >>> net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
# >>> net.SGD(training_data, 30, 10, 0.5,
# ... lmbda = 5.0,
# ... evaluation_data=validation_data,
# ... monitor_evaluation_accuracy=True,
# ... monitor_evaluation_cost=True,
# ... monitor_training_accuracy=True,
# ... monitor_training_cost=True)

# >>> evaluation_cost, evaluation_accuracy, 
# ... training_cost, training_accuracy = net.SGD(training_data, 30, 10, 0.5,
# ... lmbda = 5.0,
# ... evaluation_data=validation_data,
# ... monitor_evaluation_accuracy=True,
# ... monitor_evaluation_cost=True,
# ... monitor_training_accuracy=True,
# ... monitor_training_cost=True)