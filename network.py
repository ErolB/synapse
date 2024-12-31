import pandas as pd
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from sklearn import datasets
from utils import *

class InputNode:
    def __init__(self, id):
        self.id = id
        self.value = None
        self.current_z = None
        self.gradient = None
        self.activation = LinearActivation  # hardwired linear function

    def set_value(self, new_value):
        self.value = new_value
        self.current_z = new_value

    def get_value(self):
        return self.value

    def get_gradient(self):
        return self.gradient

    def set_gradient(self, value):
        self.gradient = value

    def evaluate(self):
        return self.value

    def clear(self):
        self.value = None
        self.current_z = None
        self.gradient = None

class Neuron:
    def __init__(self, id, inputs, activation=ReluActivation):
        self.id = id
        self.previous = inputs
        self.n_inputs = len(inputs)
        self.weight_mapping = {inputs[i]: random.random()*0.1 for i in range(self.n_inputs)}
        self.bias = random.random() * 0.1
        self.current_output = None
        self.current_z = None  # weighted sum of inputs
        self.gradient = None  # derivative of cost function with respect to z
        self.activation = activation

    def get_bias(self):
        return self.bias

    def set_bias(self, value):
        self.bias = value

    def get_weights(self):
        return np.array(list(self.weight_mapping.values()))

    def evaluate(self):
        if self.current_output is None:
            inputs = np.array([node.evaluate() for node in self.previous])
            z = np.dot(inputs, self.get_weights()) + self.bias
            self.current_z = z
            output = self.activation.function(z)
            self.current_output = output
        return self.current_output

    def get_value(self):
        return self.current_output

    def get_gradient(self):
        return self.gradient

    def set_gradient(self, value):
        self.gradient = value

    def clear(self):
        self.current_output = None
        self.current_z = None
        self.gradient = None


class Network:
    def __init__(self, n_neurons, n_inputs, n_outputs, p_connect_hidden=0.5, p_connect_input=0.1, p_connect_output=0.1,
                 max_distance=None, cost_function=MSE, learning_rate=0.01):
        if max_distance is None:
            max_distance = int(np.sqrt(n_neurons))
        self.input_layer = {i: InputNode(i) for i in range(n_inputs)}
        self.neurons = {i: None for i in range(n_neurons)}
        self.output_layer = {i: None for i in range(n_outputs)}
        self.cost_function = cost_function
        self.learning_rate = learning_rate
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        # define connections to input layer
        self.input_connections = np.zeros((n_inputs, n_neurons))
        for i in range(n_inputs):
            for j in range(n_neurons):
                if random.random() <= p_connect_input:
                    self.input_connections[i, j] = 1
        # define connections to output layer
        self.output_connections = np.zeros((n_outputs, n_neurons))
        for i in range(n_outputs):
            for j in range(n_neurons):
                if random.random() <= p_connect_output:
                    self.output_connections[i, j] = 1
        # define connections in hidden portion
        self.hidden_connections = np.zeros((n_neurons, n_neurons))
        for i in range(n_neurons):
            for j in range(n_neurons):
                if random.random() <= p_connect_hidden:
                    test_matrix = copy.copy(self.hidden_connections)
                    test_matrix[i,j] = 1
                    # check for cycles
                    if detect_cycle(test_matrix):
                        continue
                    # check maximum path length
                    max_distance_exceeded = False
                    for k in range(n_outputs):
                        connected_nodes = [l for l in range(n_inputs) if self.output_connections[k,l]==1]
                        for n in connected_nodes:
                            if max_path_length(n, test_matrix) > max_distance:
                                max_distance_exceeded = True
                                break
                        if max_distance_exceeded:
                            break
                    if not max_distance_exceeded:
                        self.hidden_connections = test_matrix


    def build_network(self):
        # define recursive function
        def build_neuron(neuron_index):
            if self.neurons[neuron_index] is not None:
                return self.neurons[neuron_index]
            previous_nodes = []
            for i, input_node in enumerate(self.input_connections[:, neuron_index]):
                if input_node == 1:
                    previous_nodes.append(self.input_layer[i])
            for i, hidden_node in enumerate(self.hidden_connections[neuron_index, :]):
                if hidden_node == 1:
                    previous_nodes.append(build_neuron(i))
            new_neuron = Neuron(neuron_index, previous_nodes)
            self.neurons[neuron_index] = new_neuron
            return new_neuron
        # clear existing neurons
        self.neurons = {i: None for i in range(self.n_neurons)}
        # build output neurons
        for i in range(self.n_outputs):
            connection_indices = [j for j in range(self.n_neurons) if self.output_connections[i,j]==1]
            connections_to_output = [build_neuron(j) for j in connection_indices]
            self.output_layer[i] = Neuron(i, connections_to_output)

    def solve(self, input_values, clear_node_outputs=False):
        if len(input_values) != len(self.input_layer):
            raise Exception
        for node, value in zip(self.input_layer.values(), input_values):
            node.set_value(value)
        outputs = [node.evaluate() for node in self.output_layer.values()]
        if clear_node_outputs:
            self.clear_node_outputs()
        return outputs

    def get_next(self, node):
        if type(node) == InputNode:
            next_neurons = [self.neurons[i] for i in range(self.n_neurons) if self.input_connections[node.id, i] == 1]
        else:
            hidden_ids = [i for i in range(self.n_neurons) if self.hidden_connections[i, node.id] == 1]
            output_ids = [i for i in range(self.n_outputs) if self.output_connections[i, node.id] == 1]
            next_neurons = [self.neurons[i] for i in hidden_ids] + [self.output_layer[i] for i in output_ids]
        return [item for item in next_neurons if item is not None]

    def compute_all_gradients(self, expected_output):
        # define recursive function
        def compute_gradient(neuron):
            next_neurons = self.get_next(neuron)
            if len(next_neurons) == 0:
                return
            # retrieve relevant weights
            weights = []
            for n in next_neurons:
                for i in n.weight_mapping:
                    if i == neuron:
                        weights.append(n.weight_mapping[i])
            # get upstream gradients
            gradients = []
            for n in next_neurons:
                if n.get_gradient() is None:
                    compute_gradient(n)
                gradients.append(n.get_gradient())
            # calculate gradient at the current neuron
            neuron.gradient = np.dot(np.array(gradients), np.array(weights)) * neuron.activation.derivative(neuron.current_z)

        # set the gradients in the output layer
        output_values = np.array([self.output_layer[i].get_value() for i in self.output_layer])
        output_gradients = self.cost_function.derivative(output_values, expected_output)
        print('cost')
        print(self.cost_function.function(output_values, expected_output))
        for i, grad in enumerate(output_gradients):
            self.output_layer[i].set_gradient(grad)
        # start at input layer and solve recursively
        for input_node in self.input_layer.values():
            compute_gradient(input_node)

    def adjust_weights(self):
        def adjust_neuron(neuron):
            if type(neuron) == InputNode:
                return
            for previous_node in neuron.weight_mapping:
                w = neuron.weight_mapping[previous_node]
                new_w = w * -neuron.get_gradient() * previous_node.get_value() * self.learning_rate
                neuron.weight_mapping[previous_node] = new_w
            current_bias = neuron.get_bias()
            new_bias = current_bias * -neuron.get_gradient() * self.learning_rate
            neuron.set_bias(new_bias)
        for node in self.output_layer.values():
            adjust_neuron(node)

    def train(self, x, y, epochs=20):
        if x.shape[1] != self.n_inputs or y.shape[1] != self.n_outputs:
            assert 'mismatch'
        for _ in range(epochs):
            for i in range(x.shape[0]):
                self.solve(list(x[i,:]))
                self.compute_all_gradients(list(y[i,:]))
                self.adjust_weights()
                self.clear_node_outputs()


    def clear_node_outputs(self):
        def clear(node):
            if node.get_value() is None:
                return
            node.clear()
            if type(node) == InputNode:
                return
            for previous_node in node.previous:
                clear(previous_node)
        for output_node in self.output_layer.values():
            clear(output_node)


if __name__ == '__main__':
    data = datasets.load_iris()
    flower_info = data['data']
    targets = one_hot(data['target'])
    net = Network(20, 4, 3, p_connect_input=0.2, p_connect_hidden=0.5, p_connect_output=0.2, learning_rate=1)
    print('building network')
    net.build_network()
    print('forward propagation')
    #print(net.solve([0.1,0.3,0.4], clear_node_outputs=False))
    #net.compute_all_gradients([0.5,0.5])
    print('done')
    net.train(flower_info, targets)
    print('test')
    print(net.solve(list(flower_info[-1,:])))





