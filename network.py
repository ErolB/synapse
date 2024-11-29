import pandas as pd
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from utils import *

class InputNode:
    def __init__(self):
        self.value = None

    def set_value(self, new_value):
        self.value = new_value

    def evaluate(self):
        return self.value

class Neuron:
    def __init__(self, inputs, activation='relu'):
        self.previous = inputs
        self.n_inputs = len(inputs)
        self.weights = np.array([random.random()*0.01 for _ in range(self.n_inputs)])
        self.bias = random.random() * 0.01
        # define activation functions
        if activation == 'relu':
            self.activation = relu
        elif activation == 'sigmoid':
            self.activation = sigmoid
        else:
            self.activation = linear

    def evaluate(self):
        inputs = np.array([node.evaluate() for node in self.previous])
        z = np.dot(inputs, self.weights) + self.bias
        return self.activation(z)


class Network:
    def __init__(self, n_neurons, n_inputs, n_outputs, p_connect_hidden=0.5, p_connect_input=0.1, p_connect_output=0.1,
                 max_distance=None):
        if max_distance is None:
            max_distance = int(np.sqrt(n_neurons))
        self.neurons = {i: None for i in range(n_neurons)}
        self.input_layer = {i: InputNode() for i in range(n_inputs)}
        self.output_layer = {i: None for i in range(n_outputs)}
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
            new_neuron = Neuron(previous_nodes)
            self.neurons[neuron_index] = new_neuron
            return new_neuron
        # clear existing neurons
        self.neurons = {i: None for i in range(self.n_neurons)}
        # build output neurons
        for i in range(self.n_outputs):
            connection_indices = [j for j in range(self.n_neurons) if self.output_connections[i,j]==1]
            connections_to_output = [build_neuron(j) for j in connection_indices]
            self.output_layer[i] = Neuron(connections_to_output)



if __name__ == '__main__':
    net = Network(50, 5, 2, p_connect_input=0.1, p_connect_hidden=0.2, p_connect_output=0.1)
    net.build_network()






