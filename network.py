import pandas as pd
import numpy as np
import random
from utils import *


class Neuron:
    def __init__(self, n_inputs, activation='relu'):
        self.weights = np.array([random.random()*0.01 for _ in range(n_inputs)])
        self.bias = random.random() * 0.01
        if activation == 'relu':
            self.activation = relu
        elif activation == 'sigmoid':
            self.activation = sigmoid
        else:
            self.activation = linear

    def evaluate(self, inputs):
        z = np.dot(inputs, self.weights) + self.bias
        return self.activation(z)


class Network:
    def __init__(self, n_neurons, n_inputs, n_outputs, p_connect_hidden=0.3, p_connect_input=0.1, p_connect_output=0.1):
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        # define connections in hidden portion
        self.hidden_connections = np.zeros((n_neurons, n_neurons))
        for i in range(n_neurons):
            for j in range(n_neurons):
                if random.random() <= p_connect_hidden:
                    self.hidden_connections[i,j] = 1
        # define connections to input layer
        self.input_connections = np.zeros((n_inputs, n_neurons))
        for i in range(n_inputs):
            for j in range(n_neurons):
                if random.random() <= p_connect_input:
                    self.input_connections[i,j] = 1
        # define connection to output layer
        self.output_connections = np.zeros((n_outputs, n_neurons))
        for i in range(n_outputs):
            for j in range(n_neurons):
                if random.random() <= p_connect_output:
                    self.output_connections[i,j] = 1

    def build_network(self):
        for i in range(self.n_neurons):
            total_inputs = self.input_connections[i]


