import pandas as pd
import numpy as np
import random
import copy

# activation functions + derivatives

class SigmoidActivation:
    """
    This class defines the sigmoid activation function and its derivative
    """
    def __init__(self):
        self.name = 'sigmoid'

    @staticmethod
    def function(z):
        return np.exp(z) / (1 + np.exp(z))

    @staticmethod
    def derivative(z):
        temp = np.exp(z) / (1 + np.exp(z))
        return temp * (1 - temp)

class ReluActivation:
    """
    This class defines the rectified linear unit (ReLU) activation function and its derivative
    """
    def __init__(self):
        self.name = 'relu'

    @staticmethod
    def function(z):
        if z < 0:
            return 0
        else:
            return z

    @staticmethod
    def derivative(z):
        if z < 0:
            return 0
        else:
            return 1

class LinearActivation:
    """
    This class defines the linear activation and its derivative
    """
    def __init__(self):
        self.name = 'linear'

    @staticmethod
    def function(z):
        return z

    @staticmethod
    def derivative(z):
        return 1

# loss functions

class MSE:
    """
    This class defines the mean squared error (MSE) loss function and its derivative
    """
    def __init__(self):
        self.name = 'mse'

    @staticmethod
    def function(output, target):
        return sum([(o-y)**2 for o, y in zip(output, target)]) / len(output) * 0.5

    @staticmethod
    def derivative(output, target):
        return [o-y for o, y in zip(output, target)]

# graph functions

def dfs_cycle(node_index, adj_matrix, visited, coverage):
    """
    node_index: the index of the current node
    adj_matrix: an adjacency matrix for all hidden neurons
    visited: a dictionary mapping node indices to Boolean values representing whether a node has been examined in the current run
    coverage: a dictionary mapping node indices to Boolean values representing whether a node has been visited at any point

    This function examines nodes in a directed graph to detect cycles via a depth-first search
    """
    if visited[node_index]:
        return True
    if coverage[node_index]:
        return False
    visited[node_index] = True
    coverage[node_index] = True
    next_nodes = [i for i in range(adj_matrix.shape[0]) if adj_matrix[i, node_index]==1]
    for n in next_nodes:
        cycle_complete = dfs_cycle(n, adj_matrix, copy.copy(visited), coverage)
        if cycle_complete:
            return True
    return False  # no cycles found

def detect_cycle(adj_matrix):
    """
    adj_matrix: an adjacency matrix for all hidden neurons

    This function iterates through the hidden neurons, using each one as a starting point for 'dfs_cycle'. The purpose
    is to detect any cycles in the network
    """
    coverage = {i: False for i in range(adj_matrix.shape[0])}
    for starting_point in range(adj_matrix.shape[0]):
        visited = {i: False for i in range(adj_matrix.shape[0])}
        if dfs_cycle(starting_point, adj_matrix, visited, coverage):
            return True
    return False

def dfs_distance(node_index, distance, adj_matrix):
    """
    node_index: the index of the current node
    distance: the distance of the path taken to reach the current node
    adj_matrix: an adjacency matrix for all hidden neurons

    This function performs a depth-first search to determine the maximum path length through the network from a given
    starting point.
    """
    previous_nodes = [i for i in range(adj_matrix.shape[0]) if adj_matrix[node_index, i] == 1]
    if len(previous_nodes) == 0:
        return distance
    lengths = [dfs_distance(n, distance+1, adj_matrix) for n in previous_nodes]
    return max(lengths)

def max_path_length(node_index, adj_matrix):
    """
    node_index: the index of the starting point
    adj_matrix: an adjacency matrix for all hidden neurons

    This function calls 'dfs_distance' with a distance parameter of zero
    """
    return dfs_distance(node_index, 0, adj_matrix)


# miscellaneous function

def one_hot(data_1d):
    """
    data_1d: a one-dimensional array of numerical data

    This function converts a one-dimensional array to one-hot format. The length of the output array is the same as the
    length of the input, and the width is equal to the maximum value in the input. Each row contains a single value of
    one, while the other values are zero.
    """
    data_array = np.zeros((data_1d.shape[0], max(data_1d)+1))
    for i in range(data_1d.shape[0]):
        data_array[i,data_1d[i]] = 1
    return data_array


if __name__ == "__main__":
    adj = [[0,0,0,0],
           [1,0,0,0],
           [1,0,0,1],
           [0,1,0,0]]
    print(max_path_length(0, np.array(adj)))

