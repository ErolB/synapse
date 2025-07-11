import pandas as pd
import numpy as np
import random
from sklearn import datasets
from utils import *

class Node:
    def __init__(self, id):
        self.id = id
        self.value = None
        self.current_z = None
        self.gradient = None
        self.activation = None
        self.current_output = None

    def get_value(self):
        """
        This method returns the current output of the node.
        """
        return self.current_output

    def get_gradient(self):
        """
        This method retrieves the current gradient at the node.
        """
        return self.gradient

    def set_gradient(self, value):
        """
        value: updated gradient

        This method retrieves the current gradient at the node.
        """
        self.gradient = value

    def clear(self):
        """
        This method clears the stored gradient and output.
        """
        self.current_output = None
        self.current_z = None
        self.gradient = None


class InputNode(Node):
    """
    This class defines the behavior of input nodes. The purpose of an input node is to store and transmit the inputs
    of the network. No computation happens here.
    """
    def __init__(self, id):
        super().__init__(id)
        self.activation = LinearActivation  # hardwired linear function

    def set_value(self, new_value):
        """
        new_value: value to be stored and transmitted to the neurons

        Both 'value' and 'current_z' are set to 'new_value'. These attributes exist to ensure compatibility with neurons
        """
        self.value = new_value
        self.current_z = new_value
        self.current_output = new_value

    def evaluate(self):
        """
        This method returns the input value stored in this node.
        """
        return self.value

class Neuron(Node):
    """
    This class defines the behavior of neuron. Neurons retrieve values from connected nodes, compute a weighted sum, and
    apply an activation function.
    """
    def __init__(self, id, inputs, activation=ReluActivation):
        super().__init__(id)
        self.previous = inputs  # a list of references to preceding nodes
        self.n_inputs = len(inputs)
        self.weight_mapping = {inputs[i]: random.random()*0.01 for i in range(self.n_inputs)}
        # Node (neuron or input node) objects are mapped to weights. Weights are randomly initialized and can be
        # updated during backpropagation
        self.bias = random.random() * 0.01
        self.activation = activation

    def get_bias(self):
        """
        This method returns the current bias of the neuron.
        """
        return self.bias

    def set_bias(self, value):
        """
        value: updated bias

        This method allows the bias to be updated
        """
        self.bias = value

    def get_weights(self):
        """
        This method returns an array containing the weights of the neuron
        """
        return np.array(list(self.weight_mapping.values()))

    def evaluate(self):
        """
        This method returns the output of the neuron, if defined. If the output is undefined, it calls the 'evaluate'
        method of preceding nodes to retrieve their outputs, computes a weighted average, and applies an activation
        function.
        """
        if self.current_output is None:
            inputs = np.array([node.evaluate() for node in self.previous])
            z = np.dot(inputs, self.get_weights()) + self.bias
            self.current_z = z
            output = self.activation.function(z)
            self.current_output = output
        return self.current_output


class Network:
    """
    This is a base class that defines the behavior of a network.
    """
    def __init__(self, n_inputs, n_outputs, cost_function=MSE, learning_rate=0.001):
        self.input_layer = {i: InputNode(i) for i in range(n_inputs)}  # initializes input layer
        self.neurons = {}  # a dictionary that will later contain hidden neurons
        self.output_layer = {i: None for i in range(n_outputs)}  # a dictionary that will later contain output neurons
        self.cost_function = cost_function
        self.learning_rate = learning_rate
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_neurons = None  # defined in subclasses
        self.input_connections = None  # adjacency matrix for input nodes and hidden neurons (defined in subclasses)
        self.hidden_connections = None  # adjacency matrix for hidden neurons (defined in subclasses)
        self.output_connections = None  # adjacency matrix for hidden neurons and output neurons (defined in subclasses)

    def build_network(self):
        """
        Based on the adjacency matrices defined during initialization, create hidden neurons with the appropriate
        connections. The network is built recursively.
        """
        def build_neuron(neuron_index):
            # define recursive function
            if self.neurons[neuron_index] is not None:
                return self.neurons[neuron_index]  # avoids redundantly initializing neurons
            previous_nodes = []
            # find all preceding nodes, both in the input layer and set of hidden neurons
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
        """
        input values: These are the values to be passed into the input nodes
        clear_node_outputs: This is a Boolean flag that indicates whether the node outputs and gradients should be
            cleared after an output is obtained. This flag should be 'False' during the training process, but should be
            'True' when using the fully trained network

        This method passes inputs to the network and obtains an output. It calls the 'evaluate' method of the neurons
        in the output layer, which recursively calculates the outputs of all preceding nodes.
        """
        if len(input_values) != len(self.input_layer):
            raise Exception
        for node, value in zip(self.input_layer.values(), input_values):
            node.set_value(value)
        outputs = [node.evaluate() for node in self.output_layer.values()]
        if clear_node_outputs:
            self.clear_node_outputs()
        return outputs

    def get_next(self, node):
        """
        node: This is a reference to the node being examined.

        This method returns all nodes that accept inputs from 'node'. Neurons only contain references to all preceding
        nodes, which makes this function necessary.
        """
        if type(node) == InputNode:
            next_neurons = [self.neurons[i] for i in range(self.n_neurons) if self.input_connections[node.id, i] == 1]
        else:
            hidden_ids = [i for i in range(self.n_neurons) if self.hidden_connections[i, node.id] == 1]
            output_ids = [i for i in range(self.n_outputs) if self.output_connections[i, node.id] == 1]
            next_neurons = [self.neurons[i] for i in hidden_ids] + [self.output_layer[i] for i in output_ids]
        return [item for item in next_neurons if item is not None]

    def compute_all_gradients(self, expected_output):
        """
        expected_output: The expected outputs from the training set

        This method sets the gradients for all nodes
        """
        # define recursive function
        def compute_gradient(neuron):
            next_neurons = self.get_next(neuron)  # get the following neurons in the network
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
        for i, grad in enumerate(output_gradients):
            self.output_layer[i].set_gradient(grad)
        # start at input layer and solve recursively
        for input_node in self.input_layer.values():
            compute_gradient(input_node)

    def adjust_weights(self):
        """
        This method recursively adjusts the weights of each neuron based on the gradients calculated previously.
        """
        def adjust_neuron(neuron):
            if type(neuron) == InputNode:
                return  # no weights to adjust
            for previous_node in neuron.weight_mapping:
                if previous_node.get_gradient() is None:
                    continue  # skip if no gradient was calculated
                w = neuron.weight_mapping[previous_node]
                new_w = w - (neuron.get_gradient() * previous_node.get_value() * self.learning_rate)
                neuron.weight_mapping[previous_node] = new_w
                adjust_neuron(previous_node)  # call function for previous neuron
            # update bias
            current_bias = neuron.get_bias()
            new_bias = current_bias - (neuron.get_gradient() * self.learning_rate)
            neuron.set_bias(new_bias)
        # call 'adjust weights' for all output nodes
        for node in self.output_layer.values():
            adjust_neuron(node)

    def train(self, x, y, epochs=20):
        """
        x: training data
        y: training labels
        epochs: number of times to iterate through the training set

        This method repeatedly calculates gradients and updates parameters (weights and biases)
        """
        if x.shape[1] != self.n_inputs or y.shape[1] != self.n_outputs:
            assert 'mismatch'
        for _ in range(epochs):
            for i in range(x.shape[0]):  # iterate through training examples
                self.solve(list(x[i,:]))  # forward pass
                self.compute_all_gradients(list(y[i,:]))  # calculate all weights
                self.adjust_weights()  # update weights based on gradients
                self.clear_node_outputs()  # clear outputs and gradients

    def evaluate_accuracy(self, x_test, y_test):
        """
        x_test: test data
        y_test: test labels
        
        This method evaluates the accuracy of the network on test data
        """
        correct = 0
        total = len(x_test)
        
        for i in range(total):
            test_output = self.solve(x_test[i,:], clear_node_outputs=True)
            expected_label = list(y_test[i,:]).index(1)
            actual_label = list(test_output).index(max(test_output))
            if expected_label == actual_label:
                correct += 1
        
        return correct / total

    def copy_connections(self):
        """
        This method creates a deep copy of the current connection matrices
        """
        return {
            'input_connections': copy.deepcopy(self.input_connections),
            'hidden_connections': copy.deepcopy(self.hidden_connections),
            'output_connections': copy.deepcopy(self.output_connections)
        }

    def restore_connections(self, connections):
        """
        This method restores connection matrices from a saved state
        """
        self.input_connections = copy.deepcopy(connections['input_connections'])
        self.hidden_connections = copy.deepcopy(connections['hidden_connections'])
        self.output_connections = copy.deepcopy(connections['output_connections'])

    def clear_node_outputs(self):
        """
        This method recursively clears all outputs and gradients
        """
        def clear(node):
            if node.get_value() is None:
                return
            node.clear()
            if type(node) == InputNode:
                return
            for previous_node in node.previous:
                clear(previous_node)
        for output_node in self.output_layer.values():
            clear(output_node)  # start with output nodes


class RandomNetwork(Network):
    """
    This class is for creating networks with random connections
    """
    def __init__(self, n_neurons, n_inputs, n_outputs, p_connect_input=0.1,p_connect_output=0.1, max_distance=None,
                 cost_function=MSE, learning_rate=0.001):
        super().__init__(n_inputs, n_outputs, cost_function=cost_function, learning_rate=learning_rate)
        self.n_neurons = n_neurons
        self.neurons = {i: None for i in range(n_neurons)}  # a dictionary that will later contain hidden neurons
        self.input_connections = np.zeros((n_inputs, n_neurons))  # adjacency matrix for input nodes and hidden neurons
        self.hidden_connections = np.zeros((n_neurons, n_neurons))  # adjacency matrix for hidden neurons
        self.output_connections = np.zeros((n_outputs, n_neurons))  # adjacency matrix for hidden neurons and outputs
        self.max_distance = max_distance if max_distance else int(np.sqrt(n_neurons))
        self.define_connections(p_connect_input=p_connect_input, p_connect_output=p_connect_output, max_distance=self.max_distance)

    def define_connections(self, p_connect_input=0.1, p_connect_output=0.1, max_distance=None):
        if max_distance is None:
            max_distance = int(np.sqrt(self.n_neurons))
        for i in range(self.n_inputs):
            for j in range(self.n_neurons):
                if random.random() <= p_connect_input:
                    self.input_connections[i, j] = 1
            # randomly define connections to output layer
        for i in range(self.n_outputs):
            for j in range(self.n_neurons):
                if random.random() <= p_connect_output:
                    self.output_connections[i, j] = 1
        # define connections in hidden portion
        input_nodes = list(range(self.n_neurons))
        random.shuffle(input_nodes)
        output_nodes = list(range(self.n_neurons))
        random.shuffle(output_nodes)
        for i in input_nodes:
            for j in output_nodes:
                if i == j:
                    continue
                test_matrix = copy.copy(self.hidden_connections)  # a temporary matrix used to test viability
                test_matrix[i, j] = 1
                # check for cycles
                if detect_cycle(test_matrix):
                    continue
                # check maximum path length
                max_distance_exceeded = False
                for k in range(self.n_outputs):
                    connected_nodes = [l for l in range(self.n_inputs) if self.output_connections[k, l] == 1]
                    for n in connected_nodes:
                        if max_path_length(n, test_matrix) > max_distance:
                            max_distance_exceeded = True
                            break
                    if max_distance_exceeded:
                        break
                if not max_distance_exceeded:
                    self.hidden_connections = test_matrix

    def prune_network(self, x_train, y_train, x_test, y_test, max_cycles=50, convergence_threshold=0.001, 
                     mutation_rate=0.1, training_epochs=10):
        """
        Evolutionary pruning algorithm that adds/removes connections to improve network performance
        
        x_train, y_train: training data
        x_test, y_test: test data for evaluation
        max_cycles: maximum number of pruning cycles
        convergence_threshold: stop if accuracy improvement is below this threshold
        mutation_rate: probability of modifying each connection
        training_epochs: epochs to train after each structural change
        """
        print("Starting network pruning...")
        
        # Get initial baseline accuracy
        self.build_network()
        self.train(x_train, y_train, epochs=training_epochs)
        best_accuracy = self.evaluate_accuracy(x_test, y_test)
        best_connections = self.copy_connections()
        
        print(f"Initial accuracy: {best_accuracy:.4f}")
        
        no_improvement_count = 0
        
        for cycle in range(max_cycles):
            print(f"\nPruning cycle {cycle + 1}/{max_cycles}")
            
            # Save current state
            current_connections = self.copy_connections()
            
            # Apply mutations to the network structure
            mutations_applied = self._apply_structural_mutations(mutation_rate)
            
            if mutations_applied == 0:
                print("No valid mutations found, continuing...")
                continue
            
            print(f"Applied {mutations_applied} structural mutations")
            
            # Rebuild and retrain the network
            try:
                self.build_network()
                self.train(x_train, y_train, epochs=training_epochs)
                current_accuracy = self.evaluate_accuracy(x_test, y_test)
                
                print(f"Current accuracy: {current_accuracy:.4f}")
                
                # Check if this is an improvement
                if current_accuracy > best_accuracy + convergence_threshold:
                    improvement = current_accuracy - best_accuracy
                    print(f"Improvement found: +{improvement:.4f}")
                    best_accuracy = current_accuracy
                    best_connections = self.copy_connections()
                    no_improvement_count = 0
                else:
                    # Revert to previous state if no improvement
                    print("No improvement, reverting changes")
                    self.restore_connections(current_connections)
                    no_improvement_count += 1
                    
            except Exception as e:
                print(f"Error during training: {e}, reverting changes")
                self.restore_connections(current_connections)
                no_improvement_count += 1
            
            # Check for convergence
            if no_improvement_count >= 10:
                print("No improvement for 10 cycles, stopping early")
                break
        
        # Restore best configuration
        self.restore_connections(best_connections)
        self.build_network()
        self.train(x_train, y_train, epochs=training_epochs)
        
        final_accuracy = self.evaluate_accuracy(x_test, y_test)
        print(f"\nPruning complete!")
        print(f"Final accuracy: {final_accuracy:.4f}")
        print(f"Total improvement: {final_accuracy - (best_accuracy - (final_accuracy - best_accuracy)):.4f}")
        
        return final_accuracy

    def _apply_structural_mutations(self, mutation_rate):
        """
        Apply random structural mutations to the network
        """
        mutations_applied = 0
        
        # Mutate input connections
        for i in range(self.n_inputs):
            for j in range(self.n_neurons):
                if random.random() < mutation_rate:
                    if self._can_modify_input_connection(i, j):
                        self.input_connections[i, j] = 1 - self.input_connections[i, j]
                        mutations_applied += 1
        
        # Mutate output connections
        for i in range(self.n_outputs):
            for j in range(self.n_neurons):
                if random.random() < mutation_rate:
                    if self._can_modify_output_connection(i, j):
                        self.output_connections[i, j] = 1 - self.output_connections[i, j]
                        mutations_applied += 1
        
        # Mutate hidden connections
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                if i != j and random.random() < mutation_rate:
                    if self._can_modify_hidden_connection(i, j):
                        self.hidden_connections[i, j] = 1 - self.hidden_connections[i, j]
                        mutations_applied += 1
        
        return mutations_applied

    def _can_modify_input_connection(self, input_idx, neuron_idx):
        """
        Check if an input connection can be safely modified
        """
        # Always allow input connection modifications
        return True

    def _can_modify_output_connection(self, output_idx, neuron_idx):
        """
        Check if an output connection can be safely modified
        """
        # Ensure at least one connection to each output
        if self.output_connections[output_idx, neuron_idx] == 1:
            return np.sum(self.output_connections[output_idx, :]) > 1
        return True

    def _can_modify_hidden_connection(self, from_neuron, to_neuron):
        """
        Check if a hidden connection can be safely modified without creating cycles or violating constraints
        """
        # Test the modification
        test_matrix = copy.deepcopy(self.hidden_connections)
        test_matrix[from_neuron, to_neuron] = 1 - test_matrix[from_neuron, to_neuron]
        
        # Check for cycles
        if detect_cycle(test_matrix):
            return False
        
        # Check maximum path length constraint
        for k in range(self.n_outputs):
            connected_nodes = [l for l in range(self.n_neurons) if self.output_connections[k, l] == 1]
            for n in connected_nodes:
                if max_path_length(n, test_matrix) > self.max_distance:
                    return False
        
        return True


class LayeredNetwork(Network):
    def __init__(self, n_inputs, n_outputs, n_layers, hidden_layer_width, cost_function=MSE, learning_rate=0.001):
        super().__init__(n_inputs, n_outputs, cost_function=cost_function, learning_rate=learning_rate)
        self.n_neurons = n_layers * hidden_layer_width
        self.neurons = {i: None for i in range(self.n_neurons)}  # a dictionary that will later contain hidden neurons
        self.input_connections = np.zeros((n_inputs, self.n_neurons))  # adjacency matrix for input nodes and hidden neurons
        self.hidden_connections = np.zeros((self.n_neurons, self.n_neurons))  # adjacency matrix for hidden neurons
        self.output_connections = np.zeros((n_outputs, self.n_neurons))  # adjacency matrix for hidden neurons and outputs
        self.define_connections(n_layers, hidden_layer_width)

    def define_connections(self, n_layers, hidden_layer_width):
        for i in range(self.n_inputs):
            for j in range(hidden_layer_width):
                self.input_connections[i,j] = 1  # connect all input nodes to first layer
        for i in range(self.n_outputs):
            for j in range(hidden_layer_width):
                self.output_connections[i, self.n_neurons-j-1] = 1  # connect all output neurons to last layer
        for i in range(n_layers-1):
            for j in range(hidden_layer_width):
                for k in range(hidden_layer_width):
                    input_node_index = (i+1) * hidden_layer_width + j
                    output_node_index = i * hidden_layer_width + k
                    self.hidden_connections[input_node_index, output_node_index] = 1


if __name__ == '__main__':
    data = datasets.load_iris()
    flower_info = data['data']
    targets = one_hot(data['target'])
    # shuffle data
    order = list(range(len(flower_info)))
    random.shuffle(order)
    flower_info = flower_info[order,:]
    targets = targets[order,:]
    training_data = flower_info[:120,:]
    training_targets = targets[:120,:]
    testing_data = flower_info[120:,:]
    testing_targets = targets[120:,:]
    
    # train model with pruning
    net = RandomNetwork(40, 4, 3, p_connect_input=0.2, p_connect_output=0.2, learning_rate=0.01)
    print('Building initial network...')
    net.build_network()
    
    # Apply pruning to optimize network structure
    final_accuracy = net.prune_network(
        training_data, training_targets, 
        testing_data, testing_targets,
        max_cycles=30,
        convergence_threshold=0.005,
        mutation_rate=0.05,
        training_epochs=15
    )
    
    print(f"\nFinal optimized accuracy: {final_accuracy:.4f}")