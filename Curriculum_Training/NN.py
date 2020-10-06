"""
Created on Tue Sep 22 14:34:46 CST 2020

@author: GKSch
"""

import torch
import torch.nn as nn

# Dynamically coded class definition for Q learning neural network
# It was dynamically coded to allow user to select number of neurons and layers
class Neural_Network(nn.Module):
    
    # Initialization function
    # @param num_inputs - size of input vector
    # @param num_outputs - size of output vector
    # @param num_hidden_layers - total number of hidden layers in NN (total number of layers - 1)
    # @param num_neurons_in_layer - number of neurons in each hidden layer
    def __init__(self, num_inputs, num_outputs, num_hidden_layers, num_neurons_in_layer):
        
        # Initialize inherited class
        super(Neural_Network, self).__init__()
        
        # initialize class variables
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons_in_layer = num_neurons_in_layer
        
        # Initialize input fully-connected layer (fc1)
        self.fc1 = nn.Linear(num_inputs, num_neurons_in_layer)
        
        # Initialize num_hidden_layers fully connected hidden layers
        for current_hidden_layer in range(num_hidden_layers):
            
            # The current layer were are defining is the hidden layer index + 1 (because of the input layer)
            current_layer = current_hidden_layer + 1
            
            # This defines the current fully connected layer name as a string literal
            hidden_layer_name = "fc" + str(current_layer + 1)
            
            # For all but the last layer, create num_neurons_in_layer inputs and num_neurons_in_layer outputs
            if (current_hidden_layer != num_hidden_layers - 1):
                exec("self." + hidden_layer_name + "= nn.Linear(self.num_neurons_in_layer, self.num_neurons_in_layer)")

            # For the last layer, create num_neurons_in_layer inputs and num_outputs outputs
            else:
                exec("self." + hidden_layer_name + "= nn.Linear(self.num_neurons_in_layer, self.num_outputs)")

    # Feed-forward function
    # @param x - set of inputs
    # @return - set of outputs
    def forward(self, x):
        
        # Feed-forward x through all layers except the output layer (using tanh activation fnc)
        for current_layer in range(self.num_hidden_layers):
            current_layer_name = "fc" + str(current_layer + 1)
            x = eval("torch.tanh(self." + current_layer_name + "(x))")
        
        # Feed x through the output layer
        current_layer_name = "fc" + str(self.num_hidden_layers + 1)
        x = eval("self." + current_layer_name + "(x)")
        
        # Return the permuted x_copy (now in output form)
        return x