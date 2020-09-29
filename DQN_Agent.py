"""
Created on Tuesday Sep 22 13:59:21 CST 2020

@author: Grayson Schaer
"""

import torch
import numpy as np
import random
import NN
import copy
import sys
from operator import itemgetter 

class DQN_Agent():
    
    # Agent constructor
    # @param num_actions - integer size of the action space
    # @param state_dimension - defines the dimension of the state space
    # @param max_data_set_size - The maximum number of experiences that can be stored
    # @param start_data_set_size - The number of frames a uniform random policy is run to populate the data set before learning begins
    # @param sequence_size - The number of previous states that define an input sequence
    # @param minibatch_size - The number of experiences sampled from the data set during Q improvement
    # @param num_hidden_layers - total number of hidden layers in NN (total number of layers - 1)
    # @param num_neurons_in_layer - number of neurons in each hidden layer
    # @param target_reset_interval -  the number of quality function improvement steps taken before the current Q is cloned to the target Q
    # @param alpha - learning rate
    # @param gamma - discount ratio
    # @param epsilon_start - the starting elpsilon-greedy action choice parameter
    # @param epsilon_end - the minimum elpsilon-greedy action choice parameter
    # @param epsilon_depreciation_factor - the rate at which the epsilon parameter decreases from maximum to minimum 
    def __init__(self, num_actions, state_dimension, max_data_set_size, start_data_set_size, sequence_size, minibatch_size, num_hidden_layers, num_neurons_in_layer, target_reset_interval, alpha, gamma, epsilon_start, epsilon_end, epsilon_depreciation_factor):
        
        # Initialize the input arguments
        self.num_actions = num_actions
        self.state_dimension = state_dimension
        self.max_data_set_size = max_data_set_size
        self.start_data_set_size = start_data_set_size
        self.sequence_size = sequence_size
        self.minibatch_size = minibatch_size
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons_in_layer = num_neurons_in_layer
        self.target_reset_interval = target_reset_interval
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_depreciation_factor = epsilon_depreciation_factor
        
        # Initialze the NN used to estimate Q(s,a) and the optimizer used for gradient descent
        self.Q_Network = NN.Neural_Network(self.sequence_size * self.state_dimension, self.num_actions, self.num_hidden_layers, self.num_neurons_in_layer)
        self.opt = torch.optim.Adam(self.Q_Network.parameters() , lr=self.alpha)
        self.loss_fn = torch.nn.MSELoss()
            
        # Initialize the NN used to provide the target for Q improvement
        self.Q_Target_Network = copy.deepcopy(self.Q_Network)
        self.steps_since_reset = 0
        
        # Initialize the sequence of states used as the input to the Q_Network (data structure : QUEUE)
        self.state_sequence = np.asarray([])
        self.state_sequence_is_full = False
        
        # Initializes data_set
        self.state_sequence_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.next_state_sequence_memory = []
        self.data_set_at_start_size = False
        
        # Defines the variables used in the log
        self.r_tot = 0.0
        self.r_tot_discount = 0.0
        self.curr_step = 0
        
        # Defines the variables used in the logbook
        self.entry_number = 1
        self.highest_r_tot = -1 * sys.float_info.max
        self.r_tot_prev = 0.0
        self.highest_r_tot_discounted = 0.0
        
        # Creates a log that tracks learning during a single episode set
        # @entry s - the state at each iteration step
        # @entry a - the action taken at each iteration step
        # @entry r - the reward given at each iteration step
        # @entry r_tot - the sum of all rewards given up to the current iteration step
        # @entry r_tot_discount - the sum of all the discounted rewards given up to the current iteration step
        # @entry loss - the output of the loss function at the current simulation step
        self.log = {
            's': [],
            'a': [],
            'r': [],
            'r_tot': [],
            'r_tot_discount': [],
            'loss': [],
            }
        
        # Creates a book that tracks results from previous episode sets
        # @entry best_s - best episode state history
        # @entry best_a - best episode action history
        # @entry best_r - best episode reward history
        # @entry r_tot_avg - the sum of all rewards given during an epsidoe set averaged over all agents
        # @entry r_tot_discount_avg - the sum of all the discounted rewards given during an epsiode set averaged over all agents
        # @entry loss_avg - the average loss evolution of all agents
        # @entry num_actions - integer size of the action space
        # @entry state_dimension - defines the dimension of the state space
        # @entry max_data_set_size - The maximum number of experiences that can be stored
        # @entry start_data_set_size - The number of frames a uniform random policy is run to populate the data set before learning begins
        # @entry sequence_size - The number of previous states that define an input sequence
        # @entry minibatch_size - The number of experiences sampled from the data set during Q improvement
        # @entry num_hidden_layers - total number of hidden layers in NN (total number of layers - 1)
        # @entry num_neurons_in_layer - number of neurons in each hidden layer
        # @entry target_reset_interval -  the number of quality function improvement steps taken before the current Q is cloned to the target Q
        # @entry alpha - learning rate
        # @entry gamma - discount ratio
        # @entry epsilon - the terminal elpsilon-greedy action choice parameter
        # @entry epsilon_depreciation_factor - the rate at which the epsilon parameter decreases from maximum to minimum 
        self.logbook = {
            'best_s': [],
            'best_a': [],
            'best_r': [],
            'best_Q': self.Q_Network,
            'r_tot_avg': 0,
            'r_tot_discount_avg': 0,
            'loss_avg': 0,
            'num_actions': [],
            'state_dimension': [],
            'max_data_set_size': [],
            'start_data_set_size': [],
            'sequence_size': [],
            'minibatch_size': [],
            'num_hidden_layers': [],
            'num_neurons_in_layer': [],
            'target_reset_interval': [],
            'alpha': [],
            'gamma': [],
            'epsilon': [],
            'epsilon_depreciation_factor': [],
            }
    
    
    # Adds a state to the state sequence
    # @param s - the state to be appended to the state sequence
    def add_state_to_sequence(self, s):
        
        # Append the state
        self.state_sequence = np.append(self.state_sequence, s)
        
        # If, after apending, the state sequence is too long, remove the first element(s). Mark the sequence as full
        while len(self.state_sequence) > self.sequence_size * self.state_dimension:
            self.state_sequence = np.delete(self.state_sequence, 0)
            self.state_sequence_is_full = True
    
    # Ask for an action based on the current quality fnc, epsilon-greedy parameters, and the current state sequence
    # @param s - The current state
    # @return - The calculated epsilon-greedy action
    def get_action(self, s):
        
        # Choose a random variable to determine action
        choice = random.random()
        
        # If the choice variable is smaller than epsilon, or if the state sequence is too small to feed forward, 
        # or if the data set is not at the start size, take a random action in the action space
        if (choice <= self.epsilon) or (len(self.state_sequence) < (self.sequence_size * self.state_dimension)) or not(self.data_set_at_start_size):
            a = random.randint(0, self.num_actions - 1)
            
        # Otherwise, choose the best action based on the current Q function
        else:
            a = int(torch.argmax(self.Q_Network.forward(torch.Tensor(self.state_sequence))))
            
        # Update the current value of epsilon iff the data filling sequence is complete
        if self.data_set_at_start_size and (self.epsilon > self.epsilon_end):
            self.epsilon = self.epsilon * self.epsilon_depreciation_factor
            
        return a
    
    # Adds an experince (s1, a1, r, s2) to the data set. Updates state sequence
    # @parma s1 - initial state
    # @param a1 - initial action
    # @param r - reward given
    # @param s2 - final state
    def add_experience_to_data_set(self, s1, a1, r, s2):

        # Add newest state the to state sequence
        old_sequence = self.state_sequence
        self.add_state_to_sequence(s2)

        # only add to the memory if the state sequence was full
        if (len(old_sequence) == self.state_dimension * self.sequence_size):

            # Add experience to data set
            self.state_sequence_memory.append(old_sequence)
            self.action_memory.append(a1)
            self.reward_memory.append(r)
            self.next_state_sequence_memory.append(self.state_sequence)
            
            # Update the filling status
            if not(self.data_set_at_start_size) and (len(self.state_sequence_memory) >= self.start_data_set_size):
                self.data_set_at_start_size = True
            
            # If required, trim the data set
            while len(self.state_sequence_memory) > self.max_data_set_size:
                self.state_sequence_memory = self.state_sequence_memory[1:]
                self.action_memory = self.action_memory[1:]
                self.reward_memory =  self.reward_memory[1:]
                self.next_state_sequence_memory = self.next_state_sequence_memory[1:]
            
        # Update the log if learning has started
        if self.data_set_at_start_size:
            self.r_tot += r
            self.r_tot_discount += (self.gamma ** (self.curr_step)) *  r
            self.log['s'].append(s1)
            self.log['a'].append(a1)
            self.log['r'].append(r)
            self.log['r_tot'].append(self.r_tot)
            self.log['r_tot_discount'].append(self.r_tot_discount)
            self.curr_step = self.curr_step + 1
    
    # Perform gradient descent step on (target - Q(state_sequence, action; theta))**2 with respect to the network parameters, theta
    def learn(self):

        # Learn iff the state sequence buffer is full and the data_set is at the start size
        if self.state_sequence_is_full and self.data_set_at_start_size:
            
            # Get minibatch
            size = min(len(self.state_sequence_memory), self.minibatch_size)
            sample = np.random.randint(len(self.state_sequence_memory), size=size)
            state_sequence_memory_minibatch = (itemgetter(*sample)(self.state_sequence_memory))
            action_memory_minibatch = (itemgetter(*sample)(self.action_memory))
            reward_memory_minibatch = (itemgetter(*sample)(self.reward_memory))
            next_state_sequence_memory_minibatch = (itemgetter(*sample)(self.next_state_sequence_memory))
            
            # Get target
            target = torch.Tensor([0]*size)
            if size > 1:
                r = torch.Tensor(reward_memory_minibatch)
                s2 = torch.Tensor(next_state_sequence_memory_minibatch)
                Q_hat = torch.max(self.Q_Target_Network.forward(s2).detach(), dim=1).values
            else:
                r = torch.Tensor([reward_memory_minibatch])
                if self.state_dimension == 1:
                    s2 = torch.Tensor([next_state_sequence_memory_minibatch])
                else:    
                    s2 = torch.Tensor(next_state_sequence_memory_minibatch)
                Q_hat = torch.max(self.Q_Target_Network.forward(s2).detach())
            target = r + (self.gamma * Q_hat)
            
            # Get the evaluation
            evaluation = torch.Tensor([0]*size)
            if size > 1:
                s1 = torch.Tensor(state_sequence_memory_minibatch)
                Qj = self.Q_Network.forward(s1)
                for j in range(size):
                    evaluation[j] = Qj[j][action_memory_minibatch[j]]
            else:
                if self.state_dimension == 1:
                    s1 = torch.Tensor([state_sequence_memory_minibatch])
                else:    
                    s1 = torch.Tensor(state_sequence_memory_minibatch)
                Qj = self.Q_Network.forward(s1)
                evaluation[0] = Qj[action_memory_minibatch]
            
            # Grad-descent
            loss = self.loss_fn(target, evaluation)
            loss = torch.clamp(loss, -1.0, 1.0)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.log['loss'].append(loss.item())
        
            # Count how many iterations it has been since the target network has been reset
            self.steps_since_reset += 1
            if self.steps_since_reset == self.target_reset_interval:
                self.Q_Target_Network = copy.deepcopy(self.Q_Network)   
                self.steps_since_reset = 0
    
    # Updates the best trajectory at the end of each episode
    def end_episode(self):
        # Update the best trajectory log if learning has started
        if self.data_set_at_start_size:
            change_in_r_tot = self.log['r_tot'][-1] - self.r_tot_prev
            if change_in_r_tot > self.highest_r_tot and len(self.log['s']) > 0:
                self.highest_r_tot = change_in_r_tot
                self.logbook['best_s'] = self.log['s']
                self.logbook['best_a'] = self.log['a']
                self.logbook['best_r'] = self.log['r']
            self.r_tot_prev = self.log['r_tot'][-1]
        self.log['s'] = []
        self.log['a'] = []
        self.log['r'] = []
   
    # Archives the log from the current episode to the log book. Resets agent to initial conditions
    # @param alpha - learning rate
    # @param gamma - discount ratio
    # @param epsilon - elpsilon-greedy action choice parameter
    def terminate_agent(self):
        
        # Update the average learning curves
        n = self.entry_number
        self.entry_number += 1
        r_tot_avg = (1/n) * ((n - 1) * self.logbook['r_tot_avg'] + np.asarray(self.log['r_tot']))
        r_tot_discount_avg = (1/n) * ((n - 1) * self.logbook['r_tot_discount_avg'] + np.asarray(self.log['r_tot_discount']))
        loss_avg = (1/n) * ((n - 1) * self.logbook['loss_avg'] + np.asarray(self.log['loss']))
        
        # In the case of the first set, the average value is just the current value
        if n==1:
            r_tot_avg = self.log['r_tot']
            r_tot_discount_avg = self.log['r_tot_discount'] 
            loss_avg = self.log['loss']
            
        # Input the average function values
        self.logbook['r_tot_avg'] = r_tot_avg
        self.logbook['r_tot_discount_avg'] = r_tot_discount_avg  
        self.logbook['loss_avg'] = loss_avg
        
        # Save the best Q network
        if self.data_set_at_start_size and (self.log['r_tot_discount'][-1] > self.highest_r_tot_discounted):
            self.logbook['best_Q'] = copy.deepcopy(self.Q_Network)
        
        # Input the agent parameters
        self.logbook['num_actions'].append(self.num_actions)
        self.logbook['state_dimension'].append(self.state_dimension)
        self.logbook['max_data_set_size'].append(self.max_data_set_size)
        self.logbook['start_data_set_size'].append(self.start_data_set_size)
        self.logbook['sequence_size'].append(self.sequence_size)
        self.logbook['minibatch_size'].append(self.minibatch_size)
        self.logbook['num_hidden_layers'].append(self.num_hidden_layers)
        self.logbook['num_neurons_in_layer'].append(self.num_neurons_in_layer)
        self.logbook['target_reset_interval'].append(self.target_reset_interval)
        self.logbook['alpha'].append(self.alpha)
        self.logbook['gamma'].append(self.gamma)
        self.logbook['epsilon'].append(self.epsilon)
        self.logbook['epsilon_depreciation_factor'].append(self.epsilon_depreciation_factor)
        
        # Reset epsilon
        self.epsilon = self.epsilon_start
        
        # Reset the NN used to estimate Q(s,a) and the optimizer used for gradient descent
        self.Q_Network = NN.Neural_Network(self.sequence_size * self.state_dimension, self.num_actions, self.num_hidden_layers, self.num_neurons_in_layer)
        self.opt = torch.optim.Adam(self.Q_Network.parameters() , lr=self.alpha)
        self.loss_fn = torch.nn.MSELoss()
            
        # Initialize the NN used to provide the target for Q improvement
        self.Q_Target_Network = copy.deepcopy(self.Q_Network)
        self.steps_since_reset = 0
        
        # Initialize the sequence of states used as the input to the Q_Network (data structure : QUEUE)
        self.state_sequence = np.asarray([])
        self.state_sequence_is_full = False
        
        # Initializes data_set
        self.state_sequence_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.next_state_sequence_memory = []
        self.data_set_at_start_size = False
        
        # Defines the variables used in the log
        self.r_tot = 0.0
        self.r_tot_discount = 0.0
        self.curr_step = 0
        
        # Reset the log that tracks learning during a single episode set
        self.log = {
            's': [],
            'a': [],
            'r': [],
            'r_tot': [],
            'r_tot_discount': [],
            'loss': [],
            }