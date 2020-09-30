import Vortex_Panel_Solver as vps 
import DQN_Agent as dqn
import time
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import pickle

# Defines what a set of episodes is
def run_set(curr_set, n_sets, n_episodes, env, agent):
            
    # Start Training
    print("Training DQN agent " + str(curr_set) + " of " + str(n_sets-1) + "...")  
      
    # Train agent over n_episodes of episodes
    for curr_episode in range(n_episodes):
        
        # Initialize simulation
        s1 = env.reset()
        agent.add_state_to_sequence(s1)
        
        # Simulate until episode is done
        done = False
        while not done:
            
            # With probability e select a random action a1, otherwise select a1 = argmax_a Q(s1, a; theta)
            a1 = agent.get_action(s1)
            
            # Execute action a1 in emulator and observer reward r and next state s2
            (s2, r, done) = env.step(a1)
            
            # Update state sequence buffer, store experience in data_set
            agent.add_experience_to_data_set(s1, a1, r, s2)
            
            # Sample a random minibatch from the data_set, define targets, perform gradient descent, and ocasionally update Q_Target_Network
            agent.learn()

            # Update state and action
            s1 = s2
            
        # After an episode is done, update the logbook
        agent.end_episode()

    # Onces an episode set is complete, update the logbook and terminate the current log
    agent.terminate_agent()
    
    return agent


if __name__ == '__main__':
    # Environment
    env = vps.Vortex_Panel_Solver()
    num_actions = env.num_actions
    state_dimension = 2
    
    # Simulation parameters
    n_sets = 100
    n_episodes = 1005                          # Best = 505
    n_steps = env.max_num_steps
    
    # Agent parameters
    max_data_set_size = 50000                 # Best = 50000
    start_data_set_size = 500                 # Best = 500
    sequence_size = 1                         # Best = 1 (likely because we already have state derivative data)
    minibatch_size = 32                       # Best = 32
    num_hidden_layers = 2                     # Constrained = 2
    num_neurons_in_layer = 64                 # Constrained = 64
    clone_interval = 1000                     # Best = 1000
    alpha = 0.0025                            # Best = 0.0025
    gamma = 0.95                              # Constrained = 0.95
    epsilon_start = 1.00                      # Best = 1.00
    epsilon_end = 0.10                        # Best = 0.10
    epsilon_depreciation_factor = 0.99977    # Best = 0.99977
    
    # Create agent
    agent = dqn.DQN_Agent(num_actions, state_dimension, max_data_set_size, start_data_set_size, sequence_size, 
                          minibatch_size, num_hidden_layers, num_neurons_in_layer, clone_interval, 
                          alpha, gamma, epsilon_start, epsilon_end, epsilon_depreciation_factor)  
    
    # Run the defined number of sets and update the average
    start = time.time()
    for curr_set in range(n_sets):
        # Run a set of episodes
        agent = run_set(curr_set, n_sets, n_episodes, env, agent)
    
    elapsed = time.time() - start
    print("Simulation took:", f'{elapsed:.3f}', "seconds.")
    
    # plot results
    print("Plotting training and trajectory data...")
    start = time.time()
    
    elapsed = time.time() - start
    print("Plotting took:", f'{elapsed:.3f}', "seconds.")