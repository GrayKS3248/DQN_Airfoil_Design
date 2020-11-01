import Vortex_Panel_Solver as vps 
import DQN_Agent as dqn
import time
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle

# Defines what a set of episodes is
def run_set(curr_set, n_sets, n_episodes, n_draw, env, agent):
            
    # Start Training
    print("Training DQN agent " + str(curr_set) + " of " + str(n_sets-1) + "...")  
      
    # Train agent over n_episodes of episodes
    total_steps = 0
    percent_complete = 0.0
    total_reward = 0.0
    for curr_episode in range(n_episodes):
        
        # Initialize simulation
        s1 = env.reset()
        agent.add_state_to_sequence(s1)
        
        # Visualization parameters
        if (curr_episode == n_episodes - 1):
            env.visualize_airfoil(0, path="curricula_2/")
        n = 1
                
        # Simulate until episode is done
        print_str = '{:03.2f}'.format(100.0 * percent_complete) + "% Complete... | Total Reward: " + '{:.0f}'.format(total_reward) + " | Average Reward: " + '{:03.2f}'.format(total_reward/(total_steps+1))
        print(print_str, end="".join(['\b']*len(2*print_str))+"\r", flush=True)
        done = False
        while not done:
            
            # With probability e select a random action a1, otherwise select a1 = argmax_a Q(s1, a; theta)
            a1 = agent.get_action(s1)
            
            # Determine wether to draw foil or not
            vis_foil = (n % n_draw == 0) and (curr_episode == n_episodes - 1)
            n = n + 1 
            
            # Calculate reward depreciation
            percent_complete = (total_steps) / (n_episodes * env.max_num_steps)
            total_steps += 1
            
            # Execute action a1 in emulator and observer reward r and next state s2
            (s2, r, done) = env.step(a1, vis_foil=vis_foil, n=n-1, reward_depreciation=1.0, path="curricula_2/")
            total_reward += r
            
            # Update state sequence buffer, store experience in data_set
            agent.add_experience_to_data_set(s1, a1, r, s2)
            
            # Sample a random minibatch from the data_set, define targets, perform gradient descent, and ocasionally update Q_Target_Network
            agent.learn()

            # Update state and action
            s1 = s2
            
        # After an episode is done, update the logbook
        agent.end_episode()

    # Onces an episode set is complete, update the logbook, terminate the current log, draw the cp dist
    agent.terminate_agent(keep_NN=True)
    env.visualize_cp_save_performance(path="curricula_2/")
    print("100.00% Complete!    |    Total Reward: " + '{:.0f}'.format(total_reward) + "    |    Average Reward: " + '{:03.2f}'.format(total_reward/(total_steps+1)))
    
    return agent


if __name__ == '__main__':

    # Target design parameters
    v_inf_test_points = np.array([14.207, 14.207, 14.207, 14.207, 14.207, 14.207])
    alpha_test_points = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0]) * np.pi / 180
    cl_test_points = np.array([0.2442, 0.4549, 0.7153, 0.9016, 1.0885, 1.2696])
    cdp_test_points = np.array([0.00122, 0.00168, 0.00236, 0.00381, 0.00642, 0.00970])
    cm4c_test_points = np.array([-0.0525, -0.0482, -0.0566, -0.0497, -0.0440, -0.0378])

    # Simulation parameters
    n_panel_per_surface = 10
    n_sets = 1
    n_episodes = 1501
    n_steps = int(24.5 * (2*n_panel_per_surface + 1)) # In this number of steps, all vertices can be moved from min to max value
    n_draw = n_steps // 19
    
    # Environment
    env = vps.Vortex_Panel_Solver(n_steps, n_panel_per_surface, v_inf_test_points, alpha_test_points,
                                  cl_test_points, cdp_test_points, cm4c_test_points)
    num_actions = env.num_actions
    state_dimension = env.state_dimension
    
    # Agent parameters
    max_data_set_size = 1000000
    start_data_set_size = 1000
    sequence_size = 1
    minibatch_size = 32
    num_hidden_layers = 2
    num_neurons_in_layer = 128
    clone_interval = 10000
    alpha = 0.00025
    gamma = 1.0
    epsilon_start = 0.75
    epsilon_end = 0.10
    epsilon_depreciation_factor = math.pow(epsilon_end,(3.0 / (n_episodes*n_steps - start_data_set_size)))
    
    # Load previous agent and reset its epsilon parameters and logbook
    with open("curricula_1/results/outputs", "rb") as f:
        outputs = pickle.load(f)
    agent = outputs['last_agent']
    agent.epsilon = epsilon_start
    agent.epsilon_start = epsilon_start
    agent.epsilon_end = epsilon_end
    agent.epsilon_depreciation_factor = epsilon_depreciation_factor
    agent.logbook = {
        'best_s': [],
        'best_a': [],
        'best_r': [],
        'best_Q': 0,
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
        'epsilon_depreciation_factor': []
    }
    
    # Run the defined number of sets and update the average
    start = time.time()
    for curr_set in range(n_sets):
        # Run a set of episodes
        agent = run_set(curr_set, n_sets, n_episodes, n_draw, env, agent)
    
    elapsed = time.time() - start
    print("Simulation took:", f'{elapsed:.3f}', "seconds.")
    
    # plot results
    print("Plotting training and trajectory data...")
    start = time.time()
  
    # Save all useful variables
    outputs = {
        'n_sets' : n_sets, 
        'n_episodes' : n_episodes, 
        'depreciation' : 0.75,
        'env' : env, 
        'last_agent' : agent,
        }
    with open('curricula_2/results/outputs', 'wb') as f:
        pickle.dump(outputs, f)  
  
    # Calculate locations of cloning and terminal epsilon
    last_data_point = len(agent.logbook['r_tot_discount_avg'])-1
    num_times_cloned = int(float(last_data_point) // float(clone_interval))
    cloning_points = np.linspace(clone_interval, last_data_point, num_times_cloned)
    final_exploration_frame = math.log(epsilon_end) // math.log(epsilon_depreciation_factor)  
  
    # Plot reward curve    
    plt.clf()
    title_str = "DQN Reward Curve: \u03B1 = " + str(alpha) + ", \u03B3 = " + str(gamma) + ", \u03B5 = " + str(epsilon_start) + " → " + str(epsilon_end)
    plt.title(title_str)
    plt.plot([*range(len(agent.logbook['r_tot_avg']))], agent.logbook['r_tot_avg'], label="Total Reward")
    for cloning_point in cloning_points:
        if cloning_point in [clone_interval]:
            plt.axvline(cloning_point, c='r', linestyle=':', linewidth=0.5, label="Clone")
        else:
            plt.axvline(cloning_point, c='r', linestyle=':', linewidth=0.5, label=None)
    plt.axvline(final_exploration_frame, c='k', linestyle=':', linewidth=2, label="ε Stable")
    plt.legend()
    plt.xlabel('Simulation Step')
    plt.ylabel('Total Reward')
    plt.savefig('curricula_2/results/rwd_cur.png', dpi = 200)
    
    # plot loss curve
    plt.clf()
    title_str = "DQN Loss Curve: \u03B1 = " + str(alpha) + ", \u03B3 = " + str(gamma) + ", \u03B5 = " + str(epsilon_start) + " → " + str(epsilon_end)
    plt.title(title_str)
    plt.plot([*range(len(agent.logbook['loss_avg']))], agent.logbook['loss_avg'], label="Loss")
    for cloning_point in cloning_points:
        if cloning_point in [clone_interval]:
            plt.axvline(cloning_point, c='r', linestyle=':', linewidth=0.5, label="Clone")
        else:
            plt.axvline(cloning_point, c='r', linestyle=':', linewidth=0.5, label=None)
    plt.axvline(final_exploration_frame, c='k', linestyle=':', linewidth=2, label="ε Stable")
    plt.legend()
    plt.xlabel('Simulation Step')
    plt.ylabel('Loss')
    plt.savefig('curricula_2/results/los_cur.png', dpi = 200)
    plt.close()
    
    elapsed = time.time() - start
    print("Plotting took:", f'{elapsed:.3f}', "seconds.")