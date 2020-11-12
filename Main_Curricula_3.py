import Vortex_Panel_Solver_Sum_of_Forces as vps 
import DQN_Agent as dqn
import time
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Defines what a set of episodes is
def run_set(curr_set, n_sets, n_episodes, n_draw, env, agent, target_avg_reward=0.80, max_episodes=5000):
            
    # Start Training
    print("Training DQN agent " + str(curr_set+1) + " of " + str(n_sets) + "...")  
      
    # Train agent over n_episodes of episodes
    exit_cond = 0
    total_steps = 0
    percent_complete = 0.0
    total_reward = 0.0
    episode_reward = 0.0
    curr_episode = 0
    running_reward = [0.0]*10
    reward_history = []
    running_average = [0.0]*100
    best_airfoil_reward = -100.0
    best_surface_y_sequence = []
    surface_x_sequence = []
    surface_y_sequence = []
    n_sequence = []
    end_episode = False
    while True:
        
        # Initialize simulation
        s1 = env.reset()
        agent.add_state_to_sequence(s1)
                
        # Display parameters
        running_reward.append(episode_reward)
        running_reward.pop(0)
        reward_history.append((sum(running_average)/len(running_average)))
        running_average.append(episode_reward / env.max_num_steps)
        running_average.pop(0)
        episode_reward = total_reward
        
        # Termination conditions
        n = 0
        if (curr_episode >= max_episodes):
            exit_cond = 1
            end_episode = True
        if ((curr_episode >= n_episodes) and (total_reward > 0.0 and (sum(running_reward)/len(running_reward)) < 0.0)):
            exit_cond = 2
            end_episode = True
        if ((sum(running_average)/len(running_average)) >= target_avg_reward) and (curr_episode >= n_episodes):
            exit_cond = 3
            end_episode = True
        
        # User readout
        print_str = (('{:03.2f}'.format(100.0 * percent_complete) + "% Complete").ljust(21) + 
            ("| Foil: " + str(curr_episode) + "/" + str(n_episodes)).ljust(18) + 
            ("| Tot R: " + '{:.0f}'.format(total_reward)).ljust(17) + 
            ("| R/step: " + '{:.2f}'.format(sum(running_average)/len(running_average)) + "/" + '{:.2f}'.format(target_avg_reward)).ljust(22) + 
            ("| Foil R: " + ('{:.2f}'.format(running_reward[-1]))).ljust(19) + 
            ("| R/foil: " + '{:.2f}'.format(sum(running_reward)/len(running_reward))).ljust(20) + 
            ("| Best foil: " + '{:.0f}'.format(best_airfoil_reward)).ljust(18) + 
            ("| Epsilon: " + '{:.2f}'.format(agent.epsilon)).ljust(17) + 
            "|")
        print(print_str, end="\r", flush=True)
        
        # Simulate until episode is done
        done = False
        n = 0
        while not done:
            
            # Save the best surfaces
            if n % n_draw == 0 or n==env.max_num_steps-1:
                if curr_episode == 0:
                    surface_x_sequence.append(env.surface_x)
                    n_sequence.append(n)
                surface_y_sequence.append(env.surface_y)
            n = n + 1 
            
            # With probability e select a random action a1, otherwise select a1 = argmax_a Q(s1, a; theta)
            a1 = agent.get_action(s1)
            
            # Execute action a1 in emulator and observer reward r and next state s2
            (s2, r, done) = env.step(a1, n=n, reward_depreciation=1.5, path="curricula_3/")
            total_reward += r
            
            # Update state sequence buffer, store experience in data_set
            agent.add_experience_to_data_set(s1, a1, r, s2)
            
            # Sample a random minibatch from the data_set, define targets, perform gradient descent, and ocasionally update Q_Target_Network
            agent.learn()

            # Update state and action
            s1 = s2
            
            # Calculate completedness
            percent_complete = (total_steps) / (n_episodes * env.max_num_steps)
            total_steps += 1
            
        # After an episode calculate the reward and whether or not to keep the save airfoil sequence
        episode_reward = total_reward - episode_reward
        if episode_reward >= best_airfoil_reward:
            best_airfoil_reward = episode_reward
            best_surface_y_sequence = surface_y_sequence
        surface_y_sequence = []
        
        # After an episode, update the agent's logs
        agent.end_episode()
        
        # Termination check
        if end_episode:
            break
        else:
            curr_episode = curr_episode + 1

    # Onces an episode set is complete, update the logbook, terminate the current log
    agent.terminate_agent(keep_NN=True)
    
    # Once an episode set is complete, draw the best cp distribution and generate results
    env.surface_x = surface_x_sequence[-1]
    env.surface_y = best_surface_y_sequence[-1]
    x_upper = env.surface_x[0:env.n_panels_per_surface+1]
    x_lower = env.surface_x[env.n_panels_per_surface:env.n_panels_per_surface+env.n_panels_per_surface+1]
    y_upper = env.surface_y[0:env.n_panels_per_surface+1]
    y_lower = env.surface_y[env.n_panels_per_surface:env.n_panels_per_surface+env.n_panels_per_surface+1]
    env.surface_normal = np.append(env.get_normal(x_upper, y_upper), env.get_normal(x_lower, y_lower), axis=1)
    env.visualize_cp_save_performance(path="curricula_3/")
    
    # Visualize the best airfoil sequence
    env.visualize_airfoil_sequence(surface_x_sequence, best_surface_y_sequence, n_sequence, path="curricula_3/")
    
    # Print the final training results
    print_str = (("100.00% Complete").ljust(21) + 
        ("| Foil: " + str(curr_episode) + "/" + str(n_episodes)).ljust(18) + 
        ("| Tot R: " + '{:.0f}'.format(total_reward)).ljust(17) + 
        ("| R/step: " + '{:.2f}'.format(sum(running_average)/len(running_average)) + "/" + '{:.2f}'.format(target_avg_reward)).ljust(22) + 
        ("| Foil R: " + ('{:.2f}'.format(running_reward[-1]))).ljust(19) + 
        ("| R/foil: " + '{:.2f}'.format(sum(running_reward)/len(running_reward))).ljust(20) + 
        ("| Best foil: " + '{:.0f}'.format(best_airfoil_reward)).ljust(18) + 
        ("| Epsilon: " + '{:.2f}'.format(agent.epsilon)).ljust(17) + 
        "|")
    print(print_str)
    if exit_cond==1:
        print("EXIT: Maximum number of episodes reached.")
    elif exit_cond==2:
        print("EXIT: Negative learning detected.")
    elif exit_cond==3:
        print("EXIT: Target episodes and average reached.")
    return agent, reward_history


if __name__ == '__main__':

    # Target design parameters
    re_test = 1000000
    alpha_test_points = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0]) * np.pi / 180.0
    cl_test_points = np.array([0.2442, 0.4549, 0.7153, 0.9016, 1.0885, 1.2696])
    cdp_test_points = np.array([0.00122, 0.00168, 0.00236, 0.00381, 0.00642, 0.00970])
    cm4c_test_points = np.array([-0.0525, -0.0482, -0.0566, -0.0497, -0.0440, -0.0378])

    # Simulation parameters
    n_panel_per_surface = 10
    target_avg_reward = 1.50
    n_sets = 1
    n_episodes = 2000
    max_episodes = 2500
    n_steps = 25 * (2*n_panel_per_surface + 1)
    n_draw = n_steps // 19
    
    # Environment
    env = vps.Vortex_Panel_Solver(n_steps, n_panel_per_surface, re_test, alpha_test_points, cl_test_points, 
                                  cdp_test_points, cm4c_test_points)
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
    gamma = 0.99
    epsilon_start = 0.50
    epsilon_end = 0.10
    percent_at_epsilon_complete = 0.75
    epsilon_depreciation_factor = (epsilon_start - epsilon_end) / (percent_at_epsilon_complete * n_episodes * n_steps)
    
    # Create agent
    agent = dqn.DQN_Agent(num_actions, state_dimension, max_data_set_size, start_data_set_size, sequence_size, 
                          minibatch_size, num_hidden_layers, num_neurons_in_layer, clone_interval, 
                          alpha, gamma, epsilon_start, epsilon_end, epsilon_depreciation_factor)
    
    # Load previous agent network
    with open("curricula_2/results/outputs", "rb") as f:
        outputs = pickle.load(f)
    agent.copy_agent(outputs['last_agent'].Q_Network)
    
    # Run the defined number of sets and update the average
    start = time.time()
    for curr_set in range(n_sets):
        # Run a set of episodes
        agent, reward_history = run_set(curr_set, n_sets, n_episodes, n_draw, env, agent, target_avg_reward=target_avg_reward, max_episodes=max_episodes)
    
    elapsed = time.time() - start
    print("Simulation took:", f'{elapsed:.3f}', "seconds.")
    
    # plot results
    print("Plotting training and trajectory data...")
    start = time.time()
  
    # Save all useful variables
    outputs = {
        'n_sets' : n_sets, 
        'n_episodes' : n_episodes, 
        'depreciation' : 1.0,
        'env' : env, 
        'last_agent' : agent,
        }
    with open('curricula_3/results/outputs', 'wb') as f:
        pickle.dump(outputs, f)  
  
    # Calculate locations of cloning and terminal epsilon
    last_data_point = len(agent.logbook['r_tot_discount_avg'])-1
    num_times_cloned = int(float(last_data_point) // float(clone_interval))
    cloning_points = np.linspace(clone_interval, last_data_point, num_times_cloned)
    final_exploration_frame = percent_at_epsilon_complete * n_episodes * n_steps
  
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
    plt.savefig('curricula_3/results/rwd_cur.png', dpi = 200)
    
    # plot learning curve
    plt.clf()
    title_str = "DQN Learning Curve: \u03B1 = " + str(alpha) + ", \u03B3 = " + str(gamma) + ", \u03B5 = " + str(epsilon_start) + " → " + str(epsilon_end)
    plt.title(title_str)
    plt.plot([*range(len(reward_history))], reward_history)
    plt.xlabel('Episode')
    plt.ylabel('Reward Per Step')
    plt.savefig('curricula_3/results/learn_cur.png', dpi = 200)
    
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
    plt.savefig('curricula_3/results/los_cur.png', dpi = 200)
    plt.close()
    
    elapsed = time.time() - start
    print("Plotting took:", f'{elapsed:.3f}', "seconds.")