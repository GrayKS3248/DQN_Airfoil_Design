# Accelerated Optimal Airfoil Design by Deep Q-Learning
*Grayson Schaer - gschaer2\
AE 416 - Term Project\
10/05/2020*

## Abstract
This project was inspired by research conducted by Hajela, P. and Goel, S. into reinforcement learning based optimization of turbine airfoils [1]. In their research, they utilized an outdated form of reinforcement learning to afford “progressively smarter optimization” of an airfoil given design constraints. Since then, significant research has been conducted into airfoil morphing optimization via reinforcement learning [2] [3] [4], however little work has been conducted into airfoil shape optimization via newer reinforcement learning methods [5] [6], and none have used Deep Q Networks. This term project seeks to address this.\
A simple 2D panel method solver will be implemented in Python to solve the pressure distribution over the upper and lower surfaces of an airfoil at several user defined flight conditions. These will be converted lift, drag, and moment per unit span coefficients (Cl, Cd, and Cm,c/4) and compared to desired design values. A Deep Q Network (DQN) agent will be trained to optimize the airfoil such that differences between the calculated coefficients and desired design coefficients at all flight conditions are minimized.

## Panel Method Solver
The panel method solver will be based on the vortex panel numerical method presented in Chapter 4.9 of Fundamental of Aerodynamics by Anderson [7]. Effects not considered include compressibility and viscous effects; however future iterations of this research could use higher fidelity fluid simulators to increase design accuracy. This solver will compute the pressure distribution over the surface of an airfoil and use this to calculate the airfoils lifting, pressure drag, and moment about the quarter chord coefficients.

## Deep Q-Learning Networks
A DQN agent is any reinforcement learning agent that converts a state sampled from a continuous state space and converts it to an optimal action to maximize return via Q-function estimation with a neural network (NN) [8]. Learning is achieved via stochastic gradient descent along Bellman Optimality lines between a target network and a learning network through action replay. This learning process defines a function that converts state data to optimal action data based on the observed rewards. In the scope of this proposal, the state is defined as the point positions of the panels’ endpoints, an action is a perturbation of these positions, and the reward is proportional to how closely the airfoil performs compared to the desired performance.

## Reward
After each design and test cycle (produce airfoil based on DQN agent, test the airfoil in vortex panel method solver), a reward must be returned to the DQN agent. This reward will be high when the airfoil performs as desired and low when the airfoil does not perform as desired. The specific reward function will be a major part of this research as changing the reward function changes the definition of the problem the DQN agent is optimizing. It is likely, however, that the reward function will be in the from of a weighted average of square errors from the design requirements.

## DQN Implementation
A DQN agent has previously been implemented by the author. As such, a Python wrapper function will be developed to feed inputs from the agent to the panel method solver, and outputs from the panel method solver to the agent. The hyperparameters and training intervals will determine the rate and quality of agent learning, and thusly, in combination with the reward function, will be the focus of this research.

## References
1. S. Goel and P. Hajela, "Turbine Aerodynamic Design Using Reinforcement Learning Based Optimization," American Institute of Aeronautics and Astronautics, pp. 528-543, 1998.

2. A. Lampton, A. Niksch and J. Valasek, "Morphing Airfoils with Four Morphing Parameters," AIAA Guidance, Navigation and Control Conference and Exhibit, 2008.

3. A. Lampton, A. Niksch and J. Valasek, "Reinforcement Learning of Morphing Airfoils with Aerodynamic and Structural Effects," Journal of Aerospace Computing, Information, and Communication, vol. 6, no. 1, pp. 30-50, 2009.

4. D. Xu, Z. Hui, Y. Liu and G. Chen, "Morphing control of a new bionic morphing UAV with deep reinforcement learning," Aerospace Science and Technology, vol. 92, pp. 232-243, 2019.

5. J. Viquerat, J. Rabault, A. Kuhnle, H. Ghraieb, A. Larcher and E. Hachem, "Direct shape optimization through deep reinforcement learning," arXiv preprint, 2019.

6. J. Rabault, F. Ren, W. Zhang, H. Tang and H. Xu, "Deep reinforcement learning in fluid mechanics: A promising method for both active flow control and shape optimization," Journal of Hydrodynamics, vol. 32, no. 2, pp. 234-246, 2020.

7. J. D. Anderson, "4.9 Lifting Flows over Arbitrary Bodies: The Vortex Panel Numerical Method," in Fundamentals of Aerodynamics, 2nd ed., New York, McGraw-Hill, 1991, pp. 282-289.

8. V. Mnih, K. Kavukcuoglu, D. Silver and e. al., "Human-level control through deep reinforcement," Nature, p. 529–533, 2015.

# License
MIT License

Copyright (c) 2020 Grayson Schaer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
