# Collect_Gather_Build Simulation
We build a Actor2Critic model built using Python and Pytorch, where an agent learns to navigate a 6x6 grid with randomly placed resources, collect those resources, and build houses with them - learning to maximise their reward. We run this for multiple episodes, within which we have each ste pof either moving, collecting a resource, or building a huose. Each episode ends either when the agent has exhausted all of its moves (to speed up training and reduce overfitting on one grid configuration, even if unlikely), or when the agent has collected all resources, and built all the houses it can with these resources.

We then compare these to a more simple and straight forward simulation approach of the same problem. In this method, we tell the agent how to act instead of having it learn. This leads to immediately optimal results... as long as we know what the optimial result(s) are when writing them sim!



## The images below show the final results using the Reinforcement Learning approach:
Agents collect experiences during each step within the episode, and optimise their network at the end of each episode, given what they have learned. Run (training) time: approx. 25 minutes.

Grid environment during starting episodes:

<img width="380" alt="Screenshot 2023-07-15 at 15 36 47" src="https://github.com/lblcbc/ReinforcemenetLearning_Simulation/assets/136857271/ebe273f9-f797-4b6c-a994-2783797c1cf0">


Grid environment during the end of training:

<img width="389" alt="Screenshot 2023-07-15 at 16 01 40" src="https://github.com/lblcbc/ReinforcemenetLearning_Simulation/assets/136857271/5bd34bdf-1ff2-4691-b553-15e18dbf0096">


Plotted progress:

<img width="436" alt="Screenshot 2023-07-15 at 16 01 51" src="https://github.com/lblcbc/ReinforcemenetLearning_Simulation/assets/136857271/c7a1de67-5f33-4ea6-aa1d-2e72a82a87b3">


## The images below show the final results using the Reinforcement Learning approach:
Here the agent is programmed to behave with the same set of possible actions in the same grid. However, we use a simple A* search algorithm to find and move towards the nearest resource, collect it once reached, and build a house as soon as the agent has the required resource and is standing on an empty grid slot. We don't need to run multiple episodes on multiple grid configurations, as the agent has nothing to learn. Run time: < 1 second. 

Grid environment during starting steps:
<img width="116" alt="Screenshot 2023-07-30 at 17 18 11" src="https://github.com/lblcbc/Collect_Gather_Build/assets/136857271/8ba257b4-2ef3-413f-a27a-d1609a93517d">

Grid environment at the end of the one-and-only episode:
<img width="96" alt="Screenshot 2023-07-30 at 17 08 34" src="https://github.com/lblcbc/Collect_Gather_Build/assets/136857271/ba7e12cf-b223-41dc-a02c-47fa5cdfdf16">

We see the reward achieved is close to the maximum, but lower. Normally I would investigate this, by plotting the moves and rewards progression of the simulation and comparing it against a highly-trained agent, on the same map. However, in this case, I know where the RL agent is gaining; the simulation agent is only programmed with a simple A* algorithm that looks for the nearest resource, without taking into account the most efficient path of collecting ALL resources. This could of course be changed, but this exercise was more to compare the performance differences. It is clear simulations are much superior (1 sec run time vs 20+ highly-sensitive minutes) in the cases where we know the right solution, or at least the range of possible solutions, though it can be overconfident to assume we often do! Programming the RL agent took some tuning – with the main change being increasing the learning rate, and not scheduling it to decrease so quickly –, though I was positively surprised by how quickly the agent started to learn and perform optimally. Overall an interesting experience - unfortunately with the average hardware extending the RL agent leads to unpractical run times (a 10x10 grid of the same problem took 6+ HOURS to train). 



