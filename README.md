# Collect_Gather_Build
In this project we build an Actor2Critic model built using Python and Pytorch, where an agent learns to navigate a 6x6 grid with randomly placed resources, collect those resources, and build houses with them - learning to maximise their reward. We run this for multiple episodes, within which we have each step of either moving, collecting a resource, or building a house. Each episode ends either when the agent has exhausted all of its moves (to speed up training and reduce overfitting on one grid configuration, even if unlikely), or when the agent has collected all resources and built all the houses it can with these resources.

We later compare these to a more simple and straightforward simulation approach of the same problem. In this method, we tell the agent how to act instead of having it learn. This leads to immediately optimal results... as long as we know what the optimal result(s) are when writing the sim!



## The images below show the final results using the Reinforcement Learning approach:
Agents collect experiences during each step within the episode, and optimise their network at the end of each episode, given what they have learned. Run (training) time: ~25 minutes (30,000 episodes).

Grid environment during starting episodes (each episode increment in the figure represents 200 episodes, for legibility):

<img width="380" alt="Screenshot 2023-07-15 at 15 36 47" src="https://github.com/lblcbc/ReinforcemenetLearning_Simulation/assets/136857271/ebe273f9-f797-4b6c-a994-2783797c1cf0">


Grid environment during the end of training:

<img width="389" alt="Screenshot 2023-07-15 at 16 01 40" src="https://github.com/lblcbc/ReinforcemenetLearning_Simulation/assets/136857271/5bd34bdf-1ff2-4691-b553-15e18dbf0096">

The agent has built all 5 possible houses in as few movement steps as it can; as per our termination condition, building the last possible house ends the episode, which is why we don't see the 5th house, the agent is standing on it :).

Plotted progress:

<img width="436" alt="Screenshot 2023-07-15 at 16 01 51" src="https://github.com/lblcbc/ReinforcemenetLearning_Simulation/assets/136857271/c7a1de67-5f33-4ea6-aa1d-2e72a82a87b3">


## The images below show the final results using the Simulation approach:
Here the agent is programmed to behave with the same set of possible actions in the same grid environment. However, instead of having the agent learn how to act, we use a simple A* search algorithm to find and move towards the nearest resource, collect it once reached, and build a house as soon as the agent has the required resource and is standing on an empty grid slot. We don't need to run multiple episodes on multiple grid configurations, as the agent has nothing to learn. Run time: <1 second. 

Grid environment during starting steps:

<img width="95" alt="Screenshot 2023-08-03 at 12 32 10" src="https://github.com/lblcbc/Collect_Gather_Build/assets/136857271/21c8a07a-816b-4b98-8f85-3879c651463e">



Grid environment at the end of the one-and-only episode:

<img width="98" alt="Screenshot 2023-08-03 at 12 33 09" src="https://github.com/lblcbc/Collect_Gather_Build/assets/136857271/fddc0923-a8fe-4bd0-937f-fb9d336c1bfa">
Following the same episode termination conditions, the agent's last action is to build the final possible 5th house, and then stays on that square, hiding the final house under itself. 

We see the reward achieved by the sim agent is immediately higher than that of the RL agent - simulation rewards ranged from roughly 512 - 519, while the RL agent would peak right around to just below 500. This is no surprise as, in this case, we knew in advance what it took for the agent to operate and act optimally. Naturally, we could further extend the simulation by improving the A* algorithm to evaluate the most efficient path for collecting ALL resources, rather than going to the nearest resource one at a time. On this small resource-dense map, results would improve only marginally, but this evolution would be important on more scattered, larger maps. To summarise, it is clear simulations are much superior (1 sec run time vs 20+ highly-sensitive minutes) in the cases where we know the right solution, or at least the range of possible solutions (though we mustn't be overconfident to assume we often do!). Programming the RL agent took some tuning – with the main change being increasing the learning rate, and not scheduling it to decrease so quickly –, though I was positively surprised by how quickly the agent started to learn and perform quite optimally. Overall an interesting experience; unfortunately with "normal" hardware, extending the RL agent further to compare performance on larger maps led/leads to unpractical run times (a 10x10 grid of the same problem, for example, took 6+ HOURS to train). 



