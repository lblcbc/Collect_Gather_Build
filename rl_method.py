import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored

# Hyperparameters
grid_size = 6
max_steps = grid_size**2 * 5
action_size = 5
max_episodes = 30000
learning_rate = 1e-3
gamma = 0.99
tau = 0.95
lr_step_size = 800
lr_gamma = 0.992


class GridEnvironment:
    def __init__(self):
        self.size = 6 # 6x6 grid
        self.state_space = self.size * self.size * 3 + 2 # inputs of all resource positions, agent grid 1 and 2, and + 1 for the resources of each agent
        self.action_space = 6 # up, down, left, right, collect, build
        self.reset()

    def reset(self):
        self.agent_pos = [0, 0]
        self.resource_pos = np.random.choice(36, 15, replace=False)
        self.house_pos = []  # positions of the houses
        self.grid = np.zeros((self.size, self.size))  # resources grid
        np.put(self.grid, self.resource_pos, 1)
        self.resource_count = 0  # count of collected resources
        # TODO: Add count of houses built
        self.steps = 0
        return self.get_state()


    def get_state(self):
        agent_grid = np.zeros((self.size, self.size))
        agent_grid[self.agent_pos[0], self.agent_pos[1]] = 2
        return np.concatenate((self.grid.flatten(), agent_grid.flatten(), np.array([self.resource_count])))

    # We keep agent and env grid separate to improve learning, 
    # this way no need to worry what takes precedent if agent is standing on a resource

    def step(self, action):
        if action == 0:   # up
            self.agent_pos[0] = max(0, self.agent_pos[0]-1)
            reward = -1
        elif action == 1: # down
            self.agent_pos[0] = min(self.size-1, self.agent_pos[0]+1)
            reward = -1
        elif action == 2: # left
            self.agent_pos[1] = max(0, self.agent_pos[1]-1)
            reward = -1
        elif action == 3: # right
            self.agent_pos[1] = min(self.size-1, self.agent_pos[1]+1)
            reward = -1
        elif action == 4: # collect
            if self.grid[self.agent_pos[0], self.agent_pos[1]] == 1:
                self.grid[self.agent_pos[0], self.agent_pos[1]] = 0
                self.resource_count += 1
                reward = 16
            else:
                reward = -2 # penalty for doing something invalid # opportunity cost of sorts
        elif action == 5: # build
            if self.grid[self.agent_pos[0], self.agent_pos[1]] == 0 and self.resource_count >= 3:
                self.grid[self.agent_pos[0], self.agent_pos[1]] = 3  # add house on grid
                self.resource_count -= 3  # decrement resource count
                reward = 60 # higher than 3*16 to see if agents learn to build
            else:
                reward = -2   
                # we need to be careful that the agent doesn't try to build too early and only get's high negative rewards
                # since the condition of at least 3 resource is not placed before the action
                


        self.steps += 1
        done = (self.steps >= max_steps) or (np.sum(self.grid == 1) == 0 and self.resource_count == 0) # second condition checks if all 1s now sum to 0 (which they should since they were collected)
        return self.get_state(), reward, done



# Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        return probs, value



def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


def print_env(env):
    grid = np.copy(env.grid)  # create a copy of the grid
    grid[env.agent_pos[0], env.agent_pos[1]] = 2  # place agent on the grid

    for i in range(env.size):
        for j in range(env.size):
            if grid[i, j] == 1:
                print(colored("R", 'yellow'), end=' ')  # print 1 in yellow for resources
            elif grid[i, j] == 2:
                print(colored("A", 'red'), end=' ')  # print A in red for agent
            elif grid[i, j] == 3:
                print(colored("H", 'blue'), end=' ')  # print 3 in blue for houses
            else:
                print(colored(0, 'grey'), end=' ')
        print()
    
# Training
def train(net, env, lr, max_episodes, lr_step_size, lr_gamma):
    # Optimizer and learning rate scheduler
    optimizer = optim.Adam(net.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    episode_rewards = []  # list to store the reward for each episode
    avg_rewards = []  # list to store the average reward of the last 10 episodes

    for episode_idx in range(max_episodes):
        state = env.reset()
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0

        episode_reward = 0  # track total rewards for each episode

        for _ in range(max_steps):
            state = torch.FloatTensor(state).unsqueeze(0)
            prob, value = net(state)

            action = prob.multinomial(1)
            log_prob = prob.log().gather(1, action)

            entropy += -(log_prob * prob).sum(-1).mean()

            state, reward, done = env.step(action.item())
            episode_reward += reward  # add reward to total

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            masks.append(1 - done)

            if done:

                episode_rewards.append(episode_reward)

                if episode_idx > 1 and episode_idx % 200 == 0:
                    print_env(env)
                    avg_reward = np.mean(episode_rewards[-200:])
                    avg_rewards.append(avg_reward)
                    lr = optimizer.param_groups[0]['lr']
                    print(f'LR: {lr:.4f}, Avg Reward (200): {avg_reward:.2f}, Episode: {episode_idx/200}')
                break

        next_state = torch.FloatTensor(state).unsqueeze(0)
        _, next_value = net(next_state)
        returns = compute_gae(next_value, rewards, masks, values, gamma, tau)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    plt.plot(avg_rewards)
    plt.title('Average Rewards over the past 200 episodes')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.show()

if __name__ == "__main__":
    env = GridEnvironment()
    net = ActorCritic(input_dim=73, hidden_dim=512, output_dim=6)
    train(net, env, learning_rate, max_episodes, lr_step_size, lr_gamma)
