import numpy as np
import heapq
import os 
import time
from termcolor import colored

class GridEnvironment:
    def __init__(self):
        self.size = 6
        self.reset()

    def reset(self):
        self.agent_pos = [0, 0]
        self.resource_pos = np.random.choice(36, 15, replace=False)
        self.grid = np.zeros((self.size, self.size))
        np.put(self.grid, self.resource_pos, 1)
        self.resource_count = 0
        self.rewards = 0
        self.steps = 0
        return self.get_state()

    def get_state(self):
        state = {'grid': self.grid, 'agent_pos': self.agent_pos, 'resource_count': self.resource_count}
        return state

    def step(self, action):
        if action == 'up':
            self.agent_pos[0] = max(0, self.agent_pos[0]-1)
            self.rewards -= 1
        elif action == 'down':
            self.agent_pos[0] = min(self.size-1, self.agent_pos[0]+1)
            self.rewards -= 1
        elif action == 'left':
            self.agent_pos[1] = max(0, self.agent_pos[1]-1)
            self.rewards -= 1
        elif action == 'right':
            self.agent_pos[1] = min(self.size-1, self.agent_pos[1]+1)
            self.rewards -= 1
        elif action == 'collect':
            if self.grid[self.agent_pos[0], self.agent_pos[1]] == 1:
                self.grid[self.agent_pos[0], self.agent_pos[1]] = 0
                self.resource_count += 1
                self.rewards += 16
        elif action == 'build':
            if self.grid[self.agent_pos[0], self.agent_pos[1]] == 0 and self.resource_count >= 3:
                self.grid[self.agent_pos[0], self.agent_pos[1]] = 2
                self.resource_count -= 3
                self.rewards += 60

        self.steps += 1
        done = self.is_episode_done()
        return self.get_state(), done

    def is_episode_done(self):
        # End the episode after a maximum number of steps
        if self.steps >= self.size**2 * 5 or (np.sum(self.grid == 1) == 0 and self.resource_count == 0):
            return True

    

    def a_star(self, start, goal, grid):
        def heuristic(a, b):
            return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2

        neighbors = [(0,1), (0,-1), (1,0), (-1,0)]
        close_set = set()
        came_from = {}
        gscore = {start: 0}
        fscore = {start: heuristic(start, goal)}
        oheap = []

        heapq.heappush(oheap, (fscore[start], start))

        while oheap:
            current = heapq.heappop(oheap)[1]

            if current == goal:
                data = []
                while current in came_from:
                    data.append(current)
                    current = came_from[current]
                return data

            close_set.add(current)
            for i, j in neighbors:
                neighbor = current[0] + i, current[1] + j
                tentative_g_score = gscore[current] + heuristic(current, neighbor)
                if 0 <= neighbor[0] < grid.shape[0]:
                    if 0 <= neighbor[1] < grid.shape[1]:
                        if grid[neighbor[0]][neighbor[1]] == 2:
                            continue
                    else:
                        continue
                else:
                    continue

                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                    continue

                if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))


def print_env(state):
    grid = np.copy(state['grid'])  
    grid[tuple(state['agent_pos'])] = 9  

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == 1:
                print(colored("1", 'yellow'), end=' ')  
            elif grid[i, j] == 9:
                print(colored("A", 'red'), end=' ')  
            elif grid[i, j] == 2:
                print(colored("2", 'blue'), end=' ')  
            else:
                print(colored("0", 'grey'), end=' ')
        print()
    print(f"{action}")
    print(f"{env.rewards}")
    print()

class SimpleAgent:
    def __init__(self, env):
        self.env = env

    def act(self, state):
        # Check if agent is standing on a resource and collect it
        if state['grid'][state['agent_pos'][0]][state['agent_pos'][1]] == 1:
            return 'collect'

        # Check if agent has at least 3 resources and is standing on an empty square
        if state['resource_count'] >= 3 and state['grid'][state['agent_pos'][0]][state['agent_pos'][1]] == 0:
            return 'build'

        # Determine nearest resource
        resource_pos = np.argwhere(state['grid'] == 1)
        if len(resource_pos) > 0:
            distances = np.linalg.norm(resource_pos - state['agent_pos'], axis=1)
            nearest_resource = resource_pos[np.argmin(distances)].tolist()
            path = self.env.a_star(tuple(state['agent_pos']), tuple(nearest_resource), state['grid'])
            if path is not None and len(path) > 0:
                path = list(reversed(path)) # Reverse the path
                next_pos = path[1] if len(path) > 1 else path[0] # Get the next position
                return 'right' if next_pos[1] > state['agent_pos'][1] else 'left' if next_pos[1] < state['agent_pos'][1] else 'down' if next_pos[0] > state['agent_pos'][0] else 'up'

        # Determine nearest empty square if no resources left
        empty_pos = np.argwhere(state['grid'] == 0)
        if len(empty_pos) > 0:
            distances = np.linalg.norm(empty_pos - state['agent_pos'], axis=1)
            nearest_empty = empty_pos[np.argmin(distances)].tolist()
            path = self.env.a_star(tuple(state['agent_pos']), tuple(nearest_empty), state['grid'])
            if path is not None and len(path) > 0:
                path = list(reversed(path)) # Reverse the path
                next_pos = path[1] if len(path) > 1 else path[0] # Get the next position
                return 'right' if next_pos[1] > state['agent_pos'][1] else 'left' if next_pos[1] < state['agent_pos'][1] else 'down' if next_pos[0] > state['agent_pos'][0] else 'up'

        # No available moves
        return None




if __name__ == "__main__":
    env = GridEnvironment()
    agent = SimpleAgent(env)

    for episode_idx in range(1):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            if action is None:
                break
            state, done = env.step(action)
            print_env(state)
