import gymnasium as gym
import numpy as np

class SimpleGridWorld(gym.Env):
    def __init__(self):
        self.grid_size = 4
        self.state = 0  # Starting position
        self.goal = 15  # Bottom-right corner

        self.obstacles = [5, 10]
        
        self.action_space = gym.spaces.Discrete(4)  # Up, Right, Down, Left
        self.observation_space = gym.spaces.Discrete(16)
        
    def step(self, action):
        # Current position
        row = self.state // self.grid_size
        col = self.state % self.grid_size
        
        # Move based on action
        if action == 0:  # Up
            row = max(0, row - 1)
        elif action == 1:  # Right
            col = min(self.grid_size - 1, col + 1)
        elif action == 2:  # Down
            row = min(self.grid_size - 1, row + 1)
        elif action == 3:  # Left
            col = max(0, col - 1)
            
        self.state = row * self.grid_size + col
        
        # Reward structure
        done = self.state == self.goal
        if self.state in self.obstacles:
            reward = -1.0
        else:
            reward = 1.0 if done else -0.1
        
        return self.state, reward, done, False, {}
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.state = 0
        return self.state, {}

class QLearning:
    def __init__(self, states, actions, learning_rate=0.1, discount=0.95):
        self.q_table = np.zeros((states, actions))
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
    def get_action(self, state, epsilon=0.1):
        if np.random.random() < epsilon:
            return np.random.randint(self.q_table.shape[1])
        return np.argmax(self.q_table[state])
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def update(self, state, action, reward, next_state):
        old_value = self.q_table[state, action] # Current q-value
        next_max = np.max(self.q_table[next_state])
        
        # Q-learning formula
        # Q = current_Q + learning_rate * (reward + discount * max_next_Q - current_Q)
        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_max)
        self.q_table[state, action] = new_value

env = SimpleGridWorld()
agent = QLearning(states=16, actions=4)
episodes = 1000

for episode in range(episodes):
    state, _ = env.reset()
    done = False

    while not done:
        action = agent.get_action(state)
        next_state, reward, done, _, _ = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state

# State -> Action -> Reward -> New State