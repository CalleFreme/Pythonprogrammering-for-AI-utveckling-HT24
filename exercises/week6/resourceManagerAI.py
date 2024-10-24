# En agent som ska lära sig hantera resurser (t.ex. serverkapacitet)
# för att möta varierande efterfrågan

import gymnasium as gym
import numpy as np
from collections import defaultdict
import time

class ResourceManager(gym.Env):
    """
    Miljö där en agent ska hantera resurser (t.ex. serverkapacitet)
    för att möta varierande efterfrågan till lägsta möjliga kostnad.
    """
    def __init__(self):
        # Tillstånd: (nuvarande_kapacitet, nuvarande_efterfrågan)
        self.observation_space = gym.spaces.Discrete(50)  # 5 kapacitetsnivåer * 10 efterfrågansnivåer
        
        # Handlingar: minska kapacitet (-1), behåll (0), öka kapacitet (+1)
        self.action_space = gym.spaces.Discrete(3)
        
        self.max_capacity = 4  # Max kapacitetsnivå
        self.max_demand = 9    # Max efterfrågansnivå
        self.reset()
        
    def encode_state(self):
        """Konverterar kapacitet och efterfrågan till ett unikt tillståndsnummer"""
        return self.capacity * 10 + self.demand
    
    def get_demand(self):
        """Simulerar varierande efterfrågan med viss slumpmässighet"""
        # Efterfrågan följer ett mönster men med slumpmässig variation
        self.time_step += 1
        base_demand = int(4 + 2 * np.sin(self.time_step / 10))  # Sinuskurva för att simulera dagscykel
        variation = self.np_random.integers(-1, 2)  # Slumpmässig variation
        return max(0, min(self.max_demand, base_demand + variation))
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.capacity = 2  # Börja med medelhög kapacitet
        self.demand = 4    # Börja med medelhög efterfrågan
        self.time_step = 0
        return self.encode_state(), {}
    
    def step(self, action):
        """
        Utför en handling och returnerar nytt tillstånd, belöning, etc.
        action: 0 (minska), 1 (behåll), 2 (öka)
        """
        # Uppdatera kapacitet baserat på handling
        if action == 0:  # Minska
            self.capacity = max(0, self.capacity - 1)
        elif action == 2:  # Öka
            self.capacity = min(self.max_capacity, self.capacity + 1)
        
        # Uppdatera efterfrågan
        self.demand = self.get_demand()
        
        # Beräkna belöning
        # Kostnad för kapacitet: -1 per enhet
        capacity_cost = -self.capacity
        
        # Kostnad för otillräcklig kapacitet: -2 per enhet som saknas
        shortage_cost = -2 * max(0, self.demand - self.capacity)
        
        # Total belöning
        reward = capacity_cost + shortage_cost
        
        done = False  # Denna miljö har ingen naturlig slutpunkt
        
        return self.encode_state(), reward, done, False, {}


# Exempel på hur man tränar agenten för ResourceManager
def train_resource_manager(episodes=1000):
    """
    Tränar en Q-learning agent för ResourceManager-miljön
    """
    env = ResourceManager()
    
    # Initialisera Q-tabell
    q_table = defaultdict(lambda: np.zeros(3))  # 3 möjliga handlingar
    
    # Hyperparametrar
    learning_rate = 0.1
    discount_factor = 0.95
    epsilon = 0.1
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        for _ in range(100):  # Max 100 steg per episode
            # Epsilon-greedy handlingsval
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
            
            # Utför handling
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            
            # Q-learning uppdatering
            old_value = q_table[state][action]
            next_max = np.max(q_table[next_state])
            new_value = (1 - learning_rate) * old_value + learning_rate * (reward + discount_factor * next_max)
            q_table[state][action] = new_value
            
            state = next_state
            if done:
                break
        
        # Visa framsteg var 100:e episode
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}, Genomsnittlig belöning: {total_reward/100:.2f}")
    
    return q_table

# För att köra träningen:
if __name__ == "__main__":
    print("Tränar resurshanteraren...")
    q_table = train_resource_manager()
    print("Träning klar!")