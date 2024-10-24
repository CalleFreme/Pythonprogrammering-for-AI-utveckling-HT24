# En agent som ska lära sig plocka upp och lämna av passagerare på rätt platser
# Använder en förenklad version av Gymnasiums Taxi-miljö

import gymnasium as gym
import numpy as np
from collections import defaultdict
import time

class SimpleTaxi(gym.Env):
    """
    Enkel taxivärld där en taxi ska plocka upp en passagerare och lämna av den.
    Världen är ett 5x5 rutnät med fyra möjliga upphämtningsplatser (R, G, B, Y).
    """
    def __init__(self):
        # Världen är 5x5, taxi kan vara på vilken ruta som helst
        # För varje ruta kan passageraren vara på 4 olika platser eller i taxin
        # Passageraren ska till någon av de 4 platserna
        self.observation_space = gym.spaces.Discrete(5 * 5 * 5 * 4)
        
        # Möjliga handlingar: upp, höger, ner, vänster, plocka upp, lämna av
        self.action_space = gym.spaces.Discrete(6)
        
        # Platsernas positioner (rad, kolumn)
        self.locations = {
            'R': (0, 0),  # Röd plats
            'G': (0, 4),  # Grön plats
            'B': (4, 0),  # Blå plats
            'Y': (4, 4)   # Gul plats
        }
        
        self.reset()
    
    def encode_state(self):
        """Konverterar nuvarande tillstånd till ett unikt nummer"""
        # taxi_row * 100 + taxi_col * 20 + passenger_loc * 4 + destination
        return (self.taxi_row * 100 + 
                self.taxi_col * 20 + 
                self.passenger_location * 4 + 
                self.destination)
    
    def decode_state(self, state):
        """Avkodar ett tillståndsnummer till dess komponenter"""
        taxi_row = state // 100
        state = state % 100
        taxi_col = state // 20
        state = state % 20
        passenger_loc = state // 4
        destination = state % 4
        return taxi_row, taxi_col, passenger_loc, destination
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Placera taxi slumpmässigt
        self.taxi_row = self.np_random.integers(5)
        self.taxi_col = self.np_random.integers(5)
        
        # Välj slumpmässig start och destination (0-3 motsvarar R,G,B,Y)
        self.passenger_location = self.np_random.integers(4)  # Passagerarens startplats
        self.destination = self.np_random.integers(4)        # Destinationen
        while self.destination == self.passenger_location:   # Se till att de är olika
            self.destination = self.np_random.integers(4)
        
        return self.encode_state(), {}
    
    def step(self, action):
        """
        Utför en handling och returnerar nytt tillstånd, belöning, om klart, etc.
        Förbättrad belöningsstruktur för att undvika loopar
        """
        prev_row, prev_col = self.taxi_row, self.taxi_col
        reward = -1  # Standardbelöning för varje steg
        done = False
        
        if action < 4:  # Förflyttningshandlingar
            if action == 0:  # Upp
                self.taxi_row = max(0, self.taxi_row - 1)
            elif action == 1:  # Höger
                self.taxi_col = min(4, self.taxi_col + 1)
            elif action == 2:  # Ner
                self.taxi_row = min(4, self.taxi_row + 1)
            elif action == 3:  # Vänster
                self.taxi_col = max(0, self.taxi_col - 1)
            
            # Extra bestraffning om taxin inte rörde sig (körde in i vägg)
            if (self.taxi_row, self.taxi_col) == (prev_row, prev_col):
                reward = -2
                
            # Mindre bestraffning när taxin rör sig mot målet
            if self.passenger_location < 4:  # Om passageraren väntar
                target = list(self.locations.values())[self.passenger_location]
            else:  # Om passageraren är i taxin
                target = list(self.locations.values())[self.destination]
                
            # Beräkna om vi kom närmare målet
            prev_dist = abs(prev_row - target[0]) + abs(prev_col - target[1])
            new_dist = abs(self.taxi_row - target[0]) + abs(self.taxi_col - target[1])
            if new_dist < prev_dist:
                reward = -0.5  # Mindre bestraffning för att röra sig mot målet
            
        elif action == 4:  # Plocka upp
            if self.passenger_location < 4:  # Om passageraren inte redan är i taxin
                loc_coords = list(self.locations.values())[self.passenger_location]
                if (self.taxi_row, self.taxi_col) == loc_coords:
                    self.passenger_location = 4  # 4 betyder "i taxin"
                    reward = 15  # Öka belöningen för upphämtning
                else:
                    reward = -10
            else:
                reward = -10
                
        elif action == 5:  # Lämna av
            dest_coords = list(self.locations.values())[self.destination]
            if (self.taxi_row, self.taxi_col) == dest_coords and self.passenger_location == 4:
                self.passenger_location = self.destination
                reward = 30  # Öka belöningen för korrekt avlämning
                done = True
            else:
                reward = -10
                
        return self.encode_state(), reward, done, False, {}
    
def visualize_taxi(env):
    """
    Skapar en textbaserad visualisering av taxi-miljön
    """
    # Skapa en representation av världen
    world = [['░' for _ in range(5)] for _ in range(5)]
    
    # Markera platserna
    locations = {
        0: 'R',  # Röd plats
        1: 'G',  # Grön plats
        2: 'B',  # Blå plats
        3: 'Y'   # Gul plats
    }
    
    # Placera ut platsmarkörer
    world[0][0] = 'R'
    world[0][4] = 'G'
    world[4][0] = 'B'
    world[4][4] = 'Y'
    
    # Placera taxi
    world[env.taxi_row][env.taxi_col] = '🚕' if env.passenger_location == 4 else '🚖'
    
    # Markera passagerarens position om den inte är i taxin
    if env.passenger_location < 4:
        pos = list(env.locations.values())[env.passenger_location]
        world[pos[0]][pos[1]] = f'P{locations[env.passenger_location]}'
    
    # Markera destinationen
    dest = list(env.locations.values())[env.destination]
    world[dest[0]][dest[1]] = f'D{locations[env.destination]}'
    
    # Skriv ut världen
    print("\033[H\033[J")  # Rensa terminalfönstret
    print("Taxi World - 🚖=tom taxi, 🚕=taxi med passagerare")
    print("P=Passagerare, D=Destination")
    for row in world:
        print(' '.join(row))
    print("\n")

def train_taxi_driver(episodes=2000, render=False):  # Öka antal episoder
    """
    Tränar en Q-learning agent för taxi-miljön med förbättrade parametrar
    """
    env = SimpleTaxi()
    
    # Initialisera Q-tabell med små slumpmässiga värden istället för nollor
    q_table = defaultdict(lambda: np.random.uniform(low=-1, high=1, size=6))
    
    # Justerade hyperparametrar
    learning_rate = 0.15
    discount_factor = 0.99  # Öka för att värdera framtida belöningar högre
    epsilon = 1.0
    epsilon_decay = 0.998  # Långsammare decay
    epsilon_min = 0.05  # Högre minimum för att behålla viss exploration
    
    # För att spåra prestanda
    rewards_history = []
    steps_history = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            if render and episode % 100 == 0:
                visualize_taxi(env)
                time.sleep(0.1)
            
            # Epsilon-greedy med lite modifiering för att uppmuntra målmedvetet beteende
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                # Om det finns flera handlingar med samma värde, välj slumpmässigt bland dem
                best_value = np.max(q_table[state])
                best_actions = np.where(q_table[state] == best_value)[0]
                action = np.random.choice(best_actions)
            
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            # Q-learning uppdatering med lite högre learning rate för positiva belöningar
            old_value = q_table[state][action]
            next_max = np.max(q_table[next_state])
            
            # Använd högre learning rate för positiva belöningar
            current_lr = learning_rate * 1.5 if reward > 0 else learning_rate
            
            new_value = (1 - current_lr) * old_value + current_lr * (reward + discount_factor * next_max)
            q_table[state][action] = new_value
            
            state = next_state
            
            if done or steps > 100:  # Avbryt om klar eller för många steg
                break
        
        # Uppdatera epsilon med decay
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # Spara historik
        rewards_history.append(total_reward)
        steps_history.append(steps)
        
        # Visa framsteg var 100:e episode
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            avg_steps = np.mean(steps_history[-100:])
            print(f"Episode {episode + 1}")
            print(f"Genomsnittlig belöning: {avg_reward:.2f}")
            print(f"Genomsnittliga steg: {avg_steps:.2f}")
            print(f"Epsilon: {epsilon:.2f}")
            print("-" * 40)
    
    return q_table

def run_trained_taxi(q_table, episodes=5):
    """
    Kör den tränade taxin några episoder för demonstration
    """
    env = SimpleTaxi()
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        print(f"\nEpisod {episode + 1}")
        
        while True:
            visualize_taxi(env)
            time.sleep(0.5)  # Paus för att kunna se vad som händer
            
            # Välj bästa handling enligt Q-tabellen
            action = np.argmax(q_table[state])
            
            # Utför handling
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            if done or steps > 100:
                visualize_taxi(env)
                print(f"Episod klar! Total belöning: {total_reward}, Antal steg: {steps}")
                time.sleep(2)
                break

# För att köra träningen och demonstrationen:
if __name__ == "__main__":
    print("Tränar taxichauffören...")
    q_table = train_taxi_driver(episodes=1000, render=True)
    print("\nTräning klar! Kör några demonstrationsepisoder...")
    run_trained_taxi(q_table)
    
# För att köra träningen:
if __name__ == "__main__":
    print("Tränar taxichaufför...")
    q_table = train_taxi_driver()
    print("Träning klar!")