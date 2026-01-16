import gymnasium as gym
import numpy as np
import time
import os

# 1. SETUP THE ENVIRONMENT (The Game)
# is_slippery=True means the elf slides around randomly (hard mode!)
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True, render_mode="ansi")

# 2. CREATE THE BRAIN (The Q-Table)
# It's a spreadsheet: Rows = States (Where am I?), Cols = Actions (Move Left, Down, Right, Up)
# Initially, the brain knows nothing (all zeros).
state_size = env.observation_space.n
action_size = env.action_space.n
q_table = np.zeros((state_size, action_size))

# Hyperparameters (The "Personality" of the learner)
learning_rate = 0.8   # How fast we update our beliefs (High = impulsive)
discount_factor = 0.95 # How much we care about future rewards (High = long-term thinker)
epsilon = 1.0          # Exploration rate: 100% random actions at start
epsilon_decay = 0.995  # We slowly become more confident
min_epsilon = 0.01

num_episodes = 2000

print("Training the Agent...")

# 3. THE TRAINING LOOP
for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    
    while not done:
        # A. DECIDE ACTION (Explore vs Exploit)
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore: Do something random
        else:
            action = np.argmax(q_table[state, :]) # Exploit: Do what we think is best

        # B. TAKE ACTION
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # C. UPDATE THE BRAIN (The Q-Learning Formula)
        # This is the "Magic Math": NewValue = OldValue + LearningRate * (Reward + BestFutureValue - OldValue)
        q_table[state, action] = q_table[state, action] + learning_rate * (
            reward + discount_factor * np.max(q_table[new_state, :]) - q_table[state, action]
        )

        state = new_state

    # Decay exploration (The agent grows up)
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

print("Training Complete!")
print("\nFinal Cheat Sheet (Q-Table):\n", np.round(q_table, 2))

# 4. WATCH IT PLAY
input("\nPress Enter to watch the smart agent play...")
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True, render_mode="human")
state, info = env.reset()
done = False

while not done:
    action = np.argmax(q_table[state, :]) # Use the learned brain
    state, _, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    env.render()
    time.sleep(0.5)

env.close()