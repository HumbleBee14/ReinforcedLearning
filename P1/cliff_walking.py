import gymnasium as gym
import numpy as np
import time

# 1. ENVIRONMENT - CliffWalking-v1: A standard grid world with a "death zone"
env = gym.make("CliffWalking-v1", render_mode="ansi")

# 2. CREATE THE BRAIN (Q-Table)
# Grid is 4x12 = 48 states
state_size = env.observation_space.n
action_size = env.action_space.n
q_table = np.zeros((state_size, action_size))

# Hyperparameters
learning_rate = 0.1     # Slower learning usually works better here
discount_factor = 0.99  # We care a LOT about the long-term goal
epsilon = 1.0           # Start by exploring
epsilon_decay = 0.995   
min_epsilon = 0.01

num_episodes = 500
print("Training the Agent on the Cliff...")

# 3. TRAINING LOOP
for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    
    while not done:
        # A. Choose Action
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state, :])

        # B. Take Action
        # Note: In this game, falling off the cliff gives Reward = -100
        # Regular steps give Reward = -1 (to encourage speed)
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # C. Update Q-Table
        q_table[state, action] = q_table[state, action] + learning_rate * (
            reward + discount_factor * np.max(q_table[new_state, :]) - q_table[state, action]
        )

        state = new_state

    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

print("Training Complete!")

print("\nWatch the Agent's Strategy:")
input("Press Enter to start...")

env = gym.make("CliffWalking-v1", render_mode="human")
state, info = env.reset()
done = False

while not done:
    action = np.argmax(q_table[state, :])
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()
    time.sleep(0.1)

env.close()