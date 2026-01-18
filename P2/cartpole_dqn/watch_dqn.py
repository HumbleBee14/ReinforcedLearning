import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import time
from dqn_agent import DQN # Import the brain structure

# 1. SETUP
env = gym.make("CartPole-v1", render_mode="human")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. LOAD THE BRAIN
model = DQN(state_size, action_size).to(device)
model.load_state_dict(torch.load("cartpole_dqn.pth"))
model.eval() # Set to Evaluation Mode (Turn off training specific layers if any)

print("Watching the Trained Agent...")

# 3. PLAY 5 GAMES
for i in range(5):
    state, info = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    total_reward = 0
    done = False
    
    while not done:
        env.render()
        
        # PURE EXPLOITATION (No Randomness)
        # We just ask the model: "What is the absolute best move?"
        with torch.no_grad():
            action = model(state).argmax().item()
            
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        total_reward += reward
        
        # 60 FPS cap so it doesn't fly by too fast
        time.sleep(0.016) 

    print(f"Game {i+1} Score: {total_reward}")

env.close()