import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from dqn_agent import DQN, ReplayBuffer  # Import your classes

# --- HYPERPARAMETERS ---
BATCH_SIZE = 64         # How many memories to learn from at once
GAMMA = 0.99            # Discount Factor (Patience)
EPSILON_START = 1.0     # Exploration start (100% random)
EPSILON_END = 0.01      # Exploration end (1% random)
EPSILON_DECAY = 0.995   # Rate of decay
LR = 0.001              # Learning Rate (Alpha)
TARGET_UPDATE = 10      # Update Target Network every 10 episodes

# 1. SETUP ENVIRONMENT
env = gym.make("CartPole-v1") # No render during training to go faster
state_size = env.observation_space.shape[0] # 4
action_size = env.action_space.n            # 2

# 2. SETUP NETWORKS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

policy_net = DQN(state_size, action_size).to(device)
target_net = DQN(state_size, action_size).to(device)
target_net.load_state_dict(policy_net.state_dict()) # Clone weights
target_net.eval() # Teacher mode (no training)

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayBuffer(capacity=10000)

# 3. TRAINING LOOP
num_episodes = 500
epsilon = EPSILON_START

for episode in range(num_episodes):
    state, info = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0).to(device) # Convert to Tensor
    total_reward = 0
    done = False
    
    while not done:
        # A. ACTION SELECTION (Epsilon-Greedy)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                # Ask the brain: "Which action has higher Q-value?"
                action = policy_net(state).argmax().item()

        # B. TAKE STEP
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Process next_state into Tensor
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        reward_tensor = torch.FloatTensor([reward]).to(device)
        action_tensor = torch.LongTensor([action]).to(device)
        done_tensor = torch.FloatTensor([float(done)]).to(device)
        
        # C. STORE MEMORY
        memory.push(state, action_tensor, reward_tensor, next_state_tensor, done_tensor)
        state = next_state_tensor
        total_reward += reward

        # D. TRAIN THE BRAIN (Optimizing)
        if len(memory) > BATCH_SIZE:
            # 1. Sample a random batch of memories
            states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
            
            # Stack them into big tensors (Batch processing is faster)
            states = torch.cat(states)
            actions = torch.cat(actions)
            rewards = torch.cat(rewards)
            next_states = torch.cat(next_states)
            dones = torch.cat(dones)

            # 2. Calculate "Current Q" (What we thought would happen)
            # gather(1, actions) picks the Q-value of the specific action we took
            current_q = policy_net(states).gather(1, actions.unsqueeze(1))

            # 3. Calculate "Target Q" (What actually happened + Future estimate)
            # We ask the TARGET NET (Teacher) for the future value
            max_next_q = target_net(next_states).max(1)[0].detach()
            expected_q = rewards + (GAMMA * max_next_q * (1 - dones))

            # 4. Calculate Loss (MSE: Difference between Guess and Reality)
            loss = nn.MSELoss()(current_q.squeeze(), expected_q)

            # 5. Backpropagation (Update Weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Decay Epsilon
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    # Update Target Network periodically
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

    # Stop if we solve it (Reward 500 is max in v1 usually, 200 is good enough)
    if total_reward >= 450:
        print("SOLVED! Saving Model...")
        torch.save(policy_net.state_dict(), "cartpole_dqn.pth")
        break

print("Training Complete!")