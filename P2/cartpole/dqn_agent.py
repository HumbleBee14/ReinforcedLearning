import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# 1. THE BRAIN (Neural Network)
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        # Layer 1: Takes in 4 inputs (state), outputs 128 features
        self.fc1 = nn.Linear(state_size, 128)

        self.fc2 = nn.Linear(128, 128) # Hidden layer
        
        self.fc3 = nn.Linear(128, action_size) # Layer 3: Output layer (2 actions)

    def forward(self, x):
        # Pass input through Layer 1 & 2 + ReLU activation (introduces non-linearity)
        x = torch.relu(self.fc1(x))

        x = torch.relu(self.fc2(x))
        # Pass through Output Layer (No activation, we want raw Q-values)
        return self.fc3(x)

# 2. THE MEMORY (Replay Buffer)
# Why do we need this?
# In tables, we update immediately. In Neural Networks, if we update 
# on correlated sequential frames (step 1, step 2, step 3), the network 
# overfits to the current situation and forgets everything else.
# We store experiences and sample them RANDOMLY to train.
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        # Randomly pick 'batch_size' experiences
        batch = random.sample(self.buffer, batch_size)
        
        # Unzip the batch into separate arrays (States, Actions, Rewards...)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)