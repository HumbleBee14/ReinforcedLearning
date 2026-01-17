# P2: CartPole with DQN (Deep Q-Network)

> **Goal:** Move from Q-tables to neural networks as function approximators.

## What We'll Learn

- Why Q-tables don't scale (state space explosion)
- Neural networks as universal function approximators
- Experience Replay (learning from a "diary")
- Target Networks (stability trick)
- The DQN architecture

```
P2: CartPole with DQN
        └── Learn: Neural networks as function approximators
        └── Build: Your first deep RL agent
```

## Key Bridge Concept

```
Q-Table:     Q[state][action] = value     (lookup table)
     ↓
DQN:         Q(state) = NeuralNet(state)  (function approximation)
```

## Prerequisites

- ✅ P0: Frozen Lake (random agent, understanding environments)
- ✅ P1: Cliff Walking (Q-Learning, SARSA)
- ✅ Docs 00-04 (RL Fundamentals, Q-values, exploration)

## Environment

**CartPole-v1** from Gymnasium
- State: [cart_position, cart_velocity, pole_angle, pole_velocity]
- Actions: [push_left, push_right]
- Reward: +1 for each step the pole stays upright
- Goal: Balance pole for 500 steps

## Files (To Be Created)

```
P2/
├── README.md          # This file
├── cartpole_dqn.py    # Main DQN implementation
├── replay_buffer.py   # Experience replay
├── dqn_network.py     # Neural network architecture
└── train.py           # Training script
```

## Status: 🔜 Coming Next
