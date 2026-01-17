# DQN: From Tables to Neural Networks

> Moving from tabular Q-learning to Deep Q-Networks

## The Problem with Tables

```
Frozen Lake: 16 states × 4 actions = 64 values ✅ Easy!
CartPole:    Infinite states (continuous) × 2 actions = ??? ❌ Impossible!
Atari:       210×160×3 pixels = millions of states ❌ Forget it!
```

## The Solution: Function Approximation

Instead of storing Q(s,a) in a table, we APPROXIMATE it with a neural network.

```
Q-Table:    Q[state][action] = lookup
DQN:        Q(state) = NeuralNet(state) → [Q(a1), Q(a2), Q(a3), ...]
```

```
┌─────────────────────────────────────────────────────────────────────┐
│                 THE BRIDGE TO LLM RL                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Tabular Q-Learning:  Q[state][action] = value                      │
│       ↓                                                             │
│  DQN:                 Q(state, action) = NeuralNet(state) → value   │
│       ↓                                                             │
│  Policy Gradient:     π(action|state) = NeuralNet(state) → probs    │
│       ↓                                                             │
│  LLM:                 π(token|context) = LLM(context) → probs       │
│                                                                     │
│  It's the same thing, just with bigger neural networks!             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Concepts

### 1. Experience Replay

### 2. Target Network

### 3. Loss Function

## Coming soon with P2 implementation...
