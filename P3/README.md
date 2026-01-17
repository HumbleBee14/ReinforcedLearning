# P3: Policy Gradient & Actor-Critic

> **Goal:** Move from value-based (DQN) to policy-based methods - the foundation for LLM RL.

## What We'll Learn

- Policy Gradient theorem (REINFORCE)
- Why we need policy methods for LLMs
- Actor-Critic architecture
- Advantage function (A = Q - V)
- Baseline reduction for variance

## Key Bridge Concept

```
DQN:           Learn Q(s,a), derive policy from argmax
     ↓
Policy Gradient: Learn π(a|s) DIRECTLY

LLMs ARE policies! π(next_token | context) = the LLM itself!
```

## Prerequisites

- ✅ P0, P1: Tabular RL
- ✅ P2: DQN (neural network as function approximator)
- ✅ Docs 00-05

## Environment

**LunarLander-v2** or **CartPole-v1**
- Continuous or discrete actions
- More complex than CartPole

## Files (To Be Created)

```
P3/
├── README.md              # This file
├── reinforce.py           # Basic REINFORCE algorithm
├── actor_critic.py        # A2C implementation
├── policy_network.py      # Policy network architecture
└── train.py               # Training script
```

## Status: 🔜 Coming After P2
