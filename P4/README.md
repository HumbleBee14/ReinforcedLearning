# P4: PPO - Proximal Policy Optimization

> **Goal:** Implement PPO - the algorithm behind RLHF (ChatGPT, InstructGPT).

## What We'll Learn

- Why vanilla policy gradient is unstable
- Trust region methods
- PPO clipping mechanism
- KL divergence constraint
- GAE (Generalized Advantage Estimation)

```
PPO from scratch (or Stable Baselines3)
        └── Learn: Clipping, KL divergence, advantages
        └── Build: PPO agent on simple environment
```

## Key Bridge Concept

```
Policy Gradient:  Can make too-large updates, unstable
     ↓
PPO:              Clips updates to stay "close" to old policy
     ↓
RLHF:             PPO + Reward Model = ChatGPT training!
```


## Prerequisites

- ✅ P0-P3: All previous projects
- ✅ Docs 00-06

## The PPO Objective

```python
L_CLIP = E[min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)]

Where:
- r(θ) = π_new / π_old  (probability ratio)
- A = advantage
- ε = 0.2 (clip range)
```

## Files (To Be Created)

```
P4/
├── README.md          # This file
├── ppo.py             # PPO algorithm
├── ppo_agent.py       # Agent wrapper
├── gae.py             # Generalized Advantage Estimation
└── train.py           # Training script
```

## Status: 🔜 Coming After P3
