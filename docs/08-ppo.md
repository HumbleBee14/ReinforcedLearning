# PPO - Proximal Policy Optimization

> The algorithm that powers ChatGPT's RLHF training

## Why PPO?

Policy gradient updates can be too large → training collapses.
PPO clips the updates to stay in a "trust region."

## The PPO Objective

```
L_CLIP = E[min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)]
```

## Key Concepts

### 1. Importance Sampling Ratio

### 2. Clipping Mechanism

### 3. KL Divergence Penalty

### 4. GAE (Generalized Advantage Estimation)

## Coming soon with P4 implementation...
