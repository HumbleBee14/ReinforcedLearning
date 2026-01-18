# P2: CartPole with DQN (Deep Q-Network)

> **Goal:** Move from Q-tables to neural networks as function approximators.
> **The Algorithm That Started the AI Revolution** - DeepMind used DQN to beat Atari games in 2013.

---

## The Problem: Why Tables Don't Scale

In P0/P1, we used a **Q-Table** (a Python dictionary/NumPy array) to store values:

```python
# Tabular Q-Learning (FrozenLake: 16 states × 4 actions = 64 values)
Q[state][action] = value
```

**This breaks for CartPole because:**

| Environment | State Space | Table Size |
|-------------|-------------|------------|
| FrozenLake | 16 discrete states | 64 cells ✅ |
| Taxi | 500 discrete states | 3,000 cells ✅ |
| CartPole | **Infinite** (continuous!) | ∞ cells ❌ |

CartPole state = `[position, velocity, angle, angular_velocity]` - all floating point numbers!
You can't have a row for every possible value of `0.0001` vs `0.0002` vs `0.0003`...

**The Solution:** Replace the lookup table with a **Neural Network** that *predicts* Q-values.

---

## The Bridge: Table → Neural Network

```
┌─────────────────────────────────────────────────────────────────────┐
│                    THE PARADIGM SHIFT                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Q-Table (P0/P1):                                                   │
│  ─────────────────                                                  │
│  Q[state][action] = lookup_value                                    │
│  Storage: O(States × Actions)                                       │
│  Problem: Can't handle continuous or large state spaces             │
│                                                                     │
│  DQN (P2):                                                          │
│  ─────────                                                          │
│  Q(state) = NeuralNetwork(state) → [Q_left, Q_right]                │
│  Storage: O(Network Parameters) - Fixed size regardless of states!  │
│  Solution: Generalizes to unseen states through learned patterns    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
P2/
├── README.md                      # This file
└── cartpole/
    ├── cartpole_random.py         # Step 1: Random agent (baseline ~14 reward)
    ├── dqn_agent.py               # Step 2: Neural Network + Replay Buffer
    ├── train_dqn.py               # Step 3: Training loop with Target Network
    ├── watch_dqn.py               # Step 4: Load & watch trained agent
    └── cartpole_dqn.pth           # Saved model weights (after training)
```

---

## 🚀 Welcome to Phase 2: Deep Reinforcement Learning

This is the official start of **Phase 2: Deep Reinforcement Learning.**

We are leaving behind the "Excel Spreadsheets" (Q-Tables) and building a **Neural Network Brain**.

### The Challenge: Infinite States

In the previous basic projects, we used Q-Tables to store values for each state. For example, in the Taxi project, you had exactly 500 states. You could easily make a table with 500 rows.

In this new project (**CartPole**), the state is **Continuous**:

```
State = [Cart Position, Cart Velocity, Pole Angle, Pole Velocity]

Example: [0.003, 0.45, -0.02, 1.2]
```

**Problem:** You cannot make a table row for `0.003`. If the cart moves a tiny bit to `0.0030001`, that would be a new row. You would need **infinite RAM**.

**The Solution:** Instead of storing the answer for every possible state, we train a **Neural Network** to *calculate* the answer on the fly.

- **Input:** The 4 numbers (State)
- **Output:** The Q-Values for Left and Right

### Meet the Environment: CartPole-v1

| Property | Value |
|----------|-------|
| **Goal** | Balance a pole on a cart |
| **Actions** | Push Left (0) or Push Right (1) |
| **Reward** | +1 for every frame the pole stays upright |
| **Game Over** | Pole falls >12° OR cart runs off screen |
| **Max Score** | 500 (solved if consistently reaches 450+) |

---

## Step 1: Random Agent (Baseline)

**File:** `cartpole_random.py`

```python
action = env.action_space.sample()  # Random: Left or Right
```

**Result:** Score ~14 before pole falls.

**Purpose:** Proves the environment works, but a "brainless" agent fails instantly.

---

## Step 2: The Brain & Memory

**File:** `dqn_agent.py`

### The Brain: Neural Network (DQN)

```python
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        # Input Layer:  4 features (position, velocity, angle, angular_velocity)
        # Hidden Layer: 128 neurons
        # Hidden Layer: 128 neurons  
        # Output Layer: 2 Q-values (Q_left, Q_right)
```

**Architecture:**
```
State (4 numbers) → [128 neurons] → [128 neurons] → Q-values (2 numbers)
                        ↓               ↓
                      ReLU            ReLU
```

**How it works:**
1. Feed in state: `[0.02, -0.5, 0.1, 0.8]`
2. Network outputs: `[Q_left=2.3, Q_right=5.1]`
3. Pick action: `argmax([2.3, 5.1])` → Right (index 1)

### The Memory: Replay Buffer

```python
class ReplayBuffer:
    def push(self, state, action, reward, next_state, done):
        self.buffer.append(...)  # Store experience
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)  # Random memories
```

**Why We Need This: Catastrophic Forgetting**

| Problem | Without Replay Buffer | With Replay Buffer |
|---------|----------------------|-------------------|
| Training data | Sequential (step 1, 2, 3...) | Randomized |
| Network behavior | Overfits to current moment | Learns general patterns |
| Past experiences | Forgotten immediately | Mixed into every batch |

**Analogy:** 
> If you only study the last chapter of a textbook, you'll ace that chapter but fail the rest of the exam.
> The Replay Buffer makes you randomly review old chapters while studying new ones.

---

## Step 3: Training Loop

**File:** `train_dqn.py`

### The Two-Brain Trick: Policy vs Target Network

```python
policy_net = DQN(...)  # The Student (actively learning)
target_net = DQN(...)  # The Teacher (frozen copy)
```

**The Problem: Chasing Your Own Tail**
```
In Q-Learning: Q(s,a) ← reward + γ × max Q(s', a')
                                      ↑
                              This is OUR OWN estimate!
```

If we update based on our own moving estimates, training becomes unstable.

**The Solution: Target Network**
- Policy Network: Updates every step
- Target Network: Frozen copy, updated every 10 episodes
- We ask the "Teacher" for future values, not the "Student"

```python
# Target stays stable while Policy learns
max_next_q = target_net(next_states).max(1)[0]  # Ask Teacher
expected_q = rewards + (GAMMA * max_next_q * (1 - dones))

# Compare Student's guess vs Target
loss = MSELoss(current_q, expected_q)
```

### The Training Algorithm

```
For each episode:
    1. OBSERVE state
    2. SELECT action (ε-greedy: random if exploring, network otherwise)
    3. EXECUTE action, observe reward and next_state
    4. STORE (state, action, reward, next_state, done) in memory
    5. SAMPLE random batch from memory
    6. COMPUTE loss: (Student's Q) vs (Reward + Teacher's future Q)
    7. BACKPROPAGATE and update weights
    8. DECAY epsilon (explore less over time)
    9. Every N episodes: Copy Student → Teacher
```

### Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `BATCH_SIZE` | 64 | Memories per training step |
| `GAMMA` | 0.99 | Discount factor (patience) |
| `EPSILON_START` | 1.0 | 100% random initially |
| `EPSILON_END` | 0.01 | 1% random after training |
| `EPSILON_DECAY` | 0.995 | Decay rate per episode |
| `LR` | 0.001 | Learning rate (Adam optimizer) |
| `TARGET_UPDATE` | 10 | Sync Target every 10 episodes |

### Training Progress

```
Episode 0-50:   Reward ~15-30    (Exploring randomly, epsilon high)
Episode 50-150: Reward ~50-100   (Brain connecting patterns)
Episode 150+:   Reward ~200-500  (Policy converging, "Aha!" moment)
Episode 200+:   SOLVED! 450+     (Model saved)
```

---

## Step 4: Watch the Trained Agent

**File:** `watch_dqn.py`

```python
model = DQN(state_size, action_size)
model.load_state_dict(torch.load("cartpole_dqn.pth"))
model.eval()  # No training, pure exploitation

# Pure exploitation: Always pick best action
action = model(state).argmax().item()
```

**Result:** Score 450-500 consistently! (Max is 500 in CartPole-v1)

---

## Key Takeaways: Tabular vs Deep Q-Learning

| Aspect | Tabular Q-Learning (P0/P1) | Deep Q-Learning (P2) |
|--------|---------------------------|---------------------|
| **State space** | Discrete (16, 500 states) | Continuous (infinite) |
| **Q-function** | Lookup table | Neural network |
| **Updates** | Immediate after each step | Batch from replay buffer |
| **Stability** | Always stable | Needs target network |
| **Generalization** | None (exact state match) | Learns patterns |

---

## The DQN Recipe (Summary)

```
1. NEURAL NETWORK       → Replace Q-table with function approximator
2. REPLAY BUFFER        → Store experiences, sample randomly (break correlation)
3. TARGET NETWORK       → Stable learning target (don't chase your tail)
4. ε-GREEDY EXPLORATION → Explore early, exploit later
5. BATCH TRAINING       → Learn from multiple experiences at once
```

---

## What's Next?

**P3: Policy Gradient (LunarLander)**

DQN still calculates Q-values and derives the policy from them.
Policy Gradient methods learn the policy **directly** - which is how LLMs work!

```
DQN:           State → Network → Q-values → argmax → Action
Policy Gradient: State → Network → Probabilities → Sample → Action

LLMs ARE Policy Networks: Context → Transformer → Token Probabilities → Sample → Token
```

---

## Run the Code

```bash
# 1. Test random agent (baseline)
python cartpole/cartpole_random.py

# 2. Train DQN (takes a few minutes)
python cartpole/train_dqn.py

# 3. Watch trained agent
python cartpole/watch_dqn.py
```

**Dependencies:** `gymnasium`, `torch`, `numpy`
