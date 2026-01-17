# Question: Who Controls the Next State in RL?

## The Question

I'm confused about this concept of agent interaction. The image shows the agent observes the environment, does action `a`, and receives reward/penalty from environment. But it also shows it receives "next state `s'`", and basically builds experiences `[s, a, s', r]`.

I'm a little confused about this next state thing! How can the environment give it what is the next state? Isn't it us or the agent who has to decide what should be the next state? Environment is supposed to behave based on the simulation logic its meant to behave, then how come it is giving the next state to the agent? Am I missing anything here?

<image src="../assets/agent-env-interaction.jpg" alt="Agent-Environment Interaction" />
<image src="../assets/agent-experience.jpg" alt="Agent-Environment Interaction" />

---

## The Answer

**Short Answer:** The **ENVIRONMENT** decides the next state `s'`, not the agent!

**The Agent controls:** What **action** to take  
**The Environment controls:** What **next state** results from that action

---

## Why This Makes Sense

Think of it like the real world:

### Example 1: Frozen Lake 🧊

```
Current state (s): You're at position [1,2]
Agent decides: "I want to move RIGHT" (action a)
Environment responds: "The ice is slippery! You slid DOWN instead!"
Next state (s'): You're now at position [2,2] (not where you intended!)
```

**The agent doesn't control the outcome - it only controls the attempt!**

### Example 2: Video Game 🎮

```
Current state (s): Mario is standing on ground
Agent decides: "JUMP" (action a)
Environment responds: "Gravity pulls you down after 2 seconds"
Next state (s'): Mario lands 5 pixels to the right
```

The environment has **physics rules** that determine what happens after your action.

### Example 3: Chess ♟️

```
Current state (s): Your pieces and opponent's pieces positions
Agent decides: "Move knight to E4" (action a)
Environment responds: "Opponent captures your knight with bishop"
Next state (s'): Board with your knight removed
```

The environment (which includes opponent behavior) determines the outcome.

---

## The Key Insight

```
┌─────────────────────────────────────────────────────────────┐
│                   The Agent-Environment Split              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  AGENT CONTROLS:           ENVIRONMENT CONTROLS:            │
│  ✓ Action selection (a)    ✓ Next state (s')               │
│  ✓ Learning/updating       ✓ Reward (r)                     │
│  ✓ Strategy                ✓ State transitions              │
│                           ✓ Terminal conditions            │
│                                                             │
│  Agent says: "I WANT to go right"                          │
│  Environment says: "You ENDED UP going down"               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## The Mathematical Model

The environment is defined by **transition probabilities**:

```
P(s' | s, a) = Probability of landing in state s', 
               given you're in state s and take action a
```

**Examples:**

| Environment | Transition | Probability |
|-------------|-----------|-------------|
| **Frozen Lake** | Take "Right" → Actually move Right | 0.33 |
| (Slippery!) | Take "Right" → Slip Down | 0.33 |
|             | Take "Right" → Slip Up | 0.33 |
| **Chess** | Move Knight → Opponent doesn't capture | 0.7 |
|           | Move Knight → Opponent captures | 0.3 |
| **Cliff Walking** | Walk toward cliff → Fall off | 1.0 (deterministic) |

---

## Why Can't the Agent Decide `s'`?

**If the agent could choose where to end up, RL would be trivial!**

```python
# If agent controlled s':
agent.choose_next_state(GOAL_STATE)  # Just teleport to the goal!
# Game over in 1 step. No learning needed.

# Reality:
agent.choose_action("move_right")    # Agent tries
environment.execute(s, a)            # Environment decides outcome
s_next = environment.sample_transition(s, a)  # Could be anywhere!
```

---

## The Experience Tuple `[s, a, s', r]`

```
┌─────────────────────────────────────────────────────────────┐
│              What Gets Stored in Memory                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  s   = State I was in         (Observed from environment)   │
│  a   = Action I chose         (Decided by agent)            │
│  s'  = State I ended up in    (Determined by environment)   │
│  r   = Reward I received      (Given by environment)        │
│                                                             │
│  The agent learns: "When I tried action a in state s,       │
│                     I ended up in state s' with reward r"   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Code Example from Frozen Lake

```python
# In slippery_frozen_lake.py
state = env.reset()                    # Environment gives initial state

while not done:
    action = agent.choose_action()      # Agent decides action
    
    # HERE'S THE KEY LINE:
    next_state, reward, done, _, _ = env.step(action)
    #  ↑            ↑
    #  Environment determines both!
    
    # Store the experience
    memory.append([state, action, next_state, reward])
    
    state = next_state  # Move to the state the environment gave us
```

---

## Visual Clarification

```
                Agent's Control Zone
                       ↓
    ┌─────────────────────────────────────────┐
    │  🤖 AGENT                               │
    │                                         │
    │  1. Observes state (s) ←───────────────┼──┐
    │  2. Decides action (a)                  │  │
    │  3. Stores experience [s,a,s',r]        │  │
    │                                         │  │
    └────────────┬────────────────────────────┘  │
                 │ Sends action                  │
                 ↓                                │
    ┌─────────────────────────────────────────┐  │
    │  🌍 ENVIRONMENT                         │  │
    │                                         │  │
    │  1. Receives action (a)                 │  │
    │  2. Computes next state (s') ──────────►│──┤
    │  3. Computes reward (r) ───────────────►│──┘
    │  4. Sends [s', r] back to agent         │
    │                                         │
    │  (Has physics, rules, randomness)       │
    └─────────────────────────────────────────┘
         Environment's Control Zone
```

---

## Summary

**The key realization:**

> The environment is just telling me **where my action took me**, based on how the environment behaves (physics, rules, randomness).

The agent is like a player trying things in a world they don't fully control:
- **Agent:** "I'm going to jump!"
- **Environment:** "Okay, but there's wind, so you only jumped 3 feet instead of 5"
- **Agent:** "I'll remember that: jumping in wind = shorter distance"

This is what makes RL interesting and challenging - the agent must learn to be effective in an uncertain world!

---

**Related Concepts:**
- Markov Decision Process (MDP)
- Stochastic vs Deterministic Environments
- State Transition Probability `P(s'|s,a)`
- Experience Replay
