# Reinforcement Learning: The Slippery Frozen Lake Problem

This project demonstrates a fundamental Reinforcement Learning algorithm called **Q-learning**. We are teaching an AI agent to solve the "Frozen Lake" puzzle from the `gymnasium` library.

## 1. The Goal: Cross the Frozen Lake

The "environment" is a 4x4 grid.

-   **S**: Starting position (safe)
-   **F**: Frozen tile (safe)
-   **H**: Hole (game over)
-   **G**: Goal (you win!)

The agent must navigate from **S** to **G**.

### The Twist: `is_slippery=True`

The lake is slippery. If the agent tries to move in one direction, there is a 1/3 chance it will slide to the left or right of its intended direction. This uncertainty makes the problem more challenging and is a perfect scenario for reinforcement learning.

## 2. The "Brain": The Q-Table

The core of our agent is the **Q-Table**. It's a simple table (a 2D array) that acts as a "cheat sheet" to guide the agent's decisions.

-   **Rows**: Represent the 16 possible "states" (each tile on the grid).
-   **Columns**: Represent the 4 possible "actions" (LEFT, DOWN, RIGHT, UP).
-   **Values**: Each cell contains a "Q-value," which is a score that estimates the quality of taking an action from a specific state.

The entire goal of the training process is to fill this table with accurate scores.

```
Training the Agent...
Training Complete!

Final Cheat Sheet (Q-Table):   (Final Learning after 2000 episodes, could be either it learned to reach goal or not)
 [[0.17 0.04 0.06 0.07]
 [0.   0.02 0.01 0.07]
 [0.02 0.05 0.01 0.02]
 [0.   0.   0.01 0.06]
 [0.29 0.01 0.   0.02]
 [0.   0.   0.   0.  ]
 [0.13 0.   0.01 0.  ]
 [0.   0.   0.   0.  ]
 [0.01 0.02 0.03 0.42]
 [0.   0.73 0.09 0.11]
 [0.68 0.05 0.02 0.02]
 [0.   0.   0.   0.  ]
 [0.   0.   0.   0.  ]
 [0.   0.07 0.69 0.13]
 [0.17 0.94 0.06 0.11]
 [0.   0.   0.   0.  ]]

```

## 3. The Method: Q-Learning

The agent learns by trial and error over many "episodes" (attempts at the game). In each step of an episode, it follows this loop:

### A. Explore vs. Exploit

The agent must balance two goals:
1.  **Exploration**: Acting randomly to discover new paths and learn about the consequences of its actions.
2.  **Exploitation**: Using its existing knowledge (the Q-Table) to make the best possible move.

We control this with a variable called `epsilon`. It starts high (pure exploration initially, that's how we learn :) and gradually decreases it over time as the agent learns more and more, allowing the agent to become more confident in its own knowledge.

 - The learning_rate controls how much this new information changes the old value.
 - The discount_factor determines how much the agent cares about future rewards versus immediate ones

### B. Take an Action & Get Feedback

The agent takes a step in the environment and receives feedback:
-   A **new state** (where it landed).
-   A **reward** (+1 for reaching the goal, 0 for everything else).
-   A signal if the game is **done**.

### C. Learn from the Experience

This is the "learning" step, defined by the Q-learning formula:

`New_Q_Value = Old_Q_Value + Learning_Rate * (Reward + Discount_Factor * Best_Future_Value - Old_Q_Value)`

In simpler terms, we update the Q-value (the score) for the state-action pair we just tried. The update is based on the reward we received and the potential for future rewards from our new location. This slowly improves the "cheat sheet."

## 4. The Result: A Smart Agent

After thousands of training episodes, the Q-Table is filled with optimized values. The final part of the script runs the simulation in "human" mode, where the agent **only exploits** its knowledge. It consults the Q-Table at every step and chooses the action with the highest score, allowing us to watch the fully trained agent navigate the slippery lake.

------
------

## Learning with Tables (Q-learning)

Note: We could have used basic fixed policies or Monte Carlo simulations to estimate Q-values. But what if we want the agent to learn its own policy and discover better decisions through experience?

Welcome to Q-learning: one of the most popular and foundational reinforcement learning algorithms. It lets the agent:

- Interact with the environment
- Observe the consequences of its actions
- Learn over time what actions lead to success

Unlike policies we’ve hard-coded before (like “go right then down”), Q-learning enables the agent to learn from experience and gradually figure out the best actions to take in any situation.

### How Q-learning Works

Q-learning works by learning the Q-function—a way to estimate how good it is to take a particular action in a given state. Here’s a quick outline of the basic process and underlying code:

| Step                              | Code                                                                 |
|------------------------------------|----------------------------------------------------------------------|
| **Initialize Q-values**            | `q_table = np.zeros((env.observation_space.n, env.action_space.n))`  |
| **Choose action using ε-greedy policy** | `if random.random() < epsilon: action = env.action_space.sample()`<br>`else: action = np.argmax(q_table[state])` |
| **Take action and observe outcome**| `next_state, reward, done, _, _ = env.step(action)`                  |
| **Look up the best future reward** | `best_next_reward = np.max(q_table[next_state])`                     |
| **Update Q-value using the update rule** | `q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (reward + gamma * best_next_reward)` |

---

### The Update Rule Explained

Let’s look at the update rule in more detail, because this is the heart of Q-learning. The Q-value update combines *old* knowledge and *new* experience:

```python
q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (reward + gamma * best_next_reward)
```

| Component                                   | Meaning                                                                 |
|----------------------------------------------|-------------------------------------------------------------------------|
| `(1 - alpha) * q_table[state, action]`       | The old knowledge is multiplied by the factor (1-alpha) to keep this proportion of past learning. |
| `alpha * (reward + gamma * best_next_reward)`| The new experience is added based on a combination of immediate reward + estimated future reward. |
| `alpha`                                      | alpha is the learning rate — how much weight to give new info (0 = none, 1 = fast). |
| `gamma`                                      | gamma is the discount factor — how much we value future rewards over immediate ones. |

---

## Q-Learning in Action

Let’s test Q-Learning in action! Modify previous multi-episode loop to now include the epsilon-greedy policy and Q-table update. Here the most important addition is the Q-learning update of the Q-values within the inner loop. For each step, the best next reward is selected from the Q-table, and the update rule is applied to the Q-values:

```python
	while not done:
		...
		best_next_reward = np.max(q_table[next_state])
		q_table[state, action] += alpha * (reward + gamma * best_next_reward - q_table[state, action])
		...
```

While Q-learning takes time, the learned Q-table values and policy eventually lead to a solution!

## Conclusion

* A low epsilon means the agent rarely explores. It mostly chooses actions it starts with (like moving to the left) —even if they’re wrong early on. The agent gets stuck in a suboptimal strategy, and without enough exploration, it won’t try new actions often enough to discover better ones. Good exploration is essential in the beginning so the agent can learn from more diverse experiences.

* With a high alpha, Q-values change quickly based on new experiences. This can be good at first because the agent learns quickly, but it also means the values can become unstable. If rewards vary or actions lead to different results sometimes, the Q-values might swing back and forth. A high learning rate makes the agent react strongly to each outcome, which can hurt long-term consistency.


