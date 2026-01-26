# RL Training for Coding Agents - MDP Design and Evaluation Frameworks

Understanding the relationship between Reinforcement Learning and Evaluation frameworks for training coding agents.

---

## The Core Questions

1. **How do we design an MDP environment for RL training in an IDE context?**
2. **What is the relationship between RL and Evals?**
3. **Should I use standard eval frameworks or build my own?**
4. **How do other RL training systems work for coding agents?**

---

## The Critical Relationship: RL ≠ Evals (But They're Connected!)

### What is RL Training?

**Reinforcement Learning** is the **training method**:
- Agent takes actions
- Environment responds with new state + reward
- Agent learns from experience
- Goal: Learn a policy that maximizes rewards

### What are Evals?

**Evaluations** are the **measurement system**:
- Test the agent's performance on specific tasks
- Measure success/failure rates
- Compare different versions of the agent
- Goal: Understand how good the agent is

### The Relationship

```
┌─────────────────────────────────────────────────────┐
│                  TRAINING LOOP (RL)                 │
│                                                     │
│  Agent → Action → Environment → Reward → Learn     │
│    ↑                                         ↓      │
│    └─────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────┘
                         ↓
                    (After training)
                         ↓
┌─────────────────────────────────────────────────────┐
│              EVALUATION (Evals)                     │
│                                                     │
│  Run agent on test tasks → Measure performance     │
│  - Success rate                                     │
│  - Time to complete                                 │
│  - Code quality                                     │
└─────────────────────────────────────────────────────┘
```

**Key insight:** 
- **RL training** happens in a **training environment** (MDP)
- **Evals** happen in a **test environment** (benchmark tasks)
- They're related but serve different purposes!

---

## Designing an MDP for Roopik IDE

### The MDP Components

For your Roopik IDE, here's how to design the MDP:

#### 1. State (S) - What the agent sees

```python
State = {
    # Code context
    "current_file": "path/to/file.tsx",
    "cursor_position": (line, column),
    "file_content": "...",
    "open_files": ["file1.tsx", "file2.css"],
    
    # IDE context
    "canvas_state": {
        "components": [...],
        "selected_element": "Button_1"
    },
    
    # Task context
    "task_description": "Add a login button to the navbar",
    "conversation_history": [...],
    
    # Project context
    "project_structure": {...},
    "dependencies": {...},
    
    # Feedback
    "last_action_result": "success" | "error",
    "error_message": "...",
    "test_results": {...}
}
```

#### 2. Action (A) - What the agent can do

```python
Actions = {
    # Code editing
    "edit_file": {
        "file": "path",
        "start_line": int,
        "end_line": int,
        "new_content": str
    },
    
    # IDE operations
    "create_canvas": {...},
    "create_component": {
        "type": "Button" | "Input" | ...,
        "props": {...}
    },
    "inspect_element": {"element_id": str},
    "get_screenshot": {},
    
    # Navigation
    "open_file": {"path": str},
    "search_codebase": {"query": str},
    
    # Execution
    "run_tests": {},
    "run_dev_server": {},
    "build_project": {},
    
    # Completion
    "submit_solution": {},
    "ask_for_help": {"question": str}
}
```

#### 3. Reward (R) - How the agent learns

```python
def calculate_reward(state, action, next_state):
    reward = 0
    
    # Task completion (big reward)
    if task_completed(next_state):
        reward += 100
    
    # Progress rewards (small rewards)
    if action_moved_closer_to_goal(state, action, next_state):
        reward += 10
    
    # Penalty for errors
    if next_state.error:
        reward -= 5
    
    # Penalty for inefficiency
    if action_was_unnecessary(action):
        reward -= 2
    
    # Code quality bonus
    if code_quality_improved(state, next_state):
        reward += 5
    
    return reward
```

#### 4. Transition Function - How the environment responds

```python
def step(state, action):
    # Execute the action in the IDE
    if action.type == "edit_file":
        result = execute_file_edit(action)
    elif action.type == "create_component":
        result = create_canvas_component(action)
    # ... etc
    
    # Update state
    next_state = update_state(state, action, result)
    
    # Calculate reward
    reward = calculate_reward(state, action, next_state)
    
    # Check if episode is done
    done = is_task_complete(next_state) or max_steps_reached()
    
    return next_state, reward, done
```

---

## The Evaluation Framework Question

### Do You Need a Standard Eval Framework?

**Short answer:** Yes, but you'll need to customize it heavily.

**Why standard frameworks help:**
- Established patterns for running benchmarks
- Metrics tracking and logging
- Comparison with other agents
- Community standards

**Why you'll need customization:**
- Your IDE has unique features (canvas, components)
- GUI interactions aren't standard
- Your tasks are domain-specific

### Recommended Approach: Hybrid

Use a standard framework as the **foundation**, customize for your needs.

---

## Standard Eval Frameworks for Coding Agents

### 1. **SWE-bench** (Software Engineering Benchmark)

**What it is:**
- Real GitHub issues as tasks
- Agent must fix bugs in actual repositories
- Measures: Can the agent solve real-world coding problems?

**Pros:**
- Industry standard
- Real-world tasks
- Large dataset

**Cons:**
- No GUI/IDE-specific features
- Focused on bug fixes, not feature creation
- Doesn't test canvas/component operations

**Relevance to Roopik:** 30% - Good for code editing, but misses IDE features

---

### 2. **HumanEval / MBPP** (Code Generation Benchmarks)

**What it is:**
- Function-level coding problems
- Agent writes code to pass tests
- Measures: Can the agent write correct code?

**Pros:**
- Simple to run
- Clear success/failure
- Fast evaluation

**Cons:**
- Too narrow (just function writing)
- No IDE interaction
- No real project context

**Relevance to Roopik:** 20% - Too basic for IDE agent

---

### 3. **WebArena / VisualWebArena** (Web Agent Benchmarks)

**What it is:**
- Agent interacts with web interfaces
- Tasks like "book a flight" or "edit a profile"
- Measures: Can the agent use GUI applications?

**Pros:**
- GUI interaction (like your canvas!)
- Multi-step tasks
- Visual understanding

**Cons:**
- Web-focused, not IDE-focused
- Different action space
- Not coding-specific

**Relevance to Roopik:** 50% - Good for GUI, but not coding

---

### 4. **Aider Benchmarks** (AI Pair Programming)

**What it is:**
- Conversational coding tasks
- Agent edits files based on natural language
- Measures: Can the agent collaborate with humans?

**Pros:**
- Conversational interface
- Real file editing
- Multi-file changes

**Cons:**
- No GUI features
- Terminal-based only
- Limited to text editing

**Relevance to Roopik:** 60% - Good for coding, missing GUI

---

### 5. **OpenHands (formerly OpenDevin)** (Coding Agent Platform)

**What it is:**
- Full agent development environment
- Supports multiple tasks (coding, browsing, shell)
- Extensible evaluation framework

**Pros:**
- Designed for coding agents
- Extensible action space
- Active community

**Cons:**
- Still evolving
- May not support canvas/GUI out of box
- Requires integration work

**Relevance to Roopik:** 70% - Best fit, but needs customization

---

## Recommended Architecture for Roopik

### Hybrid Approach: Standard Framework + Custom Extensions

```
┌─────────────────────────────────────────────────────┐
│         OpenHands/Custom Eval Framework             │
│                 (Base Layer)                        │
│  - Task management                                  │
│  - Metrics tracking                                 │
│  - Agent interface                                  │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│         Roopik-Specific Extensions                  │
│                                                     │
│  - Canvas operations                                │
│  - Component creation                               │
│  - Visual inspection                                │
│  - Screenshot comparison                            │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│              Roopik IDE (Environment)               │
│                                                     │
│  - Actual IDE instance                              │
│  - Headless or GUI mode                             │
│  - Action execution                                 │
└─────────────────────────────────────────────────────┘
```

---

## Practical Implementation Plan

### Phase 1: Build the MDP Environment (Required for RL)

**Goal:** Create a programmatic interface to Roopik IDE

```python
class RoopikEnvironment:
    def __init__(self, task):
        self.ide = RoopikIDE()  # Launch IDE instance
        self.task = task
        self.state = self._get_initial_state()
    
    def step(self, action):
        # Execute action in IDE
        result = self._execute_action(action)
        
        # Get new state
        next_state = self._get_state()
        
        # Calculate reward
        reward = self._calculate_reward(action, result)
        
        # Check if done
        done = self._is_task_complete()
        
        return next_state, reward, done, {}
    
    def reset(self):
        # Reset IDE to initial state
        self.ide.reset()
        return self._get_initial_state()
```

**This is REQUIRED for RL training!**

---

### Phase 2: Build Basic Evals (Measure Performance)

**Goal:** Create test tasks to measure agent performance

```python
# Example eval task
{
    "task_id": "create_login_button",
    "description": "Add a login button to the navbar component",
    "initial_state": {
        "project": "starter-template",
        "files": ["src/Navbar.tsx", "src/App.tsx"]
    },
    "success_criteria": {
        "button_exists": True,
        "button_text": "Login",
        "button_in_navbar": True,
        "tests_pass": True
    }
}
```

**Start with 10-20 simple tasks:**
1. Create a button component
2. Add a text input to a form
3. Change component styling
4. Fix a TypeScript error
5. Add a new route
... etc

---

### Phase 3: Integrate Standard Benchmarks (Optional)

**Goal:** Compare your agent to others

- Run SWE-bench tasks (code editing)
- Run HumanEval (code generation)
- Publish results for comparison

**This is OPTIONAL but good for credibility**

---

### Phase 4: RL Training Loop

**Goal:** Train the agent using your MDP environment

```python
# Simplified training loop
for episode in range(num_episodes):
    state = env.reset()
    done = False
    
    while not done:
        # Agent chooses action
        action = agent.choose_action(state)
        
        # Environment responds
        next_state, reward, done, _ = env.step(action)
        
        # Agent learns
        agent.learn(state, action, reward, next_state)
        
        state = next_state
```

**This uses the MDP environment you built in Phase 1**

---

## Key Insights

### 1. RL Training Needs an MDP Environment

**You MUST build:**
- A programmatic interface to Roopik IDE
- State representation
- Action execution
- Reward calculation

**This is non-negotiable for RL training!**

### 2. Evals Are Separate But Important

**Evals measure:**
- How well the trained agent performs
- Progress over time
- Comparison to baselines

**You can train without evals, but you won't know if it's working!**

### 3. Standard Frameworks Are Starting Points

**Use them for:**
- Inspiration (how to structure tasks)
- Metrics (what to measure)
- Comparison (industry benchmarks)

**But customize for:**
- Your unique IDE features
- Your specific use cases
- Your user needs

---

## Recommended Path

### For RL Training (Priority 1)

1. **Finish your current RL course** (P2-P4)
   - Understand DQN, PPO, RLHF
   - Learn how to design reward functions
   - Understand exploration vs exploitation

2. **Build the MDP environment**
   - Start simple (just file editing)
   - Add IDE features incrementally
   - Test with random agent first

3. **Design reward function**
   - Start with sparse rewards (task completion)
   - Add shaped rewards (progress indicators)
   - Iterate based on agent behavior

### For Evals (Priority 2)

1. **Study existing eval frameworks**
   - Read SWE-bench paper
   - Look at OpenHands evals
   - Understand metrics used

2. **Create Roopik-specific evals**
   - 10 simple tasks first
   - Test manually to verify
   - Automate evaluation

3. **Integrate standard benchmarks**
   - Run SWE-bench subset
   - Compare to baselines
   - Publish results

---

## Example: Simple Roopik Eval Task

```python
{
    "task_id": "roopik_001",
    "name": "Create Button Component",
    "description": "Create a new Button component in the canvas with text 'Click Me'",
    
    "initial_state": {
        "project": "blank-canvas",
        "canvas": {
            "components": []
        }
    },
    
    "actions_available": [
        "create_component",
        "edit_component_props",
        "get_screenshot",
        "submit_solution"
    ],
    
    "success_criteria": {
        "component_type": "Button",
        "component_text": "Click Me",
        "component_on_canvas": True
    },
    
    "evaluation": {
        "max_steps": 10,
        "timeout": 60,  # seconds
        "metrics": [
            "success_rate",
            "steps_to_completion",
            "time_to_completion"
        ]
    }
}
```

---

## Summary

### The Relationship

| Aspect | RL Training | Evals |
|--------|-------------|-------|
| **Purpose** | Learn a policy | Measure performance |
| **When** | During training | After training (or periodically) |
| **Environment** | Training MDP | Test tasks |
| **Required?** | Yes (for RL) | No (but highly recommended) |

### Framework Recommendation

**For MDP Environment:**
- Build custom (no standard framework fits)
- Use Gymnasium API style for familiarity

**For Evals:**
- Start with custom tasks (Roopik-specific)
- Add SWE-bench tasks later (for comparison)
- Consider OpenHands framework (most extensible)

---

## Further Reading

- [SWE-bench Paper](https://arxiv.org/abs/2310.06770) - Real-world coding benchmarks
- [WebArena Paper](https://arxiv.org/abs/2307.13854) - GUI agent evaluation
- [OpenHands Documentation](https://docs.all-hands.dev/) - Coding agent platform
- [Gymnasium Documentation](https://gymnasium.farama.org/) - RL environment API

---

## Key Takeaway

**RL and Evals are complementary:**
- **RL** = How you train the agent (needs MDP environment)
- **Evals** = How you measure the agent (needs test tasks)

**You need both!**
- Build the MDP environment first (for training)
- Build evals in parallel (to measure progress)
- Use standard frameworks as inspiration, but customize for Roopik

**The MDP environment IS the training ground. Evals are the report card.**
