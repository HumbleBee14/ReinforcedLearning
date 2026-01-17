# RL Agent Fine-Tuning & Evaluation Learning Plan

> **Goal**: Learn reinforcement learning concepts (RLHF, RLAIF, DPO) and use Roopik IDE as an environment for coding agent experiments - Post training evaluation/Fine-tuning and many more.


---
---

## Project Overview

### What We're Building

A **Code Agent Evaluation & Feedback System** using Roopik IDE as the execution environment. This system will:

1. Generate code using an LLM (API-based or self-hosted) [Model to be tested/trained]
2. Execute the code in Roopik's sandbox environment
3. Capture feedback (errors, screenshots, build status)
4. Use an AI judge to rate the output (RLAIF approach)
5. Log preference pairs for potential fine-tuning
6. Optionally fine-tune a small model using DPO

### Why Roopik IDE is Perfect for This

| Feature | How It Helps |
|---------|--------------|
| Dio-code Agent | Already has code generation capabilities (IDE embedded code agent) |
| Sandbox Environment | Safe execution of generated code |
| Dev Server (Vite) | Real-time preview of generated UIs |
| Screenshot Capture | Visual feedback for the reward function and use that for Multimodel LLM as judge |
| Error Logging | Objective feedback signal |
| MCP Server | API for external model communication |

### What This Project Will Teach You

- ✅ How RLHF/RLAIF works in practice
- ✅ Building reward functions for coding agents
- ✅ Creating preference datasets
- ✅ DPO (Direct Preference Optimization) basics
- ✅ Agent evaluation methodologies
- ✅ Integration of external models with your environment

---

## Key Items

### 0. Reinforcement Learning Fundamental

Before diving into RLHF/DPO, we need to understand classical RL concepts. Here's the bridge from Classical RL → LLM RL:

#### 0.1 The RL Framework (Markov Decision Process)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    THE RL LOOP (MDP)                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│    ┌─────────┐     Action (aₜ)      ┌─────────────────┐              │
│    │         │ ────────────────────>│                 │             │
│    │  AGENT  │                      │   ENVIRONMENT   │             │
│    │         │ <────────────────────│                 │             │
│    └─────────┘   State (sₜ₊₁)       └─────────────────┘              │
│                  Reward (rₜ)                                         │
│                                                                     │
│   Goal: Learn a POLICY π(a|s) that maximizes cumulative reward      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**The 5 Key Components**:

| Component | Classical RL (Game) | LLM RL (Coding Agent) |
|-----------|---------------------|----------------------|
| **State (s)** | Game screen pixels | Code + Error message |
| **Action (a)** | Move left/right/jump | Generate next token/code |
| **Reward (r)** | +1 for coin, -1 for death | +1 compile, -1 error |
| **Policy π(a\|s)** | "If enemy near, jump" | "If error, fix import" |
| **Value V(s)** | Expected future score | Expected code quality |

#### 0.2 Key RL Concepts to Understand

**1. Policy (π)**
- A mapping from states to actions
- For LLMs: The model itself IS the policy!
- π(a|s) = P(next token | current context)

```python
# Classical RL Policy (Discrete)
policy = {
    "enemy_near": "jump",
    "coin_above": "move_right",
    "pit_ahead": "stop"
}

# LLM Policy (Continuous)
def llm_policy(state: str) -> str:
    # state = "Fix this error: ImportError..."
    # action = model.generate(state)
    return generated_code
```

**2. Value Function V(s)**
- Expected total future reward from state s
- "How good is this position?"

```
V(s) = E[r₀ + γr₁ + γ²r₂ + γ³r₃ + ...]

Where:
- γ (gamma) = discount factor (0.99 typical)
- r = reward at each step
```

**3. Q-Function Q(s, a)**
- Expected total future reward from state s, taking action a
- "How good is this action in this position?"

```
Q(s, a) = r(s, a) + γ * V(s')

In LLM context:
Q("fix error", "add import numpy") = reward + future_value
```

**4. The Bellman Equation (THE CORE OF RL)**

```
V(s) = max_a [r(s,a) + γ * V(s')]

"The value of a state = best reward + discounted value of next state"
```

**5. Policy Gradient (How LLMs Learn)**

Instead of learning Q-values, we directly optimize the policy:

```
∇J(θ) = E[∇log π(a|s) * R]

"Increase probability of actions that led to high rewards"
```

#### 0.3 From Classical RL to LLM RL

```
┌─────────────────────────────────────────────────────────────────────┐
│                  EVOLUTION OF RL FOR LLMS                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Q-Learning (1989)                                               │
│     └─> Learn Q(s,a) table, select max action                       │
│                                                                     │
│  2. Deep Q-Networks (2015, DeepMind Atari)                          │
│     └─> Use neural net to approximate Q(s,a)                        │
│                                                                     │
│  3. Policy Gradient (REINFORCE, 1992)                               │
│     └─> Directly optimize the policy π(a|s)                         │
│                                                                     │
│  4. Actor-Critic (A2C, A3C)                                         │
│     └─> Policy (Actor) + Value (Critic) together                    │
│                                                                     │
│  5. Proximal Policy Optimization (PPO, 2017)                        │
│     └─> Stable policy gradient with clipping                        │
│     └─> USED IN RLHF! (InstructGPT, ChatGPT)                        │
│                                                                     │
│  6. RLHF (2022, InstructGPT)                                        │
│     └─> Human preferences → Reward Model → PPO                      │
│                                                                     │
│  7. DPO (2023)                                                      │
│     └─> Skip reward model, directly optimize on preferences         │
│                                                                     │
│  8. GRPO (2024, DeepSeek)                                           │
│     └─> Group Relative Policy Optimization                          │
│     └─> Even simpler than DPO                                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### 0.4 PPO (The Heart of RLHF)

PPO is what ChatGPT uses. Here's the core idea:

```python
# PPO Loss Function (Simplified)
def ppo_loss(old_policy, new_policy, advantage):
    ratio = new_policy / old_policy

    # Clipped objective (the key innovation!)
    clipped_ratio = clip(ratio, 1-ε, 1+ε)

    loss = -min(ratio * advantage, clipped_ratio * advantage)
    return loss

# The "Advantage" = "How much better than expected"
advantage = Q(s,a) - V(s)
```

**Why PPO works for LLMs:**
- Prevents too-large updates (model doesn't "forget" everything)
- Stable training (won't degenerate in 5 minutes)
- Works with high-dimensional action spaces (tokens!)

#### 0.5 The Reward Model (What RLHF Trains First)

```
┌─────────────────────────────────────────────────────────────────────┐
│                      REWARD MODEL TRAINING                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Human Preference Data:                                             │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │ Prompt: "Write a poem about cats"                       │        │
│  │ Response A: "Cats are fluffy and nice..." (Winner!)     │        │
│  │ Response B: "I hate cats they are evil..." (Loser)      │        │
│  └─────────────────────────────────────────────────────────┘        │
│                           │                                         │
│                           ▼                                         │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │              REWARD MODEL (Neural Net)                  │        │
│  │  Input: (prompt, response)                              │        │
│  │  Output: Scalar reward (0.0 to 1.0)                     │        │
│  │                                                         │        │
│  │  Trained with: Loss = -log(σ(r_winner - r_loser))       │        │
│  │  (Bradley-Terry Model)                                  │        │
│  └─────────────────────────────────────────────────────────┘        │
│                           │                                         │
│                           ▼                                         │
│  Now we have a "Reward Oracle" that can score ANY response!         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### 0.6 How DPO Skips the Reward Model

DPO's key insight: We don't NEED a separate reward model!

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DPO vs RLHF                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  RLHF (3 Steps):                                                    │
│  1. Train Reward Model on preferences                               │
│  2. Use Reward Model to generate rewards                            │
│  3. Use PPO to optimize policy with those rewards                   │
│                                                                     │
│  DPO (1 Step):                                                      │
│  1. Directly optimize policy on preference pairs                    │
│     (Mathematically equivalent, but simpler!)                       │
│                                                                     │
│  The DPO Loss:                                                      │
│  ────────────────────────────────────────────────────────────────   │
│  L = -log σ(β * (log π(y_w|x)/π_ref(y_w|x)                          │
│                - log π(y_l|x)/π_ref(y_l|x)))                        │
│                                                                     │
│  In English:                                                        │
│  "Make the model MORE LIKELY to produce y_w (winner)                │
│   and LESS LIKELY to produce y_l (loser)"                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### 0.7 Learning Resources for RL Foundations

**Videos (Watch in Order)**:
1. [ ] [David Silver's RL Course (DeepMind)](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ) - Lectures 1-5
2. [ ] [Spinning Up in Deep RL (OpenAI)](https://spinningup.openai.com/en/latest/)
3. [ ] [Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course/unit0/introduction)

**Papers (Read Abstracts + Key Sections)**:
1. [ ] [PPO Paper](https://arxiv.org/abs/1707.06347) - Focus on Section 3
2. [ ] [InstructGPT Paper](https://arxiv.org/abs/2203.02155) - The RLHF bible
3. [ ] [DPO Paper](https://arxiv.org/abs/2305.18290) - Section 3, 4

**Hands-On**:
1. [ ] [Gymnasium (formerly OpenAI Gym)](https://gymnasium.farama.org/) - Play with CartPole
2. [ ] [Stable Baselines3](https://stable-baselines3.readthedocs.io/) - Run PPO on simple envs
3. [ ] [TRL Library](https://huggingface.co/docs/trl/) - DPO/PPO for LLMs

#### 0.8 The Bridge: Classical RL → LLM RL

| Classical RL Concept | LLM RL Equivalent |
|---------------------|-------------------|
| State | Prompt + Conversation History |
| Action | Generate next token(s) / full response |
| Reward | Human preference / AI judge score |
| Episode | One conversation / code generation task |
| Policy | The LLM itself (π = P(token\|context)) |
| Value Function | Reward model (approximates expected reward) |
| Policy Gradient | Token-level log-prob optimization |
| PPO Clipping | KL divergence constraint (don't change too much) |

---

### 1. Reinforcement Learning from Human Feedback (RLHF)

**What it is**: A technique to fine-tune LLMs using human preferences rather than just text datasets.

**How it works**:
```
┌─────────────────────────────────────────────────────────────┐
│                        RLHF Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Step 1: Supervised Fine-Tuning (SFT)                       │
│  ─────────────────────────────────────                      │
│  Base Model → Fine-tune on high-quality examples → SFT Model│
│                                                             │
│  Step 2: Reward Model Training                              │
│  ─────────────────────────────                              │
│  Collect human preferences (A > B) → Train reward model     │
│                                                             │
│  Step 3: Policy Optimization (PPO)                          │
│  ─────────────────────────────────                          │
│  Use reward model to guide SFT model → RLHF Model           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key papers**:
- InstructGPT (OpenAI, 2022)
- Training language models to follow instructions with human feedback

### 2. Reinforcement Learning from AI Feedback (RLAIF)

**What it is**: Same as RLHF, but uses AI (another LLM) as the judge instead of humans.

**Why it matters**:
- Much cheaper than human annotation
- Scales better
- Can be more consistent
- Used by Anthropic for Constitutional AI

**How we'll use it**:
```
┌─────────────────────────────────────────────────────────────┐
│                     RLAIF for Coding                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Generator Model produces code                           │
│  2. Roopik IDE executes code                                │
│  3. Judge Model (GPT-4, Claude) evaluates:                  │
│     - Does it compile? (objective)                          │
│     - Does it render correctly? (screenshot analysis)       │
│     - Is the code clean? (style analysis)                   │
│     - Does it match the prompt? (semantic analysis)         │
│  4. Reward signal = weighted combination of above           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3. Direct Preference Optimization (DPO)

**What it is**: A simpler alternative to PPO that directly optimizes on preference data without needing a separate reward model.

**Why we'll use DPO**:
- Simpler to implement than PPO
- Works well with limited data
- No need for complex RL infrastructure
- State-of-the-art results

**The DPO Loss Function**:
```
L_DPO = -log(σ(β * (log(π(y_w|x)/π_ref(y_w|x)) - log(π(y_l|x)/π_ref(y_l|x)))))

Where:
- y_w = winning (preferred) response
- y_l = losing (rejected) response
- π = policy model
- π_ref = reference model
- β = temperature parameter
```

### 4. Constitutional AI (CAI)

**What it is**: Self-improvement through a set of principles (constitution).

**Relevance**: We can define coding principles:
- "Code should be readable"
- "Code should follow best practices"
- "Code should not have security vulnerabilities"

### 5. Outcome-Based RL

**What it is**: RL where the reward is based on the actual outcome of executing the action.

**Perfect for coding because**:
- Code either works or doesn't (objective signal)
- Compilation success/failure
- Test pass/fail
- Visual correctness

---

## Architecture Design

### High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                          SYSTEM ARCHITECTURE                       │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌─────────────────┐     ┌─────────────────────────────────────┐   │
│  │                 │     │           ROOPIK IDE                │   │
│  │  Generator      │────>│  ┌─────────────────────────────────┐│   │
│  │  Model          │     │  │  Code Execution Environment     ││   │
│  │  (API/Local)    │     │  │  - Sandbox                      ││   │
│  │                 │     │  │  - Dev Server (Vite)            ││   │
│  └────────▲────────┘     │  │  - Browser Preview              ││   │
│           │              │  └─────────────────────────────────┘│   │
│           │              │                 │                   │   │
│  ┌────────┴────────┐     │                 ▼                   │   │
│  │                 │     │  ┌─────────────────────────────────┐│   │
│  │  Preference     │◄────│  │  Feedback Collector             ││   │
│  │  Dataset        │     │  │  - Errors                       ││   │
│  │  (JSON/JSONL)   │     │  │  - Screenshots                  ││   │
│  │                 │     │  │  - Build Status                 ││   │
│  └────────▲────────┘     │  │  - Console Logs                 ││   │
│           │              │  └─────────────────────────────────┘│   │
│           │              └─────────────────────────────────────┘   │
│           │                              │                         │
│  ┌────────┴────────┐                    ▼                          │
│  │                 │     ┌─────────────────────────────────────┐   │
│  │  DPO            │     │           JUDGE MODEL               │   │
│  │  Fine-Tuning    │◄────│  (GPT-4, Claude, or self-hosted)    │   │
│  │  (Colab/Cloud)  │     │                                     │   │
│  │                 │     │  Evaluates:                         │   │
│  └─────────────────┘     │  - Functionality (0-10)             │   │
│                          │  - Code Quality (0-10)              │   │
│                          │  - Matches Prompt (0-10)            │   │
│                          └─────────────────────────────────────┘   │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
1. TASK GENERATION
   ┌─────────────────────────────────────────────────────────────┐
   │ Task: "Create a React component that displays a login form" │
   └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
2. CODE GENERATION (Generator Model)
   ┌─────────────────────────────────────────────────────────────┐
   │ function LoginForm() {                                      │
   │   const [email, setEmail] = useState('');                   │
   │   return (                                                  │
   │     <form>                                                  │
   │       <input type="email" value={email} ... />              │
   │       <button>Login</button>                                │
   │     </form>                                                 │
   │   );                                                        │
   │ }                                                           │
   └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
3. EXECUTION (Roopik IDE)
   ┌─────────────────────────────────────────────────────────────┐
   │ Sandbox/Dev Server runs the code                            │
   │ Result:                                                     │
   │   - compiled: true                                          │
   │   - errors: []                                              │
   │   - screenshot: [base64 image of rendered form]             │
   └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
4. EVALUATION (Judge Model)
   ┌─────────────────────────────────────────────────────────────┐
   │ Judge prompt:                                               │
   │ "Rate this code on: functionality, quality, prompt match"   │
   │                                                             │
   │ Judge response:                                             │
   │   - functionality: 8/10 (works but missing validation)      │
   │   - quality: 7/10 (readable but could use TypeScript)       │
   │   - prompt_match: 9/10 (matches requirements well)          │
   │   - overall: 8/10                                           │
   └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
5. REWARD CALCULATION
   ┌─────────────────────────────────────────────────────────────┐
   │ reward = 0.3 * compiled                                     │
   │        + 0.2 * (1 - error_rate)                             │
   │        + 0.2 * (judge_functionality / 10)                   │
   │        + 0.15 * (judge_quality / 10)                        │
   │        + 0.15 * (judge_prompt_match / 10)                   │
   │                                                             │
   │ reward = 0.3 + 0.2 + 0.16 + 0.105 + 0.135 = 0.80            │
   └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
6. LOG PREFERENCE (for DPO training)
   ┌─────────────────────────────────────────────────────────────┐
   │ {                                                           │
   │   "prompt": "Create a React login form...",                 │
   │   "chosen": <high_reward_response>,                         │
   │   "rejected": <low_reward_response>,                        │
   │   "chosen_reward": 0.80,                                    │
   │   "rejected_reward": 0.45                                   │
   │ }                                                           │
   └─────────────────────────────────────────────────────────────┘
```

---


---

## Resources & Papers

### Must Read Papers

1. **InstructGPT** (RLHF Foundation)
   - Paper: https://arxiv.org/abs/2203.02155
   - Summary: How OpenAI trained GPT-3 to follow instructions

2. **Constitutional AI** (RLAIF)
   - Paper: https://arxiv.org/abs/2212.08073
   - Summary: Using AI feedback instead of human feedback

3. **Direct Preference Optimization (DPO)**
   - Paper: https://arxiv.org/abs/2305.18290
   - Summary: Simpler alternative to PPO for preference learning

4. **CodeRL** (RL for Code)
   - Paper: https://arxiv.org/abs/2207.01780
   - Summary: RL specifically for code generation

### Libraries & Tools

| Tool | Purpose | Link |
|------|---------|------|
| TRL | RLHF/DPO training | https://huggingface.co/docs/trl |
| PEFT | Efficient fine-tuning | https://huggingface.co/docs/peft |
| Transformers | Model loading | https://huggingface.co/docs/transformers |
| vLLM | Fast inference | https://github.com/vllm-project/vllm |
| LangChain | LLM orchestration | https://langchain.com |

### Video Tutorials

- [ ] [RLHF Explained - Hugging Face](https://www.youtube.com/watch?v=2MBJi0ZJT4s)
- [ ] [DPO Tutorial - Weights & Biases](https://www.youtube.com/results?search_query=dpo+fine+tuning)
- [ ] [Building Coding Agents](https://www.youtube.com/results?search_query=building+coding+agents)

### Blog Posts

- [ ] [Anthropic: Constitutional AI](https://www.anthropic.com/news/constitutional-ai)
- [ ] [Hugging Face: RLHF Blog](https://huggingface.co/blog/rlhf)
- [ ] [OpenAI: Learning from Human Feedback](https://openai.com/research/learning-from-human-preferences)

---



### Focus on -

- ✅ RLHF/RLAIF concepts
- ✅ DPO implementation
- ✅ Reward function design
- ✅ Agent evaluation
- ✅ Preference data collection
- ✅ LLM fine-tuning basics

---

## Future Extensions

After this, we can onsider:

### 1. Multi-Turn Conversations
Extend to multi-turn coding assistance, not just single-shot generation.

### 2. Self-Play RL
Have the model generate, critique, and improve its own code.

### 3. Tool Use RL
Train the model to use tools (search, file operations) effectively.

### 4. Online RL
Real-time learning from user interactions in Roopik.

### 5. Custom Reward Models
Train a dedicated reward model instead of using API-based judge.

### 6. Benchmark Evaluation
Evaluate on standard coding benchmarks (HumanEval, MBPP).

---


## Questions & Decisions

### Open Questions
1. Which model to use for generation? (Phi-3, Llama-3, etc.)
2. Host model locally or use API?
3. How many preference pairs needed for meaningful DPO?

### Decisions Made
| Decision | Choice | Reasoning |
|----------|--------|-----------|
| | | |

---


