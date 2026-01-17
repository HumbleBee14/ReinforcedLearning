# P6: Roopik Code Agent with RLVR

> **Goal:** Build a coding agent evaluation system using Roopik IDE with Verifiable Rewards.

## What We'll Learn

- RLVR (Reinforcement Learning from Verifiable Rewards)
- Using code execution as objective reward signal
- AI-as-judge (RLAIF) for subjective evaluation
- Building preference datasets from execution feedback
- End-to-end agent evaluation pipeline

## Key Concept: RLVR

```
RLHF:  Human says "Response A is better" (subjective)
  ↓
RLAIF: AI says "Response A is better" (cheaper but still subjective)
  ↓
RLVR:  Code compiles? Tests pass? (OBJECTIVE & VERIFIABLE!)

RLVR = The gold standard for coding agents!
```

## Why RLVR is Perfect for Code

| Reward Signal | Type | Example |
|--------------|------|---------|
| Compilation | Verifiable ✅ | Does it compile without errors? |
| Tests | Verifiable ✅ | Do unit tests pass? |
| Execution | Verifiable ✅ | Does it run without crashing? |
| Visual Output | Semi-verifiable | Does screenshot match expected? |
| Code Quality | Subjective | Is it readable? (needs AI judge) |


---
### The RL for LLMs Family Tree

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RL FOR LLMs: THE FAMILY                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  RLHF (2022)                                                        │
│  └── Human preferences → Reward Model → PPO                         │
│  └── Problem: Humans are expensive, subjective, inconsistent        │
│                                                                     │
│  RLAIF (2023)                                                       │
│  └── AI as judge instead of humans                                  │
│  └── Problem: Still subjective (just AI's opinion)                  │
│                                                                     │
│  RLVR (2024) ← YOU'RE LEARNING THIS!                                │
│  └── Verifiable/Objective rewards                                   │
│  └── Code: Compiles? Tests pass? Output correct?                    │
│  └── Math: Answer matches? Proof valid?                             │
│  └── NO HUMAN NEEDED - just run the code!                           │
│                                                                     │
│  Our Plan: RLVR + RLAIF (Best of both!)                             │
│  └── Verifiable: Compilation, execution, tests (objective)          │
│  └── AI Judge: Code quality, readability (subjective)               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

```

```
# Your Roopik reward function = RLVR!
reward = (
    0.3 * compiled +           # VERIFIABLE ✅
    0.2 * (1 - error_rate) +   # VERIFIABLE ✅
    0.3 * tests_pass +         # VERIFIABLE ✅
    0.1 * ai_judge_quality +   # RLAIF (subjective)
    0.1 * ai_judge_match       # RLAIF (subjective)
)

```


---
## Architecture

```
            ┌─────────────────────────────────────────────────┐
            │                 ROOPIK IDE                      │
            │  ┌───────────────────────────────────────────┐  │
Prompt ────>│  │ 1. Code Generation (LLM)                  │  │
            │  │ 2. Sandbox Execution                      │  │
            │  │ 3. Capture: errors, output, screenshot    │  │
            │  └───────────────────────────────────────────┘  │
            └─────────────────┬───────────────────────────────┘
                              │
                              ▼
            ┌─────────────────────────────────────────────────┐
            │            REWARD CALCULATION                   │
            │  ┌───────────────────────────────────────────┐  │
            │  │ Verifiable Rewards (RLVR):                │  │
            │  │   - compiled: +0.3                        │  │
            │  │   - no_errors: +0.2                       │  │
            │  │   - tests_pass: +0.3                      │  │
            │  │                                           │  │
            │  │ AI Judge (RLAIF):                         │  │
            │  │   - code_quality: +0.1                    │  │
            │  │   - matches_prompt: +0.1                  │  │
            │  └───────────────────────────────────────────┘  │
            └─────────────────┬───────────────────────────────┘
                              │
                              ▼
            ┌─────────────────────────────────────────────────┐
            │         PREFERENCE DATASET                      │
            │  { prompt, chosen, rejected, scores }           │
            └─────────────────────────────────────────────────┘
```

## Prerequisites

- ✅ P0-P5: All RL and LLM fine-tuning foundations
- ✅ Roopik IDE running locally

## Files (To Be Created)

```
P6/
├── README.md              # This file
├── code_generator.py      # LLM code generation
├── executor.py            # Roopik sandbox integration
├── reward_calculator.py   # RLVR + RLAIF rewards
├── ai_judge.py            # AI judge prompts
├── preference_logger.py   # Log preference pairs
└── run_evaluation.py      # Main evaluation loop
```

## Status: 🎯 The Final Goal!
