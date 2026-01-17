# P5: RLHF & DPO for LLMs

> **Goal:** Fine-tune a small LLM using RLHF/DPO principles.

## What We'll Learn

- Reward Model training
- RLHF pipeline (SFT → RM → PPO)
- DPO as simpler alternative
- Preference data format
- TRL library usage

## Key Concept

```
RLHF (3 steps):  SFT → Train Reward Model → PPO
     ↓
DPO (1 step):    Directly optimize on preferences
                 (Mathematically equivalent, simpler!)
```
```
 RLHF/DPO for LLMs
        └── Learn: TRL library, preference data format
        └── Build: Fine-tune small model with DPO
        └── Build: Your Roopik AI-as-judge system!
```

## Prerequisites

- ✅ P0-P4: All RL foundations
- ✅ Understanding of PPO

## Files (To Be Created)

```
P5/
├── README.md              # This file
├── reward_model.py        # Simple reward model
├── dpo_training.py        # DPO fine-tuning
├── preference_data.json   # Sample preference pairs
└── evaluate.py            # Evaluate fine-tuned model
```

## Status: 🔜 The Goal!
