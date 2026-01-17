# RLHF, DPO, and RLVR

> Fine-tuning LLMs with human/AI feedback and verifiable rewards

## The Evolution

```
RLHF (2022):  Human preferences → Reward Model → PPO
DPO (2023):   Skip reward model, directly optimize preferences
RLVR (2024):  Use VERIFIABLE rewards (code execution, tests)
```

## RLHF Pipeline

1. Supervised Fine-Tuning (SFT)
2. Reward Model Training
3. PPO Optimization

## DPO: Simpler Alternative

Skip the reward model, directly optimize on preference pairs.

## RLVR: The Gold Standard for Code

When you can VERIFY correctness objectively:
- Code compiles? ✅ Verifiable
- Tests pass? ✅ Verifiable
- Output correct? ✅ Verifiable

**No human preferences needed!**

## Coming soon with P5 and P6 implementations...
