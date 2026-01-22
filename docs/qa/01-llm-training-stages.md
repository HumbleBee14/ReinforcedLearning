# LLM Training Stages - Pre-training, Mid-training, and Post-training

---

## Questions

1. **What is mid-training?** Why does it exist as a separate stage?
2. **Why can't we do multimodal/multilingual training during pre-training itself?**
3. **How does training increase context length capacity?** Isn't that an architecture thing, not a training thing?

---

## The Three Training Stages Explained

### Pre-training (The Foundation)

**What it is:**
- Training a model from scratch on massive amounts of text data
- Goal: Learn language patterns, world knowledge, and next-word prediction
- Input: Raw internet text (books, websites, code, etc.)
- Output: A "base model" that can predict the next word

**Example:**
```
Input:  "The capital of France is"
Output: "Paris" (predicted next word)
```

**Key characteristics:**
- Learns general language understanding
- Learns broad world knowledge
- Not specialized for any specific task
- Not aligned with human preferences
- **EXTREMELY EXPENSIVE** (millions of dollars, months of training)

**Models at this stage:** GPT-3 base, LLaMA base, Mistral base

---

### Mid-training (Specialization & Enhancement)

**What it is:**
- **Continued training** on the pre-trained model with **specific objectives**
- Goal: Add capabilities that are expensive or impractical during pre-training
- Input: Curated, high-quality datasets for specific enhancements
- Output: An enhanced base model with new capabilities

**Why it exists as a separate stage:**

#### Reason 1: Cost Efficiency
Pre-training is INSANELY expensive. You don't want to restart from scratch just to add a new language or modality.

**Example:**
- Pre-training GPT-4: ~$100 million, 6 months
- Mid-training to add Japanese: ~$1 million, 1 week
- **Retraining from scratch with Japanese included: $100 million again!**

#### Reason 2: Modular Capabilities
You can add capabilities incrementally without breaking what already works.

**Common mid-training objectives:**

| Capability | What it does | Why not in pre-training? |
|------------|--------------|--------------------------|
| **Multimodal** | Add vision, audio | Pre-training data is mostly text; adding images later is cheaper |
| **Multilingual** | Add new languages | Can focus on high-quality language data instead of random internet text |
| **Context Length** | Increase from 4K → 32K tokens | Pre-training with long context is extremely expensive |
| **Domain Knowledge** | Add medical/legal expertise | Requires curated, high-quality domain data |

#### **Real-world example:**
- **LLaMA 2** (base): Pre-trained on general text
- **LLaMA 2 Long** (mid-trained): Context extended from 4K → 32K tokens
- **Code LLaMA** (mid-trained): Specialized for code generation

---

### Post-training (Alignment & Usability)

**What it is:**
- Fine-tuning the model to be helpful, harmless, and honest
- Goal: Make the model follow instructions and align with human values
- Input: Instruction datasets, human feedback
- Output: A chat-ready assistant (like ChatGPT)

**Techniques:**
- **Supervised Fine-Tuning (SFT):** Train on instruction-response pairs
- **RLHF (Reinforcement Learning from Human Feedback):** Train with human preferences
- **DPO (Direct Preference Optimization):** Simpler alternative to RLHF

**Example transformation:**

| Stage | Input | Output |
|-------|-------|--------|
| **Pre-trained** | "Write a poem about" | "cats and dogs and birds and..." (just continues) |
| **Post-trained** | "Write a poem about" | "Sure! Here's a poem about nature:..." (follows instruction) |

---

## Deep Dive: Why Can't We Do Everything in Pre-training?

### The Problem: Pre-training is a "Shotgun Approach"

Pre-training uses **random internet data** at massive scale:
- Trillions of tokens
- Mostly English text
- Variable quality (good and bad content mixed)
- Costs millions of dollars

**Why this makes specialization hard:**

1. **Dilution Effect**
   - If you add 1% Japanese data to pre-training, the model learns Japanese poorly
   - You'd need 50% Japanese data to learn it well → but then English suffers!

2. **Context Length Cost**
   - Training with 32K context is **16x more expensive** than 2K context
   - Memory usage: O(n²) where n = context length
   - Pre-training with long context would cost hundreds of millions

3. **Quality vs. Quantity Trade-off**
   - Pre-training: "Eat everything, learn general patterns"
   - Mid-training: "Eat only the best food, learn specific skills"

---

## Deep Dive: How Does Training Increase Context Length?

>  Isn't it something more determined and decided by the architecture itself?

### Context Length is Architecture + Training

Context length is **initially determined by architecture**, but training can **extend** it beyond the original design.

### The Architecture Part

The Transformer architecture has a **positional encoding** system that tells the model "where" each token is in the sequence.

**Original design (e.g., GPT-3):**
```python
max_position_embeddings = 2048  # Can only handle 2048 tokens
```

The model literally has a lookup table:
```
Position 0    → Embedding vector [0.1, 0.3, ...]
Position 1    → Embedding vector [0.2, 0.4, ...]
...
Position 2047 → Embedding vector [0.9, 0.1, ...]
Position 2048 → ??? (doesn't exist!)
```

### The Training Part

You can **extend** the context length through **continued training** without changing the architecture!

#### Method 1: Position Interpolation (Most Common)

**The trick:** Squeeze the position encodings to fit more tokens.

**Original (2K context):**
```
Token 1    → Position 0
Token 2    → Position 1
...
Token 2048 → Position 2047
```

**Extended (8K context via interpolation):**
```
Token 1    → Position 0.00
Token 2    → Position 0.25
Token 3    → Position 0.50
...
Token 8192 → Position 2047
```

**Then train the model** to understand that positions can be fractional!

#### Method 2: Extrapolation (Less Common)

Train the model to **predict** what positions beyond 2048 should look like.

**Why this works:**
- The model learns that "position 2048.5" is similar to "position 2047"
- Through training, it generalizes the pattern

#### Method 3: RoPE Scaling (Modern Approach)

Used in LLaMA and modern models. Adjusts the **frequency** of positional encodings.

**Analogy:**
- Original: A ruler with marks every 1 cm (fits 100 cm)
- Extended: Same ruler, but marks every 0.25 cm (now fits 400 cm)
- Training: Teach the model to read the new ruler

### Why This Requires Training (Not Just Architecture Change)

Even though you're modifying the architecture slightly, the model needs to **learn** how to use the new positions:

1. **Attention patterns change** with longer context
2. **Information flow** is different across 32K tokens vs. 2K tokens
3. **The model must learn** which distant tokens are relevant

**Training data for context extension:**
- Long documents (books, research papers)
- Code repositories (naturally long)
- Conversations with long history

**Training time:**
- Much shorter than pre-training (days vs. months)
- Uses the already-learned language knowledge
- Just adapts to longer sequences

---

## Summary: Why Three Stages?

| Stage | Purpose | Cost | Duration | Analogy |
|-------|---------|------|----------|---------|
| **Pre-training** | Learn language & world knowledge | $$$$$$ | Months | Elementary school (learn to read) |
| **Mid-training** | Add specialized capabilities | $$$ | Days-Weeks | High school (learn specific subjects) |
| **Post-training** | Align with human preferences | $$ | Days | Job training (learn to be helpful) |

### The Economic Reality

```
Pre-training:  $100,000,000  (one time, foundational)
Mid-training:  $1,000,000    (per capability, modular)
Post-training: $100,000      (per version, frequent)
```

**Why not do everything in pre-training?**
- Would cost $500M+ to include everything
- Would take years to train
- Can't iterate or fix mistakes easily
- Can't add new capabilities without retraining everything

**The modular approach:**
- Train once, specialize many times
- Add capabilities incrementally
- Iterate quickly on alignment
- Cost-effective experimentation

---

## Related Concepts

- **Transfer Learning:** Using pre-trained models as starting points (same principle as mid-training)
- **Curriculum Learning:** Training in stages from easy to hard (similar to pre→mid→post progression)
- **Positional Encoding:** How Transformers understand sequence order
- **Attention Mechanisms:** How models focus on relevant information across long sequences
- **Fine-tuning vs. Training from Scratch:** When to use each approach

---

## Further Reading

- [LLaMA 2 Long Paper](https://arxiv.org/abs/2309.16039) - Context length extension via training
- [RoFormer Paper](https://arxiv.org/abs/2104.09864) - RoPE (Rotary Position Embedding)
- [Extending Context Window via Position Interpolation](https://arxiv.org/abs/2306.15595)

---

## Key Takeaway

**Mid-training exists because:**
1. Pre-training is too expensive to redo
2. Specialization requires focused, high-quality data
3. Context extension requires architectural tweaks + training
4. Modular approach is more economical and flexible

**Context length is:**
- ✅ Initially limited by architecture (position embeddings)
- ✅ Extendable through training (position interpolation/extrapolation)
- ✅ Requires both architectural modification AND training data

Think of it like this:
- **Architecture** = The size of your bookshelf
- **Training** = Learning how to organize and find books on a bigger bookshelf

You can build a bigger bookshelf (architecture), but you still need to practice using it (training)!
