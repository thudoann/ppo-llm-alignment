# PPO for LLM Alignment — from scratch

A minimal PyTorch implementation of **Proximal Policy Optimization (PPO)** applied to large language model alignment. GPT-2 learns to generate positive-sentiment text using a frozen sentiment classifier as the reward signal — trained entirely from scratch without any alignment libraries.

Built as part of my research preparation for a PhD in post-training and alignment for LLMs.

---

## What & Why

Large language models are powerful but not inherently aligned with human preferences. **Reinforcement Learning from Human Feedback (RLHF)** — the technique behind ChatGPT and Claude — uses PPO to fine-tune LLMs toward desired behaviors using a reward signal.

This project reimplements the core PPO loop from the [InstructGPT paper (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) on a small, interpretable task:

> *Can we train GPT-2 to generate more positive text using only a reward signal — and what happens when we remove the safety constraint?*

---

## Method

The pipeline has three components:

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   Prompt ──► GPT-2 + Value Head ──► Generated text     │
│                    │                       │            │
│                    │              Sentiment Classifier  │
│                    │                       │            │
│                    └──── PPO Update ◄── Reward (0-1)   │
│                              │                          │
│                         KL Penalty                      │
│                    (prevents reward hacking)            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Policy model** — `distilgpt2` with an added linear value head. One forward pass returns both logits (for the policy) and a scalar value estimate (for advantage computation).

**Reward model** — `distilbert-base-uncased-finetuned-sst-2-english`, frozen throughout training. Returns a score in [0, 1] where 1 = maximally positive sentiment.

**PPO update** — standard clipped policy gradient with:
- Generalised Advantage Estimation (GAE)
- Value function loss
- KL divergence penalty to prevent policy collapse

---

## Key Finding: Reward Hacking vs Stable Alignment

The central experiment compares two runs:

| | PPO with KL penalty (β=0.1) | PPO without KL penalty |
|---|---|---|
| Average reward | stable, increasing | high but erratic |
| Text quality | coherent, diverse | repetitive, degenerate |
| Behavior | aligned | reward hacking |

**Without the KL penalty**, the model quickly learns to exploit the reward function — generating repetitive high-scoring phrases rather than genuinely positive, diverse text. This is a concrete demonstration of an alignment failure mode.

**With the KL penalty**, the model is regularised toward the original GPT-2 distribution, producing stable and coherent outputs while still improving sentiment scores.

This directly illustrates why KL regularisation is a core component of production RLHF systems (InstructGPT, Claude, Gemini).

---

## Results

![Reward curves](results/reward_curves.png)

Sample outputs after 500 training steps:

**With KL penalty (β=0.1):**
```
Prompt: "Today I felt"
→ "Today I felt really happy about the way things turned out.
   It was a wonderful experience overall."

Prompt: "The movie was"
→ "The movie was absolutely brilliant. I loved every moment
   of it and would highly recommend it."
```

**Without KL penalty (reward hacking):**
```
Prompt: "Today I felt"
→ "Today I felt great great great great great great great
   great great great great great great great great."

Prompt: "The movie was"
→ "The movie was good good good good good good good good
   good good good good good good good good good."
```

---

## Repo Structure

```
ppo-llm-alignment/
├── src/
│   ├── model.py       # GPT-2 + value head
│   ├── reward.py      # sentiment reward model
│   ├── ppo.py         # PPO loss + GAE
│   └── train.py       # main training loop
├── notebooks/
│   └── results.ipynb  # reward curves + analysis
├── results/
│   ├── reward_curves.png
│   ├── reward_log_kl01.json
│   └── reward_log_kl00.json
└── requirements.txt
```

---

## Reproducing

```bash
git clone https://github.com/thudoann/ppo-llm-alignment
cd ppo-llm-alignment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# train with KL penalty
python src/train.py --kl_coef 0.1 --steps 500 --label kl01

# train without KL penalty (reward hacking)
python src/train.py --kl_coef 0.0 --steps 500 --label kl00
```

Runs on CPU (~15 min) or GPU (~2 min on Colab T4).

---

## References

- Schulman et al. (2017). [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- Ouyang et al. (2022). [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- Christiano et al. (2017). [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741)