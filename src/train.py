import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer,
    AutoTokenizer, AutoModelForSequenceClassification
)
import json
import copy

# ── PPO LOSS ──
def ppo_loss(logprobs, old_logprobs, advantages, values, returns,
             clip_eps=0.2, vf_coef=0.5):
    ratio = torch.exp(logprobs - old_logprobs)
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    policy_loss = -torch.min(ratio * advantages,
                             clipped * advantages).mean()
    value_loss = ((values - returns) ** 2).mean()
    return policy_loss + vf_coef * value_loss

def get_logprobs(logits, input_ids):
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(
        2, input_ids.unsqueeze(-1)
    ).squeeze(-1).mean(-1)

# ── TRAIN ──
def train(kl_coef=0.1, n_updates=150, batch_size=16,
          ppo_epochs=4, label="run"):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # policy model
    model = GPT2WithValueHead().to(device)

    # frozen reference model — never updated
    ref_model = copy.deepcopy(model)
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    reward_fn = SentimentReward(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    prompts = [
        "Today I felt", "The movie was", "I think that",
        "This experience made me", "Looking back, I",
        "I recently discovered", "The best part was",
        "I was surprised to find",
    ]

    reward_log = []
    kl_log = []

    print(f"\nTraining with KL_COEF={kl_coef}, batch={batch_size}")
    print("-" * 60)

    for update in range(n_updates):

        # ── ROLLOUT PHASE: collect batch ──
        batch_input_ids = []
        batch_rewards = []
        batch_old_logprobs = []
        batch_values = []
        texts = []

        model.eval()
        with torch.no_grad():
            for _ in range(batch_size):
                prompt = prompts[_ % len(prompts)]
                input_ids = model.tokenizer.encode(
                    prompt, return_tensors="pt"
                ).to(device)

                output = model.backbone.generate(
                    input_ids,
                    max_new_tokens=40,
                    do_sample=True,
                    temperature=0.9,
                    pad_token_id=model.tokenizer.eos_token_id
                )

                text = model.tokenizer.decode(
                    output[0], skip_special_tokens=True
                )
                texts.append(text)
                batch_input_ids.append(output)

                # old logprobs from policy BEFORE update
                logits, value = model(output)
                lp = get_logprobs(logits[:, :-1], output[:, 1:])
                batch_old_logprobs.append(lp.detach())
                batch_values.append(value.detach())

        # score entire batch
        rewards = reward_fn.score(texts)

        # compute KL vs frozen reference model
        kl_penalties = []
        with torch.no_grad():
            for i, ids in enumerate(batch_input_ids):
                ref_logits, _ = ref_model(ids)
                pol_logits, _ = model(ids)
                ref_lp = get_logprobs(ref_logits[:, :-1], ids[:, 1:])
                pol_lp = get_logprobs(pol_logits[:, :-1], ids[:, 1:])
                kl = (pol_lp - ref_lp).mean().item()
                kl_penalties.append(kl)

        # penalised rewards
        penalised_rewards = [
            r - kl_coef * kl
            for r, kl in zip(rewards, kl_penalties)
        ]

        avg_reward = sum(rewards) / len(rewards)
        avg_kl = sum(kl_penalties) / len(kl_penalties)
        reward_log.append(avg_reward)
        kl_log.append(avg_kl)

        # ── PPO UPDATE PHASE: multiple epochs ──
        model.train()
        for epoch in range(ppo_epochs):
            total_loss = 0
            for i, ids in enumerate(batch_input_ids):
                logits, values = model(ids)
                lp = get_logprobs(logits[:, :-1], ids[:, 1:])

                advantage = torch.tensor(
                    [penalised_rewards[i] - batch_values[i].mean().item()],
                    dtype=torch.float32
                ).to(device)

                returns = torch.tensor(
                    [penalised_rewards[i]],
                    dtype=torch.float32
                ).to(device)

                loss = ppo_loss(
                    lp.mean().unsqueeze(0),
                    batch_old_logprobs[i].mean().unsqueeze(0),
                    advantage,
                    values.mean().unsqueeze(0),
                    returns
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 0.5
                )
                optimizer.step()
                total_loss += loss.item()

        if update % 10 == 0:
            print(f"Update {update:3d} | reward: {avg_reward:.3f} | "
                  f"kl: {avg_kl:.4f} | "
                  f"loss: {total_loss/batch_size:.4f}")
            print(f"  → {texts[0][:80]}")
            print()

    with open(f"reward_log_{label}.json", "w") as f:
        json.dump({"rewards": reward_log, "kl": kl_log}, f)
    print(f"Saved reward_log_{label}.json")
    return reward_log, kl_log

# ── RUN BOTH EXPERIMENTS ──
log_kl_r, log_kl_k = train(
    kl_coef=0.5, n_updates=150,
    batch_size=16, ppo_epochs=4, label="kl05"
)
log_no_r, log_no_k = train(
    kl_coef=0.0, n_updates=150,
    batch_size=16, ppo_epochs=4, label="kl00"
)

# ── PLOT ──
import matplotlib.pyplot as plt

def smooth(values, window=5):
    return [
        sum(values[max(0,i-window):i+1]) /
        len(values[max(0,i-window):i+1])
        for i in range(len(values))
    ]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

ax1.plot(smooth(log_kl_r), color="#1D9E75",
         label="PPO with KL penalty (β=0.5)")
ax1.plot(smooth(log_no_r), color="#E05A3A",
         label="PPO without KL penalty")
ax1.set_xlabel("Update")
ax1.set_ylabel("Average reward (batch)")
ax1.set_title("Reward over training")
ax1.legend()

ax2.plot(smooth(log_kl_k), color="#1D9E75",
         label="KL with penalty (β=0.5)")
ax2.plot(smooth(log_no_k), color="#E05A3A",
         label="KL without penalty")
ax2.set_xlabel("Update")
ax2.set_ylabel("KL divergence from reference")
ax2.set_title("Policy drift from reference model")
ax2.legend()

plt.tight_layout()
plt.savefig("reward_curves.png", dpi=150)
plt.show()
print("Saved reward_curves.png")