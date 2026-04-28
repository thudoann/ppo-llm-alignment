import torch
torch.set_num_threads(1)
torch.backends.mps.enabled = False

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

print("starting...")

import torch.nn.functional as F

print("torch imported")

from model import GPT2WithValueHead

print("model imported")

from reward import SentimentReward

print("reward imported")

from ppo import compute_advantages, ppo_loss

print("all imports done, loading models...")

def generate_text(model, tokenizer, prompt, max_new_tokens=30):
    print("  encoding prompt...")
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    generated = input_ids.clone()
    print("  starting generation loop...")
    for i in range(max_new_tokens):
        logits, _ = model(generated)
        next_token_logits = logits[0, -1, :]
        next_token = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
        generated = torch.cat([generated, next_token], dim=1)
    return generated



def get_logprobs(logits, input_ids):
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1).mean(-1)

def train():
    device = "cpu"
    print("loading GPT-2...")
    model = GPT2WithValueHead(model_name="distilgpt2").to(device)
    print("loading reward model...")
    reward_fn = SentimentReward()
    print("models loaded, starting training...")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    prompts = [
        "Today I felt", "The movie was", "I think that",
        "This experience made me", "Looking back, I",
    ]

    KL_COEF = 0.0
    N_STEPS = 200
    reward_log = []

    print(f"Training with KL_COEF={KL_COEF}")
    print("-" * 60)

    for step in range(N_STEPS):
        prompt = prompts[step % len(prompts)]

        output_ids = generate_text(
            model, model.tokenizer, prompt, max_new_tokens=20
        )
        print("generation done")

        generated_text = model.tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        )
        print(f"decoded: {generated_text[:40]}")

        reward = reward_fn.score([generated_text])[0]
        print(f"reward: {reward}")

        logits, values = model(output_ids)
        print("forward pass done")

        with torch.no_grad():
            old_logits, _ = model(output_ids)

        logprobs = get_logprobs(logits[:, :-1], output_ids[:, 1:])
        old_logprobs = get_logprobs(
            old_logits[:, :-1], output_ids[:, 1:]
        ).detach()

        kl = (logprobs - old_logprobs).mean().detach()
        penalised_reward = reward - KL_COEF * kl.item()

        advantages = compute_advantages(
            [penalised_reward], [values.mean().item()]
        )
        returns = torch.tensor(
            [penalised_reward], dtype=torch.float32
        )

        loss = ppo_loss(
            logprobs.mean().unsqueeze(0),
            old_logprobs.mean().unsqueeze(0),
            advantages,
            values.mean().unsqueeze(0),
            returns
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        if step % 20 == 0 and len(reward_log) > 0:
            avg_reward = sum(reward_log) / len(reward_log)
            print(f"Step {step:3d} | reward: {avg_reward:.3f} | "
                  f"kl: {kl.item():.4f} | loss: {loss.item():.4f}")
            print(f"  Generated: {generated_text[:80]}")
            print()

    import json
    with open("reward_log.json", "w") as f:
        json.dump(reward_log, f)
    print("\nDone. Now run with KL_COEF=0 to see reward hacking.")

if __name__ == "__main__":
    train()