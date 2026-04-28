import torch

def compute_advantages(rewards, values, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    advantages = torch.tensor(advantages)
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages

def ppo_loss(logprobs, old_logprobs, advantages, values, returns,
             clip_eps=0.2, vf_coef=0.5):
    ratio = torch.exp(logprobs - old_logprobs)
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    policy_loss = -torch.min(ratio * advantages,
                             clipped * advantages).mean()
    value_loss = ((values - returns) ** 2).mean()
    return policy_loss + vf_coef * value_loss