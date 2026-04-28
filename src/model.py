import torch
torch.backends.mps.enabled = False
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class GPT2WithValueHead(nn.Module):
    def __init__(self, model_name="distilgpt2"):
        super().__init__()
        self.backbone = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        hidden_size = self.backbone.config.n_embd
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        logits = outputs.logits
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        value = self.value_head(last_hidden).squeeze(-1)
        return logits, value