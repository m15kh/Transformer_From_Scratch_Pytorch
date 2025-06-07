from torch import nn
import torch.nn.functional as F
import sys
import torch
sys.path.append(".")
from config import GPTConfig


def num_trainable_params(model):
  nums = sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6
  return nums


class FeedForward(nn.Module):
    def __init__(self, config = GPTConfig()):
        super().__init__()
        self.n_embed = config.n_embed
        self.f_expand = config.f_expand
        self.up_proj = nn.Linear(self.n_embed, int(self.n_embed * self.f_expand))
        self.down_proj = nn.Linear(int(self.n_embed * self.f_expand), self.n_embed)
        
    def forward(self, x):
        return self.down_proj(F.gelu(self.up_proj(x)))

    
if __name__ == "__main__":
    x = torch.randint(0, 10, (64, 128, 100), dtype=torch.float32)
    mlp = FeedForward()
    print(num_trainable_params)
    print(mlp(x).shape)
