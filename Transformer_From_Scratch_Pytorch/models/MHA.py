from dataclasses import dataclass
import torch.nn as nn
import torch
from SmartAITool.core import cprint, bprint
import torch.nn.functional as F
import sys
sys.path.append(".")
from config import GPTConfig




    


class MultiHeadAttension(nn.Module):
    def __init__(self, config = GPTConfig()):
        super().__init__() 
        self.n_embed = config.n_embed
        self.n_heads = config.n_heads
        self.head_size = self.n_embed // self.n_heads
        self.qkv_proj = nn.Linear(self.n_embed, 3*self.n_embed) #(128, 100) @ (100, 300) -> (128, 300)
        self.c_proj = nn.Linear(self.n_embed, self.n_embed)
    def forward(self, x):
        print(f"dim x is:{x.shape}")
        B, T, C = x.shape  # B: batch size, T: sequence length, C: embedding dimension
        q, k, v = self.qkv_proj(x).view(B, T,3*self.n_heads, self.head_size).transpose(1,2).chunk(3, dim=-3)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1,2).contiguous().view(B, T, C)# Reshape to (B, T, C)
        return y
    
    
    
    
if __name__ == "__main__":
    mha = MultiHeadAttension()
    simple_x = torch.arange(1, 81+1, dtype=torch.float32).reshape(1, 1, 3, 27)
    cprint(f"dim simple_x is:\n{simple_x.shape}", 'green')
    cprint(f"simple_x:\n{simple_x}", 'cyan')
    bprint()
    # view_simple_x = simple_x.view(1, 27, -1)  # Reshape to (1, 3, 27)

    split_matrices = torch.split(simple_x, 9, dim=1)
    cprint(f"dim view_simple_x is:\n{split_matrices}", 'yellow')
    x = torch.randint(0, 10, (64, 128, 100), dtype=torch.float32)
    bprint()
    print(simple_x.shape)
    bprint()
    # split = simple_x.view(simple_x[0], simple_x[1], 3, 9)

    # cprint(f"dim split is:\n{split.shape}", 'yellow')
    # cprint(f"split:\n{split}", 'cyan')
    cprint(f"dim x is:{x.shape}")
    y = mha.farward(x)
    cprint(f"dim outout:\n{y.shape}", 'blue')  # Should be (64, 128, 300)
