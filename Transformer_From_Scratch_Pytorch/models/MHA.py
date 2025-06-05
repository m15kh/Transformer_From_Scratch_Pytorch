from dataclasses import dataclass
import torch.nn as nn



@dataclass
class GPTConfing:
    
    num_heads : int = 8
    embed_dim : int = 512
    n_embed : int = 100

    


class MultiHeadAttension(nn.Module):
    def __init__(self, config = GPTConfing()):
        super().__init__() 
        self.n_embed = config.n_embed
        self.qkv_proj = nn.Linear(self.n_embed, 3*self.n_embed)
    def farward(self, x):
        pass
