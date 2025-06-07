from torch import nn
import sys
import os
import torch
from SmartAITool.core import cprint
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MHA import MultiHeadAttension
from feedforward import FeedForward
from config import GPTConfig
from utils.tools import num_trainable_params, calculate_time

class DecoderBlock(nn.Module):
    def __init__(self, config = GPTConfig()):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.mha = MultiHeadAttension()
        self.ln2 = nn.LayerNorm(config.n_embed)
        self.mlp = FeedForward()
        
    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

if __name__ == "__main__":
    x = torch.randint(0, 10, (64, 128, 100), dtype=torch.float32)
    decoder_block = DecoderBlock()
    print(num_trainable_params(decoder_block)*1000)
    cprint(calculate_time(decoder_block, (x, ), num_runs=20))
    print(decoder_block(x).shape)