from torch import nn
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decoder import DecoderBlock
from utils.tools import num_trainable_params, calculate_time
from config import GPTConfig
import torch
    
class GPT(nn.Module):
    def __init__(self, config = GPTConfig()):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embed)
        self.wpe = nn.Embedding(config.seq_len, config.n_embed)
        self.decoders = nn.Sequential(*[DecoderBlock() for _ in range(config.n_layer)])
        self.lnf= nn.LayerNorm(config.n_embed)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size) #1000 numbers of class that model can be !
        
    def forward(self, idx):
        self
    
    
    
if __name__ == "__main__":

    # Initialize configuration and model
    model = GPT(config)

    # Print the number of trainable parameters
    print(f"Number of trainable parameters: {num_trainable_params(model)}")

    # Example input tensor (batch_size=1, seq_len=config.seq_len)
    example_input = torch.randint(0, config.vocab_size, (1, config.seq_len))

    # Forward pass
    output = model(example_input)
    print(f"Output shape: {output.shape}")

