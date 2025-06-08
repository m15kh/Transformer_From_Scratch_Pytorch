from torch import nn
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decoder import DecoderBlock
from utils.tools import num_trainable_params, calculate_time
from config import GPTConfig
import torch
    
class GPT(nn.Module):
    def __init__(self, config=GPTConfig()):
        super().__init__()
        self.device = config.device
        self.wte = nn.Embedding(config.vocab_size, config.n_embed)
        self.wpe = nn.Embedding(config.seq_len, config.n_embed)
        self.decoders = nn.Sequential(*[DecoderBlock() for _ in range(config.n_layer)])
        self.lnf= nn.LayerNorm(config.n_embed)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size) #1000 numbers of class that model can be !
        
    def forward(self, idx):
        # Move idx to the correct device
        idx = idx   
        B, T = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(T))
        
        for decoder in self.decoders:
            x = decoder(x)
         
         
        x = self.lnf(x)
        logics = self.lm_head(x)
        return logics
    
     
if __name__ == "__main__":

    # Initialize configuration and model
    x = torch.randint(0, 10, (64, 128), dtype=torch.int64)

    model = GPT()
    print(x.shape)
    # Print the number of trainable parameters
    print(f"Number of trainable parameters: {num_trainable_params(model)}")

    # Example input tensor (batch_size=1, seq_len=config.seq_len)

    # Forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")

