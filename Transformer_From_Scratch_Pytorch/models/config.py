from dataclasses import dataclass


@dataclass
class GPTConfig:
    
    # embed_dim : int = 512
    vocab_size = 10_000
    n_embed : int = 100
    seq_len = 128
    n_layer = 8
    n_heads : int = 5
    f_expand: float = 2.2
