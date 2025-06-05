

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
    
class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = MultiHeadAttension()
        self.ffnn = FeedForward()
        
    def farward(self, x):
        pass
    
    
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(10_000, 100)
        self.wpe = nn.Embedding(128,100)
        self.decoder = nn.ModuleList([DecoderBlock() for _ in range(8)])
        self.ln = nn.LayerNorm(100)
        self.lm_head = nn.Linear(10,1000) #1000 numbers of class that model can be !
        
    def forward(self, x):
        pass
    
    
    
    
    
