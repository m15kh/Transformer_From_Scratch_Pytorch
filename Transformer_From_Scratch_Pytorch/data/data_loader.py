from torch.utils.data import Dataset

class TinyStoriesDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.batch_tokens = self.split_tokens(tokens, seq_len + 1)
        self.tokens = tokens
        
    def split_tokens(self, tokens , seq_len):
        n_tokens = (tokens.shape[0] // seq_len) * seq_len
        tokens = tokens[:n_tokens]
        return tokens.view(-1, seq_len)
    
    def __getitem__(self, idx):
        sample = self.batch_tokens[idx]
        
        return sample[:-1] , sample[1:]
    
    def __len__(self):
        
        return self.batch_tokens.shape[0]



if __name__ == "__main__":
    import torch

    tokenized_train_samples = torch.load('/home/fteam6/m15kh/Transformer_From_Scratch_Pytorch/checkpoints/tokenized_train_samples_vocab_10k.pt')
    tokenized_valid_samples = torch.load('/home/fteam6/m15kh/Transformer_From_Scratch_Pytorch/checkpoints/tokenized_valid_samples_vocab_10k.pt')

    train_dataset = TinyStoriesDataset(tokenized_train_samples, 256)
    x, y = train_dataset[3334]
    print(x.shape)
    print(y.shape)