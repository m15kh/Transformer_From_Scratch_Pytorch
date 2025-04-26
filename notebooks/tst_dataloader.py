from torch.utils.data import TensorDataset
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from Transformer_From_Scratch_Pytorch.data.data_loader import TinyStoriesDataset
# Load tokens from pytorch file
tokenized_train_samples = torch.load('/home/fteam6/m15kh/Transformer_From_Scratch_Pytorch/checkpoints/tokenized_train_samples_vocab_10k.pt')
tokenized_valid_samples = torch.load('/home/fteam6/m15kh/Transformer_From_Scratch_Pytorch/checkpoints/tokenized_valid_samples_vocab_10k.pt')



train_dataset = TinyStoriesDataset(tokenized_train_samples, 256)
x, y = train_dataset[3334]
print(x.shape)
print(y.shape)