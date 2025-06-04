import math 
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(q, k, v):
    print('shape key is:', k.shape)
    #batch size is equal to 64 it means at onece time all 64 sample calculee in parallel apprach!
    scores = q @ k.transpose(-2, -1)   # [64, 128, 100] --> [64, 100, 128] because dimention is 3d so we must ignore batch size for transpose!!
    print("score dim", scores.shape)
    scores = scores / math.sqrt(k.shape[-1])
    #---mask
    mask = torch.tril(torch.ones(q.shape[-2], q.shape[-2])) #BUG is not effciect each time create mask!
    scores.masked_fill_(mask == 0, float(-torch.inf))
    print(scores)
    #---finish masks part
    scores = scores.softmax(dim = -1)
    print("------")
    print(scores)
    z =  scores @ v
    print('Z dimension:', z.shape)
    return z

if __name__ == "__main__":
    # Example usage
    batch_size = 64 #numbers of samples that each time goes to model for calculation
    seq_len = 3 #number of tokens
    dim_model = 5 #length of each tokens

    # Random tensors for query, key, and value
    q = torch.rand(batch_size, seq_len, dim_model)
    k = torch.rand(batch_size, seq_len, dim_model)
    v = torch.rand(batch_size, seq_len, dim_model)

    # Call the function
    output = scaled_dot_product_attention(q, k, v)
