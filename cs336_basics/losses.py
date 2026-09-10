import torch
from cs336_basics.model import softmax

def cross_entropy(o: torch.Tensor, #(batch_size, seq_len, vocab_size)
                  targets: torch.Tensor
)->torch.Tensor:     #-l_t   +    log Z
    #li = -o_i[x_i+1] +с + log sum_V [exp[o_i[v]-c]]
    c = torch.amax(o,dim=-1,keepdim=True)

    l_t = torch.gather(o,-1,targets.unsqueeze(-1)) #(batch_size,seq_len,1)
    l_t = l_t - c
    z = torch.exp(o-c) #(batch_size,seq_len,vocab_size)
    Z = torch.sum(z,dim=-1,keepdim=True) #(batch_size,seq_len,1)
    log_Z = torch.log(Z)
    l = -l_t + log_Z #(batch_size,seq_len,1)
    return torch.mean(l)

