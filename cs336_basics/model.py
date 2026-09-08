import torch
import einx

class Linear(torch.nn.Module):
    def __init__(self,in_features: int,
                 out_features: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        #Initialization
        W = torch.empty(out_features,in_features,device=device,dtype=dtype)
        std = pow(2/(in_features+out_features),0.5)
        self.weight = torch.nn.Parameter(torch.nn.init.trunc_normal_(
            W,mean=0,std=std,a=-3*std,b=3*std
        ))

    def forward(self, x:torch.Tensor
    )->torch.Tensor:
        W = self.weight
        return x @ W.T

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int,
                embedding_dim:int,
                device: torch.device = None,
                dtype: torch.dtype = None
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        #Initialization
        W = torch.empty(num_embeddings,embedding_dim,device=device,dtype=dtype)
        self.weight = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                W,mean=0,std=1,a=-3,b=3
            )
        )

    def forward(self, token_ids: torch.Tensor
    )->torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int,
                 eps: float = 1e-5,
                 device: torch.device=None,
                 dtype:torch.dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        self.g = torch.nn.Parameter(
            torch.ones(d_model,device=device,dtype=dtype)
        )

    def forward(self,x:torch.Tensor
    )->torch.Tensor:
        x_type = x.dtype
        x = x.to(torch.float32)
        denominator = torch.sqrt(torch.mean(x*x,dim=-1,keepdim=True)+self.eps)
        RHS = x / denominator
        g = self.g.to(torch.float32)
        result = g * RHS
        return result.to(x_type)

class PWFF(torch.nn.Module):
    def __init__(self,d_model:int,
                 d_ff:int,
                 device:torch.device | None = None,
                 dtype:torch.dtype| None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        self.w2 = Linear(d_ff,d_model,device,dtype)
        self.w1 = Linear(d_model,d_ff,device,dtype)
        self.w3 = Linear(d_model,d_ff,device,dtype)

    def forward(self,x:torch.Tensor
    )->torch.Tensor:
        assert x.shape[-1] == self.d_model

        first = self.w1.forward(x)
        second = first*torch.sigmoid(first)
        third = second * self.w3.forward(x)
        forth = self.w2.forward(third)

        return forth

class RoPE(torch.nn.Module):
    def __init__(self,
                 theta: float,
                 d_k: int,
                 max_seq_len: int,
                 device : torch.device | None= None
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
                                                #P = max_seq_len
        position = torch.arange(0,max_seq_len,device=device)[:,None] #(P,1)
        num_of_pairs = torch.arange(0,d_k//2,device=device) #(K,)
        denominator = theta**(2*num_of_pairs/d_k)
        angle = position/denominator #(P,K)
        cos = torch.cos(angle)
        sin = torch.sin(angle)

        assert sin.shape == (max_seq_len,d_k//2)
        assert cos.shape == (max_seq_len,d_k//2)

        self.register_buffer("cos",cos,persistent=False)
        self.register_buffer("sin",sin,persistent=False)

    def forward(self, x: torch.Tensor,
                token_positions: torch.Tensor
    )->torch.Tensor:
        a = x[...,::2]; b = x[...,1::2]
        cos_pos = self.cos[token_positions]
        sin_pos = self.sin[token_positions]
        first = cos_pos*a - sin_pos*b
        second = sin_pos*a + cos_pos*b
        output = torch.empty_like(x)
        output[...,::2] = first; output[...,1::2] = second
        return output

def softmax(x: torch.Tensor,
            dim_i: int
)->torch.Tensor:
    c = torch.amax(x,dim=dim_i,keepdim=True)
    exp_x = torch.exp(x-c)
    denominator = torch.sum(exp_x,dim=dim_i,keepdim=True)
    return exp_x/denominator