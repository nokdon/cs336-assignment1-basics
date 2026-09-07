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
