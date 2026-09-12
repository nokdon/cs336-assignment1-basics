import torch
import einx
from cs336_basics.tokenizer import Tokenizer

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

    def forward(self, x: torch.Tensor, #(..., seq_len, d_k)
                token_positions: torch.Tensor #(..., seq_len)
    )->torch.Tensor:
        a = x[...,::2]; b = x[...,1::2] #(..., seq_len, d_k//2)
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

def scaled_dot_product_attention(
        Q:torch.Tensor,
        K:torch.Tensor,
        V:torch.Tensor,
        mask: torch.Tensor | None = None
)->torch.Tensor:
    #Q,K shapes = (batch_size,...,seq_len,d_k)
    #V shape = (batch_size,...,seq_len,d_v)
    #mask shape = (seq_len,seq_len)

    #Attention is all you need
    S = einx.dot("... i [d_k]," \
                "... j [d_k] -> ... i j",Q,K)
    sqrt_dk = pow(Q.shape[-1],0.5)
    S = S/sqrt_dk

    #Applying Mask
    if mask is not None:
        S = torch.masked_fill(S,~mask, -torch.inf)

    #Softmax
    A = softmax(S,dim_i=-1)

    #Weighted Sum
    O = einx.dot("... i [j], ... [j] d_v -> ... i d_v",A,V)

    return O

class multihead_self_attention(torch.nn.Module):
    def __init__(self, d_model : int,
                 num_heads: int,
                 theta: float | None = None,
                 max_seq_len: int | None = None,
                 device:torch.device | None = None,
                 dtype:torch.dtype| None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model//num_heads
        self.d_v = self.d_k
        self.device = device
        self.dtype = dtype

        self.wq = Linear(self.d_model,self.d_model,device=device,dtype=dtype)
        self.wk = Linear(self.d_model,self.d_model,device=device,dtype=dtype)
        self.wv = Linear(self.d_model,self.d_model,device=device,dtype=dtype)
        self.wo = Linear(self.d_model,self.d_model,device=device,dtype=dtype)
        self.rope_obj = None
        if theta is not None and max_seq_len is not None:
            self.rope_obj = RoPE(theta,self.d_k,max_seq_len,
                                    device=device)

    def split(self, m: torch.Tensor)->torch.Tensor:
        m = einx.id("... (h d_k) -> ... h d_k",m, h=self.num_heads)
        return einx.id("... s_l h d_k -> ... h s_l d_k",m)

    def forward(self, x:torch.Tensor,
                token_positions: torch.Tensor | None = None #(...,s_l)
    )->torch.Tensor:
        #x.shape =(batch_size ... seq_len d_model)

        #Q,K,V Ininitalization
        Q = self.wq(x)
        K = self.wk(x)
        V = self.wv(x)

        #Split
        Q = self.split(Q); K=self.split(K); V=self.split(V) #(... h s_l d_k | d_v)

        #RoPE
        if token_positions is not None and self.rope_obj is not None:
            Q = self.rope_obj(Q,token_positions)
            K = self.rope_obj(K,token_positions)

        #Mask
        seq_len = x.shape[-2]
        mask = torch.ones(seq_len,seq_len,
                    device=x.device,dtype = torch.bool)
        mask = torch.tril(mask) # lower triangular

        #Scaled_dot_product_attention per head
        H = scaled_dot_product_attention(Q,K,V,mask)

        #Concat heads and @ W_O
        H = einx.id("... h s_l d_k -> ... s_l (h d_k)",H)
        return self.wo(H)

class transformer_block(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 theta: float,
                 max_seq_len:int,
                 device : torch.device | None = None,
                 dtype : torch.dtype | None = None,
                 eps: float = 1e-5
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        self.eps = eps

        self.norm_obj_1 = RMSNorm(d_model,eps,device,dtype)
        self.norm_obj_2 = RMSNorm(d_model,eps,device,dtype)
        self.attention_obj = multihead_self_attention(d_model,num_heads,
                            theta,max_seq_len,device,dtype)
        self.pwff_onj = PWFF(d_model,d_ff,device,dtype)

    def forward(self, x:torch.Tensor #(..., s_l, d)
    )->torch.Tensor:

        #subblock 1
        x_normed = self.norm_obj_1(x)
        token_positions = torch.arange(0,x.shape[-2],device=x.device)
        r_one = self.attention_obj(x_normed,token_positions)
        y_one = x + r_one

        #subblock 2
        y_one_normed = self.norm_obj_2(y_one)
        r_two = self.pwff_onj(y_one_normed)
        return y_one + r_two

class TransformerLM(torch.nn.Module):
    def __init__(self, vocab_size: int,
                 context_length: int,
                 num_layers: int,
                 d_model: int,
                 num_heads:int ,
                 d_ff:int,
                 theta:float,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None,
                 eps: float = 1e-5):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        self.device = device
        self.dtype = dtype
        self.eps = eps

        self.embedding_obj = Embedding(vocab_size,d_model,device,dtype)
        s = [transformer_block(d_model,num_heads,d_ff,
                                theta,context_length,device,dtype,eps) for _ in range(num_layers)]
        self.transformer_block_obj = torch.nn.ModuleList(s)
        self.final_norm_obj = RMSNorm(d_model,eps,device,dtype)
        self.linear_obj = Linear(d_model,vocab_size,device,dtype)

    def forward(self,x:torch.Tensor, #(B S_L)
    )->torch.Tensor:
        #Input -> Token Embedding
        e = self.embedding_obj(x)

        #Embedding -> Transformer block
        for block in self.transformer_block_obj:
            e = block(e)

        #Transformer block -> Norm
        e_norm = self.final_norm_obj(e)

        #Norm -> linear
        logits = self.linear_obj(e_norm)

        return logits #(B S_L V)

def top_p(probabilities: torch.Tensor,
          p:float
)->torch.Tensor:
    s_prob, indicies = torch.sort(probabilities,dim=-1,
                                  descending=True)
    cumsum = torch.cumsum(s_prob,dim=-1)
    position = torch.searchsorted(cumsum,p).item()
    selected = indicies[:position+1]
    result = torch.zeros_like(s_prob)
    sum = cumsum[position]
    return result.scatter_(-1,selected,s_prob)/sum



def decoding(prompt:str,
             tokenizer_obj:Tokenizer,
             model_obj: TransformerLM,
             device: torch.device,
             temperature: float,
             p: float,
             max_tokens: int):

    assert temperature > 0
    #get id of <|endoftext|>
    special_token_b = "<|endoftext|>".encode("utf-8")
    special_token_id = tokenizer_obj.bytes_to_id[special_token_b]

    #Prompt_str -> IDs
    ids = tokenizer_obj.encode(prompt)
    start = len(ids)

    #IDs -> logits
    with torch.no_grad():
        for _ in range(max_tokens):
            if len(ids) > model_obj.context_length:
                raise BufferError(f"len_ids {len(ids)}" \
                                  "> context_length")
            logits = model_obj(torch.as_tensor(ids,device=device,
                                    dtype=torch.long).unsqueeze(0))
            probab = softmax(logits/temperature,-1)
            if p != 1:
                upd_ptob = top_p(probab[0,-1,:],p)
            else:
                upd_ptob = probab[0,-1,:]
            output = torch.multinomial(upd_ptob,1)
            if output.item() == special_token_id:
                break
            ids.append(output.item())
    return tokenizer_obj.decode(ids[start:])