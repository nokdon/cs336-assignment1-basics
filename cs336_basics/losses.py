import torch
from cs336_basics.model import softmax
from collections.abc import Callable
from typing import Optional
import math
from collections.abc import Iterable
import numpy as np
import typing
import os

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

class AdamW(torch.optim.Optimizer):
    def __init__(self,
        params: torch.nn.Parameter,
        betas: tuple[float,float],
        weight_decay: float,
        lr: float = 1e-3,
        eps: float = 1e-8
):
        if lr < 0:
            raise ValueError("lr should be > 0")
        defaults = {"lr":lr,
                    "beta": betas,
                    "lambda": weight_decay,
                    "eps": eps}
        super().__init__(params,defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            b1,b2 = group["beta"]
            lambd = group["lambda"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t",1)
                m = state.get("m",torch.zeros_like(p))
                v = state.get("v",torch.zeros_like(p))

                with torch.no_grad():
                    grad = p.grad

                    #adjusted lr for t
                    lr_2 = lr*(math.sqrt(
                            1-pow(b2,t))/(1-pow(b1,t)))

                    #Apply weight decay
                    p -= lr*lambd*p

                    #update first moment estimate
                    m = b1*m+(1-b1)*grad

                    #upd second moment estimate
                    v = b2*v+(1-b2)*grad**2

                    #moment adjusted weight upd
                    p -= lr_2*m/(torch.sqrt(v)+eps)

                    state["t"] = t+1
                    state["m"] = m
                    state["v"] = v
        return loss

def learning_rate_schedule(t: int,
                           a_max: float,
                           a_min: float,
                           T_w: int,
                           T_c: int
)->float:
    lr = None

    if t<T_w:
        lr = t*a_max/T_w
    elif t >= T_w and t <= T_c:
        lr = a_min + 0.5*(1+math.cos((t-T_w)*math.pi/(T_c-T_w)))*(a_max-a_min)
    elif t > T_c:
        lr = a_min
    return lr

def gradient_clipping(params: Iterable[torch.nn.Parameter],
                      M: float
)->torch.Tensor:
    l = list(params)
    sum = 0
    for x in l:
        if x.grad is None:
            continue
        sum += torch.sum(x.grad**2)
    if type(sum) == int:
        sum = torch.zeros_like(l[0])
    grad_norm = torch.sqrt(sum)
    if grad_norm <= M:
        return grad_norm

    for p in l:
        if p.grad is None:
            continue

        with torch.no_grad():
            p.grad.mul_(M/(grad_norm+1e-6))
    return grad_norm

def data_loading(x: np.ndarray, #[int,...] IDs
                 batch_size: int,
                 context_length: int,
                 device_str: str
)->tuple[torch.Tensor,torch.Tensor]: #(batch_size, context_len)

    ids = np.random.randint(0,len(x)-context_length,batch_size)
    c = np.arange(context_length)

    X_pos = ids[:,None] + c[None,:]
    c += 1
    Y_pos = ids[:,None] + c[None,:]

    X = torch.as_tensor(x[X_pos],device=device_str,
                        dtype=torch.long)
    Y = torch.as_tensor(x[Y_pos],device=device_str,
                        dtype=torch.long)
    return X,Y

def save_checkpoint(model:torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    iteration: int,
                    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
    checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "iteration": iteration,
    }
    torch.save(checkpoint,out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer
)->int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]