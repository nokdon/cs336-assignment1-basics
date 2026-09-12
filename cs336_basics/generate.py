import argparse
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.model import TransformerLM, decoding
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, required=True)

args = parser.parse_args()
prompt = args.prompt

#Tokenizer obj initialization
tokenizer_obj = Tokenizer.from_file("artifacts/" \
    "tokenizers/tinystories/bpe.pkl",
    ["<|endoftext|>"])

#Hyperparameters
vocab_size = 10_000
context_length = 256
d_model = 512
d_ff = 1344
theta = 10_000
num_layers = 4
num_heads = 16
betas = (0.9, 0.95)

temperature = 0.8
p = 0.9
max_tokens = 64
device = "cuda:0"

#TransformerLM obj initialization
model_obj = TransformerLM(vocab_size,context_length,
                          num_layers,d_model,
                          num_heads,d_ff,theta,
                          device=device)

#download weights from checkpoint
state = torch.load("artifacts/checkpoints/last.pt")
#upload weight state
model_obj.load_state_dict(state["model"])
#switch to eval mode
model_obj.eval()
#call decoding
output = decoding(prompt,tokenizer_obj,model_obj,
                  device,temperature,p,max_tokens)
print(output)