import regex as re
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

#V init
voc = {k: bytes([k]) for k in range(256)}
special_tokens = ["<|endoftext|>"]
voc[len(voc)] = special_tokens[0].encode("utf-8")

#Open File
with open("playground/bpe_sample.txt", "r") as f:
    text = f.read()

#Correct Split
pattern = re.escape(special_tokens[0])
parts = re.split(pattern,text)

#Str pre-token freq table | dict[str,int]
d = {}
for word in parts:
    itterator = re.finditer(PAT,word)
    for match in itterator:
        if match.group() in d:
            d[match.group()] += 1
        else:
            d[match.group()] = 1

#Byte-level pre-token table
d_b = {}
for key,value in d.items():
    b = key.encode("utf-8")
    t = tuple([bytes([x]) for x in b])
    d_b[t] = value

#sanity-check
print(d_b)