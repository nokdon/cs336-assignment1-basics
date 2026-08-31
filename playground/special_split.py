import regex as re
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
s = "hello cat<|endoftext|>hello dog<|endoftext|>hello fish"
special_tokens = ["<|endoftext|>"]

pattern = re.escape(special_tokens[0])
parts = re.split(pattern,s)

d = {}
for word in parts:
    itterator = re.finditer(PAT,word)
    for match in itterator:
        if match.group() in d:
            d[match.group()] += 1
        else:
            d[match.group()] = 1

print(d)