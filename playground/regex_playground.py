PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
import regex as re

with open("playground/tiny.txt", "r",encoding="utf-8") as f:
    text = f.read()

iter = re.finditer(PAT,text)

d = {}
for match in iter:
    if match.group() in d:
        d[match.group()] += 1
    else:
        d[match.group()] = 1

print(d)