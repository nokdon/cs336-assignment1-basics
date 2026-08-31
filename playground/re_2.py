PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
import regex as re

s = "hello 牛 hello"

d = {}
iterrator = re.finditer(PAT,s)

for match in iterrator:
    if match.group() in d:
        d[match.group()] += 1
    else:
        d[match.group()] = 1

d2 = {}
for key, value in d.items():
    b_key = key.encode("utf-8")
    t = tuple([bytes([x]) for x in b_key])
    d2[t] = value

print(d,d2)