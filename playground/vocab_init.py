d = {k: bytes([k]) for k in range(256)}

print(len(d), d[0], d[65], d[255], type(d[0]), type(d))

special_tokens = ["<|endoftext|>"]

d[len(d)] = special_tokens[0].encode("utf-8")
print(d[256])