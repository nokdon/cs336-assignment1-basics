with open("playground/tiny.txt", "r", encoding="utf-8") as f:
    text = f.read()
words = text.split()
d = {}
for key in words:
    if key in d:
        d[key] += 1
    else:
        d[key] = 1

print(d)