s = "hello 牛"
b = s.encode("utf-8")
print(s,b, type(s),type(b),len(s),len(b), "\n")
for x in b:
    print(x)