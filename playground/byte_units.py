s = "hello 牛"
b = s.encode("utf-8")
a,g,c = b[0], b[0:1], bytes([b[0]])
print(a, type(a))
print(g, type(g),len(g))
print(c, type(c),len(c))
