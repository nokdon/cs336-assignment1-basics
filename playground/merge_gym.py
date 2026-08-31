import regex as re
from time import perf_counter
#Parameters
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
timer_start = perf_counter()
#special tokens
special_tokens = ["<|endoftext|>"]
vocab_size = 500

#V init
vocab = {k: bytes([k]) for k in range(256)}
for i in range(len(special_tokens)):
    vocab[len(vocab)] = special_tokens[i].encode("utf-8")

if vocab_size < len(vocab):
    raise ValueError("vocab_size < that required initial vocab")

#File opening
with open("tests/fixtures/corpus.en", "r",encoding="utf-8") as f:
    text = f.read()

#sort soecial_tokens in descending order
sorted_s_t = sorted(special_tokens,key=len,reverse=True)

#Split by special tokens
escaped_s_t = []
for i in range(len(sorted_s_t)):
    escaped_s_t.append(re.escape(sorted_s_t[i]))
if len(escaped_s_t) > 0:
    pattern = "|".join(escaped_s_t)
    parts = re.split(pattern,text) #text between special tokens
else:
    parts = [text] #list[str]


#Count str words
pretoken_counts = {} #dict[str, int]
for word in parts:
    itterator = re.finditer(PAT,word)
    for match in itterator:
        if match.group() in pretoken_counts:
            pretoken_counts[match.group()] += 1
        else:
            pretoken_counts[match.group()] = 1


#Byte representation of pretoken_counts
R = {}; i = 0 #dict[int,tuple[bytes,...]]
d_b = {} #dict[tuple[bytes,...],int]
for key,value in pretoken_counts.items():
    s_b = key.encode("utf-8")
    t = tuple([bytes([x]) for x in s_b])
    d_b[t] = value
    R[i] = t; i+=1


M = {i : d_b[R[i]] for i in range(len(R))} #dict[int,int]
merges = [] #list[tuple[bytes,bytes]]


I = {} #dict[tuple[bytes,bytes], set[int]]
adj = {} #dict[tuple[bytes, bytes], int]
for i,x in R.items():
    for start, end in zip(x[:-1],x[1:]):
        if (start,end) in adj:
            adj[(start,end)] += M[i]
        else:
            adj[(start,end)] = M[i]

        if (start,end) not in I:
            I[(start,end)] = set()
        I[(start,end)].add(i)

#Iteration till |V| < V_size
while len(vocab) < vocab_size:
    #-----------------------------------------------------------------
    #Merge
    if not adj: #adj void case
        break
    win_pair = max(adj,key=
                lambda x: (adj[x],x))
    merged_token = win_pair[0]+win_pair[1]
    vocab[len(vocab)] = merged_token #add merged token to vocab
    merges.append(win_pair)
    #-----------------------------------------------------------------

    regions = I[win_pair].copy()

    for j in regions:
        seq = R[j]; i = 0
        updated_sequence = []
        while i < len(seq):
            if i+1 <len(seq) and (seq[i],seq[i+1]) == win_pair:
                updated_sequence.append(merged_token)
                i+=2
            else:
                updated_sequence.append(seq[i])
                i+=1
        #clean frequencies in the region
        for start,end in zip(R[j][:-1],R[j][1:]):
            pair = (start,end)
            adj[pair] -= M[j]

            if adj[pair] == 0:
                del adj[pair]

            if pair in I:
                I[pair].discard(j)

                if not I[pair]:
                    del I[pair]

        #update frequencies after merge
        R[j] = tuple(updated_sequence) #update R after merge
        for start,end in zip(R[j][:-1],R[j][1:]):
            pair = (start,end)
            if pair in adj:
                adj[pair] += M[j]
            else:
                adj[pair] = M[j]

            if pair not in I:
                I[pair] = set()
            I[pair].add(j)

    #-----------------------------------------------------------------

elapsed = perf_counter() - timer_start
print(elapsed)