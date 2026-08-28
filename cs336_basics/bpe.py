import regex as re
#Parameters
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(input_path:str,
              vocab_size:int,
              special_tokens:list[str]
) -> tuple[dict[int, bytes],list[tuple[bytes, bytes]]]:

    

    #V init
    vocab = {k: bytes([k]) for k in range(256)}
    for i in range(len(special_tokens)):
        vocab[len(vocab)] = special_tokens[i].encode("utf-8")

    if vocab_size < len(vocab):
        raise ValueError("vocab_size < that required initial vocab")

    #File opening
    with open(input_path, "r",encoding="utf-8") as f:
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
        parts = [text]
    #Count str words
    pretoken_counts = {}
    for word in parts:
        itterator = re.finditer(PAT,word)
        for match in itterator:
            if match.group() in pretoken_counts:
                pretoken_counts[match.group()] += 1
            else:
                pretoken_counts[match.group()] = 1

    #Byte representation of pretoken_counts
    d_b = {}
    for key,value in pretoken_counts.items():
        s_b = key.encode("utf-8")
        t = tuple([bytes([x]) for x in s_b])
        d_b[t] = value

    token_seq_counts = {key: value for key,value in d_b.items()} #global dict
    merges = [] #list[tuple[bytes,bytes]]

    #Iteration till |V| < V_size
    while len(vocab) < vocab_size:
        #count pair frequency
        adj = {}
        for x,freq in token_seq_counts.items():
            for start, end in zip(x[:-1],x[1:]):
                if (start,end) in adj:
                    adj[(start,end)] += freq
                else:
                    adj[(start,end)] = freq

        #Merge
        if not adj: #adj void case
            break
        win_pair = max(adj,key=
                    lambda x: (adj[x],x))
        merged_token = win_pair[0]+win_pair[1]
        vocab[len(vocab)] = merged_token #add merged token to vocab
        merges.append(win_pair)

        updated_seq_counts = {} #dict after merge

        for key,value in token_seq_counts.items():
            i = 0
            updated_sequence = []
            while i < len(key):
                if i+1 <len(key) and (key[i],key[i+1]) == win_pair:
                    updated_sequence.append(merged_token)
                    i+=2
                else:
                    updated_sequence.append(key[i])
                    i+=1
            updated_seq_counts[tuple(updated_sequence)] = value
        token_seq_counts = updated_seq_counts #work with a new dict in next iteration

    return vocab,merges