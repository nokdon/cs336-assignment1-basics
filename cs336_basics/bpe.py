import os
from typing import BinaryIO
import regex as re
from concurrent.futures import ProcessPoolExecutor
#Parameters
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def count_pretokens(text: str,
                    special_tokens: list[str]
)->dict[str,int]:
    sorted_s_t = sorted(special_tokens,key=len,reverse=True)
    
    #Split corpus at special token boundries
    escaped_s_t = []
    for i in range(len(sorted_s_t)):
        escaped_s_t.append(re.escape(sorted_s_t[i]))
    if len(escaped_s_t) > 0:
        pattern = "|".join(escaped_s_t)
        parts = re.split(pattern,text) #text between special tokens
    else:
        parts = [text]
    ##Count pre-token occurrences across all corpus segments
    pretoken_counts = {}
    for word in parts:
        itterator = re.finditer(PAT,word)
        for match in itterator:
            if match.group() in pretoken_counts:
                pretoken_counts[match.group()] += 1
            else:
                pretoken_counts[match.group()] = 1

    return pretoken_counts

def worker(input_path:str,
           start: int,
           end: int,
           special_tokens: list[str]
) -> dict[str,int]:
    #Read the UTF-8 training corpus

    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end-start).decode("utf-8",errors="ignore")

    pretoken_counts = count_pretokens(text,special_tokens)
    
    return pretoken_counts

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def count_file_pretokens(input_path: str,
                         num_processes: int,
                         special_tokens: list[str]
)->dict[str,int]:
    with open(input_path, "rb") as f:
            global_chunk_pretoken_count = {}
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []    
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            
            future = executor.submit(worker,input_path,start,end,special_tokens)
            futures.append(future)


        for future in futures:
            chunk_pretoken_count = future.result()
        
            for key,value in chunk_pretoken_count.items():
                if key in global_chunk_pretoken_count:
                    global_chunk_pretoken_count[key] += value
                else:
                    global_chunk_pretoken_count[key] = value
    return global_chunk_pretoken_count

def train_bpe(input_path:str,
              vocab_size:int,
              special_tokens:list[str],
              num_processes: int =  1
) -> tuple[dict[int, bytes],list[tuple[bytes, bytes]]]:

    """Train a byte-level BPE tokenizer from a UTF-8 text file.

    Args:
        input_path: Path to the training text
        vocab_size: Maximum final vocabulary size
        special_tokens: Tokens added to the vocabulary and excluded from merge statistics

    Returns:
        The vocabulary and the ordered list of learned merges
    """

    # seq_by_id: stable sequence ID -> current BPE token sequence
    # freq_by_id: stable sequence ID -> fixed corpus multiplicity
    # adj: adjacent token pair -> weighted global occurrence count
    # pair_to_seq_ids: adjacent token pair -> IDs of sequences currently containing it

    # Initialize the vocabulary with all 256 byte tokens and the provided special tokens
    vocab = {k: bytes([k]) for k in range(256)}
    for i in range(len(special_tokens)):
        vocab[len(vocab)] = special_tokens[i].encode("utf-8")

    if vocab_size < len(vocab):
        raise ValueError("vocab_size < that required initial vocab")

    
    pretoken_counts = count_file_pretokens(
        input_path,num_processes,special_tokens
    )

    #Assign each pre-token sequence a stable ID
    #seq_by_id maps sequence ID -> current BPE representation
    seq_by_id: dict[int,tuple[bytes,...]] = {}; i = 0
    #Byte representation of pretoken_counts
    d_b = {}
    for key,value in pretoken_counts.items():
        s_b = key.encode("utf-8")
        t = tuple([bytes([x]) for x in s_b])
        d_b[t] = value
        seq_by_id[i] = t; i+=1
    #freq_by_id maps sequence ID -> fixed corpus multiplicity
    freq_by_id: dict[int,int] = {i : d_b[seq_by_id[i]] for i in range(len(seq_by_id))}
    merges: list[tuple[bytes,bytes]] = []

    #pair_to_seq_ids maps pair -> sequence IDs in which the pair currently occurs
    pair_to_seq_ids: dict[tuple[bytes,bytes], set[int]] = {}
    #adj maps pair -> weighted global occurrence count
    adj: dict[tuple[bytes, bytes], int] = {} 
    for i,x in seq_by_id.items():
        for left_token, right_token in zip(x[:-1],x[1:]):
            if (left_token,right_token) in adj:
                adj[(left_token,right_token)] += freq_by_id[i]
            else:
                adj[(left_token,right_token)] = freq_by_id[i]
            
            if (left_token,right_token) not in pair_to_seq_ids:
                pair_to_seq_ids[(left_token,right_token)] = set()
            pair_to_seq_ids[(left_token,right_token)].add(i)

    #Iteration till |V| < vocab__size
    while len(vocab) < vocab_size:
        #Select the most frequent pair, break frequency ties lexicographically
        if not adj: #No mergeable pairs remain
            break
        win_pair = max(adj,key=
                    lambda x: (adj[x],x))
        merged_token = win_pair[0]+win_pair[1]
        vocab[len(vocab)] = merged_token #add merged token to vocab
        merges.append(win_pair)
        #Restrict updates to only the seq that currently contain the winning pair
        regions = pair_to_seq_ids[win_pair].copy()

        #Construct the updated token sequence after applying the selected merge
        for seq_id in regions:
            token_seq = seq_by_id[seq_id]; i = 0
            updated_sequence = []
            while i < len(token_seq):
                if i+1 <len(token_seq) and (token_seq[i],token_seq[i+1]) == win_pair:
                    updated_sequence.append(merged_token)
                    i+=2
                else:
                    updated_sequence.append(token_seq[i])
                    i+=1
            #Remove this sequence's old contributions from the pair-freq cache and pair idx
            for left_token,right_token in zip(seq_by_id[seq_id][:-1],seq_by_id[seq_id][1:]):
                pair = (left_token,right_token)
                adj[pair] -= freq_by_id[seq_id]

                if adj[pair] == 0:
                    del adj[pair]

                if pair in pair_to_seq_ids:
                    pair_to_seq_ids[pair].discard(seq_id)

                    if not pair_to_seq_ids[pair]:
                        del pair_to_seq_ids[pair]
                
            #Replace the sequence's current BPE representation
            seq_by_id[seq_id] = tuple(updated_sequence)
            #Add this sequence's new contributions to the pair-frequency cache and pair index
            for left_token,right_token in zip(seq_by_id[seq_id][:-1],seq_by_id[seq_id][1:]):
                pair = (left_token,right_token)
                if pair in adj:    
                    adj[pair] += freq_by_id[seq_id]
                else:
                    adj[pair] = freq_by_id[seq_id]

                if pair not in pair_to_seq_ids:
                    pair_to_seq_ids[pair] = set()
                pair_to_seq_ids[pair].add(seq_id)
    #Return the learned vocabulary and merges in creation order
    return vocab,merges