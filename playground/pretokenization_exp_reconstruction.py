import os
from typing import BinaryIO

def find_chunk_boundries(
        file: BinaryIO,
        desired_num_chunks: int,
        special_tokens: bytes
) -> list[int]:

    assert isinstance(special_tokens,bytes)

    #Get file size
    file.seek(0,os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    #Approximate size per chunk
    chunk_size = file_size//desired_num_chunks

    boundries = [i*chunk_size for i in range(desired_num_chunks+1)]
    boundries[-1] = file_size #set exact end boundry

    mini_chunk_size = 4096

    #repair internal chunk assumptions
    for bi in range(1,len(boundries)-1):
        initial_pos = boundries[bi]
        file.seek(initial_pos)
        while True:
            mini_chunk = file.read(mini_chunk_size)

            #no bytes left to read
            if mini_chunk == b"":
                boundries[bi] = file_size
                break
            #special token occured
            found_at = mini_chunk.find(special_tokens)
            if found_at != -1:
                boundries[bi] = initial_pos+found_at
                break
            initial_pos+=mini_chunk_size

    #remove dublicates -> sort -> list
    return sorted(set(boundries))

#open
with open(...,"rb") as f:
    p = 4
    boundries = find_chunk_boundries(f,p,b"<|endoftext|>")

    for start,end in zip(boundries[:-1],boundries[1:]):
        f.seek(start)
        chunk = f.read(end-start).decode("utf-8",errors="ignore")