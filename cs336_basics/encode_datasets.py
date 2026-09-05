from cs336_basics.tokenizer import Tokenizer
from time import perf_counter
import numpy as np
from pathlib import Path
def iterable_read(input_path:str):
    buffer = ""
    with open(input_path,encoding="utf-8") as f:
        for line in f:
            buffer+=line
            while "<|endoftext|>" in buffer:
                left,sep,right = buffer.partition("<|endoftext|>")
                yield left + sep
                buffer = right
        if buffer:
            yield buffer

def encode_dataset(input_path: str,
                   output_path: str,
                   tokenizer:Tokenizer,
                   buffer_size:int):

    buffer = [];total_tokens=0

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    start_time = perf_counter()
    with open(output_path, "wb") as f_o:
        documents = iterable_read(input_path)
        iter_id = tokenizer.encode_iterable(documents)
        for token_id in iter_id:
            buffer.append(token_id)
            if len(buffer) >= buffer_size:
                buffer_np = np.asarray(buffer,dtype=np.uint16)
                buffer_np.tofile(f_o)
                total_tokens += len(buffer)
                print(total_tokens, perf_counter()-start_time)
                buffer = []
        if len(buffer) > 0:
            buffer_np = np.asarray(buffer,dtype=np.uint16)
            buffer_np.tofile(f_o); total_tokens += len(buffer)

    elapsed_time = perf_counter()-start_time
    o_path = Path(output_path)
    size = o_path.stat().st_size
    print(f"Is output size is correct: {size == total_tokens*2}")
    print(f"token_count: {total_tokens}")
    print(f"File size : {size}")
    print(f"Time (seconds): {elapsed_time}")

def main():
    tiny_obj = Tokenizer.from_file("artifacts/tokenizers/tinystories/bpe.pkl",
                                       ["<|endoftext|>"])
    owt_obj = Tokenizer.from_file("artifacts/tokenizers/openwebtext/bpe.pkl",
                                      ["<|endoftext|>"])

    print("TinyStories valid")
    encode_dataset("data/TinyStoriesV2-GPT4-valid.txt",
                   "artifacts/tokenized/tinystories_valid.bin",
                   tiny_obj,
                   buffer_size=1_000_000)

    print("OWT valid")
    encode_dataset("data/owt_valid.txt",
                    "artifacts/tokenized/owt_valid.bin",
                    owt_obj,
                    buffer_size=1_000_000)

    print("TinyStories train")
    encode_dataset("data/TinyStoriesV2-GPT4-train.txt",
                    "artifacts/tokenized/tinystories_train.bin",
                    tiny_obj,
                    buffer_size=1_000_000)

    print("OWT train")
    encode_dataset("data/owt_train.txt",
                    "artifacts/tokenized/owt_train.bin",
                    owt_obj,
                    buffer_size=1_000_000)

if __name__ == "__main__":
    main()