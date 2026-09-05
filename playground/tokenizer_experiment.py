from itertools import islice
from cs336_basics.tokenizer import Tokenizer
from time import perf_counter
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

def task1(documents:list[str],
          obj: Tokenizer
)->int:
    total_bytes = 0; total_tokens = 0
    for doc in documents:
        total_bytes += len(doc.encode("utf-8"))
        total_tokens += len(obj.encode(doc))
    return total_bytes/total_tokens


def main():
    #load vocab,merges from trained BPE results
    tiny_obj = Tokenizer.from_file("artifacts/tokenizers/tinystories/bpe.pkl",
                                   ["<|endoftext|>"])
    owt_obj = Tokenizer.from_file("artifacts/tokenizers/openwebtext/bpe.pkl",
                                  ["<|endoftext|>"])

    #read 10 documents
    tiny_documents = iterable_read("data/TinyStoriesV2-GPT4-valid.txt")
    tiny_list = list(islice(tiny_documents,10))
    tiny_documents.close()

    owt_documents = iterable_read("data/owt_valid.txt")
    owt_list = list(islice(owt_documents,10))
    owt_documents.close()

    #Task 1
    print(task1(tiny_list,tiny_obj))
    print(task1(owt_list,owt_obj))

    #Task 2
    print(task1(owt_list,tiny_obj))

    #Task 3

    input_path = "data/owt_valid.txt"
    documets = iterable_read(input_path)
    sample_documents = list(islice(documets,2000))
    documets.close()

    file_size_bytes = 0

    for doc in sample_documents:
                file_size_bytes += len(doc.encode("utf-8"))
    if not (len(sample_documents) == 2000 and file_size_bytes>0):
         raise RuntimeError

    iter_result = owt_obj.encode_iterable(sample_documents)
    start_time = perf_counter()
    for x in iter_result:
        pass
    elapsed_seconds = perf_counter()-start_time

    throughput = file_size_bytes/elapsed_seconds
    estimated_seconds = 825*pow(10,9) / throughput
    print("Processed bytes:", file_size_bytes)
    print("Elapsed time (seconds):", elapsed_seconds)
    print("Throughput (bytes/second):", throughput)
    print("Estimated total time (seconds):", estimated_seconds)
    print("Estimated total time (hours):", estimated_seconds / 3600)
    print("Estimated total time (days):", estimated_seconds / 86_400)

if __name__ == "__main__":
    main()