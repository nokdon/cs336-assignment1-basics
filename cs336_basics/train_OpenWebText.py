from time import perf_counter
from cs336_basics.bpe import train_bpe
import pickle

INPUT_PATH = "data/owt_train.txt"
special_tokens=["<|endoftext|>"]

def main():
    start_time = perf_counter()

    vocab,merges = train_bpe(
        input_path=INPUT_PATH,
        vocab_size=32_000,
        special_tokens=special_tokens,
        num_processes=4
    )
    elapsed = perf_counter() - start_time

    #Save vocab,merges
    with open("artifacts/tokenizers/openwebtext/bpe.pkl", "wb") as f:
        pickle.dump((vocab,merges),f)

    #Find longest token
    longest_token = vocab[0]
    for key,value in vocab.items():
        if len(value) > len(longest_token):
            longest_token = value

    print(f"Longest token = {longest_token} | len = {len(longest_token)}")

    print(f"Training time: {elapsed:.2f} seconds")
    print(f"|V| = {len(vocab)}")
    print(f"# of merges = {len(merges)}")

if __name__ == "__main__":
    main()