from cs336_basics.bpe import train_bpe

if __name__ == "__main__":
    one = train_bpe("tests/fixtures/tinystories_sample.txt",500,special_tokens=["<|endoftext|>"])
    four = train_bpe("tests/fixtures/tinystories_sample.txt",500,special_tokens=["<|endoftext|>"],
                     num_processes=4)
    print(one == four)