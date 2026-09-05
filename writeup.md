Problem (unicode1):
    a) \x00
    b) repr() output '\x00' to inidicate that this is character of null, while print outputs the actual char and therefore appears blank
    c) \x00 has no printable appearing, so terminal displays no visable width for it, therefore the text appears adjacent

Problem (unicode2):
    a) UTF-8 can have 1,2,3,4 byte representation, while utf-16 2-4, utf-32 4 bytes. Utf-8 is more economical than utf-16,32 for ASCII english words
    b) The function would work only if character is encoded as a one byte sequence
    c) [0b11000000,0b00000001]. It breaks mechanism of encoding of utf-8

Problem (train_bpe_tinystories):
    Peak RAM: 11,146,500 KiB ≈ 10.63 GiB;
    File Read Time Cost = 25.32968
    Pre-tokenization Time Cost = 361.70
    BPE-structure prepairing Time Cost = 0.37
    Merge Loop Time Cost = 48.64
    Longest token = b' accomplishment' | len = 15 bytes
    Training time: 437.27 seconds
    |V| = 10000

Problem (train_bpe_expts_owt):
    Longest token = b'\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82' | len = 64
    Training time: 18382.17 seconds
    |V| = 32000
    # of merges = 31743
    a)The longest token makes sense, because we read noisy web text, where exists such tokens. This tokens are errors of repeated encoding, so called mojibake artifacts
    b)The the majority of first longest tokens in openwebtxt are mojibake artifacts. While tinystories are clean text where longest tokens are actual readable english words

Problem (tokenizer_experiments):
    a)Tiny = 4.04 bytes/token, OWT = 4.51 bytes/token
    b) 3.41 bytes/token. TinyStories tokenizer is trained on more simple homogeneous corpus and have 10k voc vs 32k thats why different OWT sequences splits on more short subwords
    с)  Processed bytes: 10116434
        Elapsed time (seconds): 29.467023493998568
        Throughput (bytes/second): 343313.7385613574
        Estimated total time (seconds): 2403049.7685794043
        Estimated total time (hours): 667.5138246053901
        Estimated total time (days): 27.813076025224586
        If we linearly estimate thus Esitmated time = Estimated_num_bytes / rate we obtain 667 hours