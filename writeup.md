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
    d)Uint-16 occupies 2 bytes, while int32 ocuppy around 4 bytes and int64 occupies 8 bytes. Also Uint-16 can represent 2^16=65536 possible tokens while |V| of Tiny = 10k and OWT = 32k and both vocabluraries fit this range

Problem (transformer_accounting):
    a)  
    Embedding - (vocab_size,    d_model) = 80_411_200
    Transformer_block: num_layers*(d_model+4*d_model^2+d_model+3*d_ff*d_model) = 1_479_628_800
    Norm - d_model = 1_600
    Linear - vocab_size*d_model = 80_411_200
    Sum = 1_640_452_800
    1_640_452_800 *4 = 6_561_811_200 bytes (float32) -> 6.56 GB
    b)
    48*Multi_head_self_attention:
      3*2*d_model*s_l*d_model = 15_728_640_000
        2*d_k*s_l^2 = 3_355_443_200
        2*s_l*s_l*d_v = 3_355_443_200
        2*s_l*d_l^2 = 5_242_880_000
        3*2*s_l*d_m*d_ff = 42_152_755_200
    = 3_352_087_756_800 FLOPs
    2*d_m*vocab_size*s_l = 164_682_137_600 FLOPs
    =3_516_769_894_400 FLOPs = 3.52 TFLOPs
    c) Transformer block more precisely PWFF
    d)As model increases FF proportion increases and Lm head and other proportions decrease
    e) 3.52->133.58 TFLOPs (38x), because attention score/value products grows ^2 with s_l while others linearly. The score/value  products raises from %9.16->61.73%, while SwiGLU falls 57%->24%, attention projections from 28%->12% and Lm head 4.7%->2%

Problem (learning_rate_tuning):
    1e1: 31.655099868774414
        20.259265899658203
        14.934264183044434
        ...
        4.883752822875977
        4.254292011260986
    1e2: 27.789491653442383
        4.767922878265381
        ...
        2.142694145777215e-22
        2.3807714834487026e-23
    1e3: 25.782888412475586
        9307.62109375
        ...
        7.442077094353306e+16
        2.389733822267654e+18
    1e1 converges but slower than 1e2 for 10 itteratins. 1e2 is just fine, faster than 1e2. 1e3 is too big -> diverges.

Problem (adamw_accounting):
    a) P (memory of params) = 6_561_811_200 bytes (float32) -> 6.56 GB from transformet_accounting
    m,v = 2* P =  2 * 6.56 GB = 13.12 GB
    gradients = P = 6,56 GB
    List below = 16.4*B GB
        2*RMSNorm = 2BSD
        Q,K,V = 3BSD
        S and A = 2BHS^2
        O = BSD
        r_one = BSD
        SwiGLU = 4BSF+BSD
        Final_Norm = BSD
        LM_head = BSV
        cross_entropy = BSV
    Result: 26.25+16.4*B
    b)B=3
    c)Per element of parameter cost:
        Weight decay: 2 FLOPs
        m_upd: 3 FLOPs
        v_upd : 4 FLOPs
        moment_adj: 5 FLOP
        Sum = 14
    1_640_452_800 * 14 = 22_966_339_200 FLOPs
    which is 22.97 GFLOPs
    d) 495 teraFLOP
        50% -> 247
        400K steps and B = 1024 and cost for one batch forward+backward+step
        = 3.516 TFLOP + 0.02296 TFLOP + 2*forward_cost =так = 10.57 TFLOP
        -> 400_000*1024*10.57 / 247 * 3600 = 4869 HOURS!!!
