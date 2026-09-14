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

Problem (Generating):
    uv run python cs336_basics/generate.py --prompt "Once upon a time"
    , in a small town, there was a boy named Tim. Tim was a tough boy who liked to play with his toys. One day, Tim saw a big, shiny toy in the store. He wanted it very much.
    Tim asked his mom, "Can I have the toy, please?" His mom said, "No, Tim. That toy is too much money." Tim was sad, but he still wanted the toy. So, when his mom was not looking, he took the toy and ran away.
    Later, Tim saw a small girl with a toy. She looked sad too. Tim walked up to her and said, "That's my toy! I want it." The girl looked at Tim and said, "I'm sorry, but I lost it. I can't give it back."
    Tim felt bad for the girl. He gave her the toy and said, "I'm sorry. I will give it back." The girl smiled and said, "Thank you, Tim." They became friends and played together with the toy. Tim learned that it's good to ask before taking something that does not belong to you.

Problem (Batch_size):
    I tested on 1,32,64,128. 1 has low GPU util, so tokens/s was much lower than for 32. 32 batch had around about 29 times the throughput of batch size 1. 64 slightly faster than 32 and 128 provided no no further improvement with respect to 64.
Problem (removing RMSNorm):
    I epxected much bigger difference. It was tested on 5000 steps 32 batch (1/8 of full trainaing token budget), lr=1e-3->1e-4. Final validation was arond 1.668 compared to 1.638 for the baseline. However, gradient norm spikes were much larger(expected), so similar final losses didn't mean equally stable training
Problem Post-norm:
    Same configuration as above. 1.674 val_result. Slightly worse than baseline(pre_norm). I did not observe major instability
Problem NoPE:
    Same configuration as above. val_loss 1.768. It was also worse for every checkpoint. Yes, thats worse than baseline, but i expected worse.
Problem SiLU:
    same .... val_loss 1.719. I used wider hidden layer for SiLU to approximately match the parameter count. SwiGLU performed better
Lr: 2000 steps
    run_5	hdiu7m2n	1e-3 → 1e-4	1.9067
    run_6	6on4ns9t	3e-3 → 3e-4	2.0960
    run_7	seq0laav	3e-3 → 1e-4	2.0586
    run_8	pxt3qc78	1e-2 → 1e-4	2.6628
    run_9	sp6kixyg	1e-2 → 1e-3	2.7256
    run_10	eg9qyej8	1e-5 → 1e-6	4.3279
    run_18	nq4vfkop	1e-1 → 1e-2	3.3625
    Run 5 became my future beseline -> Lets take 3x of it -> I then kept the same peak learning rate and lowered the final learning rate. I felt like at the very end it is better to take smaller lr -> I think it is logically better to run run-9 before run-8 -> 1e-5->1e-6 small and 1e-1->1e-2 big, so results are not that good. Also the graph showed that they decay at the beggining MUCH slower than e-3 family, so I decided to stop
    Final validation loss:
    For the full run lr = 1e-3->1e-4 gave val_loss = 1.4035 on baseline configuration