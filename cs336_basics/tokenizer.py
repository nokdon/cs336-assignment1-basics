import pickle
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
import regex as re
from collections.abc import Iterable, Iterator
class Tokenizer:
    def __init__(self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens : list[str] =None
    ):
        self.vocab = vocab
        self.merges = merges
        if special_tokens is None:
            self.special_tokens = []
        else:
            self.special_tokens = special_tokens
        self.bytes_to_id = {value: key for key,value in vocab.items()}
        self.merge_pair_to_rank = {x: i for i,x in enumerate(merges)}

    @classmethod
    def from_files(cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens=None
    ):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab,merges,special_tokens)

    @classmethod
    def from_file(cls,
        bpe_path: str,
        special_tokens=None
    ):
        with open(bpe_path, "rb") as f:
            vocab,merges = pickle.load(f)
        return cls(vocab,merges,special_tokens)

    def encode(self,
        text: str
    )-> list[int]:
        results_id = []
        sorted_s_t = sorted(self.special_tokens,key=len,reverse=True)

        #Split corpus ignoring special token boundries
        escaped_s_t = []
        for i in range(len(sorted_s_t)):
            escaped_s_t.append(re.escape(sorted_s_t[i]))
        if len(escaped_s_t) > 0:
            pattern = "|".join(escaped_s_t)
            parts = re.split(f"({pattern})",text) #text between special tokens
        else:
            parts = [text]
        ##Apply GTP-2 type regex
        pretokens = []
        for word in parts:
            if word in self.special_tokens:
                pretokens.append(word)
            else:
                itterator = re.finditer(PAT,word)
                for match in itterator:
                    pretokens.append(match.group())


        for word in pretokens:
            #Special token case
            if word in self.special_tokens:
                id = self.bytes_to_id[word.encode("utf-8")]
                results_id.append(id)
            else:
                #Byte representation of word
                token_seq = [bytes([b]) for b in word.encode("utf-8")]
                while True:
                    pairs = [pair for pair in zip(token_seq[:-1],token_seq[1:])]
                    token_seq_intersection_merge = [x for x in pairs if x in self.merge_pair_to_rank]
                    if len(token_seq_intersection_merge) > 0:
                        pair = min(token_seq_intersection_merge,key=self.merge_pair_to_rank.get)
                    else:
                        break
                    i = 0;updated_seq = []

                    while i < len(token_seq):
                        if i+1 <len(token_seq) and (token_seq[i],token_seq[i+1]) == pair:
                            updated_seq.append(pair[0]+pair[1])
                            i+=2
                        else:
                            updated_seq.append(token_seq[i])
                            i+=1
                    token_seq = updated_seq
                #Append ID after merge
                for b in token_seq:
                    token_id = self.bytes_to_id[b]
                    results_id.append(token_id)
        return results_id

    def encode_iterable(self,
                        iterable: Iterable[str]
    )-> Iterator[int]:
        for text_chunk in iterable:
            yield from self.encode(text_chunk)

    def decode(self,
            ids: list[int]
    )-> str:
        output = [self.vocab[token_id] for token_id in ids]
        return b"".join(output).decode("utf-8", errors="replace")