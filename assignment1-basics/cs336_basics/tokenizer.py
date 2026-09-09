from typing import Iterable, Iterator
import regex as re
from tests.common import gpt2_bytes_to_unicode
import json
import time

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# 正向：bytes → 可读字符串
byte_encoder = gpt2_bytes_to_unicode() # 这里面就是一张映射表
def encode_bytes(b: bytes) -> str:
    return "".join(byte_encoder[x] for x in b)

# 反向：可读字符串 → bytes
byte_decoder = {v: k for k, v in byte_encoder.items()}
def decode_str(s: str) -> bytes:
    return bytes([byte_decoder[c] for c in s])

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens is not None else []
        self.vocab_bytes_to_id = {v: k for k, v in vocab.items()}
        self.pair_rank_map = {pair: i for i, pair in enumerate(merges)}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        # vocab is json file, use json.load to load it
        with open(vocab_filepath, 'r', encoding='utf-8') as vf:
            vocab = json.load(vf)
            # convert keys to int and values to bytes
            # vocab = {int(index): decode_str(v) for v, index in vocab.items()}
            vocab = {int(index): decode_str(v) for index, v in vocab.items()}
        with open(merges_filepath, 'r', encoding='utf-8') as mf:
            merges = [tuple(line.strip().split()) for line in mf if line.strip()]
            # convert str to bytes
            merges = [(decode_str(a), decode_str(b)) for a, b in merges]

        sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
        if special_tokens:
            # special tokens is not in vacab, so we need to add them to the vocab with new ids
            max_id = max(vocab.keys())

            # sort special tokens by length in descending order to avoid partial matches
            # special_tokens.sort(key=len, reverse=True)
            for token in sorted_special_tokens:
                # if token is not in vocab, add it with a new id
                if token.encode("utf-8") not in vocab.values():
                    max_id += 1
                    vocab[max_id] = token.encode("utf-8")
                    print(f"Added special tokens to vocab: {vocab[max_id]} with id {max_id}")
        return cls(vocab, merges, sorted_special_tokens)



    def encode(self, text: str) -> list[int]:
        # Implement the encoding logic here
        if self.special_tokens:
            pattern = "|".join(re.escape(st) for st in self.special_tokens)
            ## chunks = re.split(pattern, text) # 这会丢失特殊token, 需要保留特殊token
            # print(f"Splitting text on special tokens: {pattern}")
            # keep the special tokens in the chunks
            chunks = re.split(f"({pattern})", text)
            # print(f"Chunks after splitting: {chunks}")
            # remove empty strings from chunks
            chunks = [chunk for chunk in chunks if chunk]
            # print(f"Removing empty chunks: {chunks}")
        else:
            chunks = [text]

        # print(f"Chunks: {chunks}")
        res = []
        for chunk in chunks:
            if chunk in self.special_tokens:
                # 如果是特殊token, 直接查表vocab_bytes_to_id
                token_id = self.vocab_bytes_to_id.get(chunk.encode("utf-8"))
                if token_id is not None:
                    res.append(token_id)
                    # print(f"Found special token '{chunk}' with ID {token_id}")
                else:
                    print(f"Special token '{chunk}' not found in vocab")
            else:
                for match in re.finditer(PAT, chunk):
                    word = match.group(0)
                    token_tuple = tuple(bytes([b]) for b in word.encode("utf-8"))
                    # print(f"Token tuple: {token_tuple}")

                    # 合并只发生在位置 i,受影响的相邻关系只有 i-1 和 i+1 两处,其余 pair 的 rank 一步都没变
                    # 维护一个"各位置 rank"的数组,每步只更新受影响的两个位置,再取最小
                    pairs = [(token_tuple[i], token_tuple[i + 1]) for i in range(len(token_tuple) - 1)]
                    rank_list = [self.pair_rank_map.get(pair, float('inf')) for pair in pairs]
                    has_merged = True
                    while has_merged:
                        if not rank_list:
                            break
                        min_rank = min(rank_list)
                        if min_rank == float('inf'):
                            break
                        min_index = rank_list.index(min_rank) # if two pairs have the same rank, we should merge the leftmost one first, so we use index() to find the first occurrence

                        # merge the best pair
                        token_tuple = token_tuple[:min_index] + (token_tuple[min_index] + token_tuple[min_index + 1],) + token_tuple[min_index + 2:]
                        has_merged = True

                        # remove the rank of the merged pair
                        rank_list.pop(min_index)

                        # update rank_list for the affected pairs
                        if min_index > 0:
                            rank_list[min_index - 1] = self.pair_rank_map.get((token_tuple[min_index - 1], token_tuple[min_index]), float('inf'))
                        if min_index < len(token_tuple) - 1:
                            rank_list[min_index] = self.pair_rank_map.get((token_tuple[min_index], token_tuple[min_index + 1]), float('inf'))

                    # 太慢，去掉第一额外循环，直接在词内合并
                    # has_merged = True
                    # while has_merged:
                    #     # get every pair of adjacent tokens in token_tuple
                    #     pairs = [(token_tuple[i], token_tuple[i + 1]) for i in range(len(token_tuple) - 1)]
                    #     # find the pair with the highest priority in merges
                    #     best_pair = None
                    #     best_rank = float('inf')
                    #     for pair in pairs: # 从左到右遍历所有相邻的token对
                    #         rank = self.pair_rank_map.get(pair, float('inf'))
                    #         if rank < best_rank: # 从merges的优先级从高到低进行合并, 得到的vocab id更短更完整
                    #             best_pair = pair
                    #             best_rank = rank
                    #     if best_pair is not None:
                    #         # merge the best pair
                    #         i = pairs.index(best_pair)
                    #         token_tuple = token_tuple[:i] + (token_tuple[i] + token_tuple[i + 1],) + token_tuple[i + 2:]
                    #         has_merged = True
                    #     else:
                    #         has_merged = False

                    # has_merged = True
                    # while has_merged:
                    #     has_merged = False
                    #     # 如果按照merges的优先级从高到低进行合并, 得到的vocab id更短更完整 [15496, 11, 995, 0, 770, 318, 257, 1332, 13]
                    #     for merge in self.merges: # 从高优先级到低优先级
                    #         # 如果按照词从左到右合并，更碎一些，结果更长 [1544, 18798, 11, 24486, 45895, 67, 0, 770, 1312, 82, 257, 573, 301, 13]
                    #         for i in range(len(token_tuple) - 1):
                    #             pair = (token_tuple[i], token_tuple[i + 1]) # 从前往后
                    #             # print(f"Checking pair: {pair}")

                    #             if pair == merge:
                    #                 # 如果能合并，就把两个token合并成一个
                    #                 token_tuple = token_tuple[:i] + (token_tuple[i] + token_tuple[i + 1],) + token_tuple[i + 2:]
                    #                 # print(f"Merged {pair} into {pair[0] + pair[1]} -> {token_tuple}")
                    #                 has_merged = True
                    #                 break
                    #         if has_merged:
                    #             break

                    # print(f"Final token tuple after merges: {token_tuple}")
                    res.extend(self.vocab_bytes_to_id[token] for token in token_tuple)
                    # print(f"Encoded IDs for this token tuple: {[self.vocab_bytes_to_id[token] for token in token_tuple]}")
        return res


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    
    def decode(self, ids: list[int]) -> str:
        # Implement the decoding logic here
        decoded_bytes = b''.join(self.vocab[i] for i in ids)
        return decoded_bytes.decode("utf-8", errors="ignore")


if __name__ == "__main__":
    # tokenizer = Tokenizer.from_files("tests/fixtures/gpt2_vocab.json", "tests/fixtures/gpt2_merges.txt", special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"])

    tokenizer = Tokenizer.from_files("results/vocab_parallel_ts.json", "results/merges_parallel_ts.txt", special_tokens=["<|endoftext|>"])
    # tokenizer = Tokenizer.from_files("results/vocab_parallel_owt.json", "results/merges_parallel_owt.txt", special_tokens=["<|endoftext|>"])
    # text = "Hello, world! <PAD> This is a test."
    # test_string = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"

    # use assignment1-basics/data/tinystories_10_sample.txt as test string
    # with open("data/owt_valid.txt", "r", encoding="utf-8") as f:
    with open("data/TinyStoriesV2-GPT4-valid.txt") as f:
    # with open("data/encode/tinystories_10_sample.txt", "r", encoding="utf-8") as f:
        test_string = f.read()

    # ids = tokenizer.encode(test_string) # warmup

    t0 = time.perf_counter()
    ids = tokenizer.encode(test_string)
    t1 = time.perf_counter()
    print(f"Encoding time: {t1 - t0:.4f} seconds")

    # calculate encoding speed, bytes/s
    encoding_speed = len(test_string.encode("utf-8")) / (t1 - t0)
    print(f"Encoding speed: {encoding_speed:.2f} bytes/second")

    # use uint16 numpy数组存储这些token id，写入文件
    # import numpy as np
    # ids_array = np.array(ids, dtype=np.uint16)
    # np.save("results/tokenized/tinystories_10_sample.npy", ids_array)

    # tokenized_string = [tokenizer.decode([x]) for x in ids]

    # write the tokenized_string to a file
    # with open("results/tokenized/tinystories_10_sample.txt", "w", encoding="utf-8") as f:
    # with open("results/tokenized/owt_10_sample_with_ts_vocab.txt", "w", encoding="utf-8") as f:
    #     for token in tokenized_string:
    #         f.write(token + "\n")
    # compress_ratio = len(test_string.encode("utf-8")) / len(ids)
    # print(f"Compression ratio: {compress_ratio:.2f} (original bytes / tokenized ids)")
    # print(f"Tokenized string: {tokenized_string}")
    # print(tokenizer.decode(ids) == test_string)