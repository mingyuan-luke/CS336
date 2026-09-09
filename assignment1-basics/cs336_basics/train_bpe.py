import os
from typing import BinaryIO
import regex as re
import cProfile
import time
import json
from tests.common import gpt2_bytes_to_unicode
from multiprocessing import Pool
from collections import defaultdict

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

# 写一个worker函数，接收一个chunk的start和end，返回该chunk的pretoken_counts
def process_chunk(args):
    start, end, input_path, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        if special_tokens:
            pattern = "|".join(re.escape(st) for st in special_tokens)
            chunks = re.split(pattern, chunk)
        else:
            chunks = [chunk]

        pretoken_counts = {}
        for chunk in chunks:
            for match in re.finditer(PAT, chunk):
                word = match.group(0)
                token_tuple = tuple(bytes([b]) for b in word.encode("utf-8"))
                pretoken_counts[token_tuple] = pretoken_counts.get(token_tuple, 0) + 1
    return pretoken_counts

# 正向：bytes → 可读字符串
byte_encoder = gpt2_bytes_to_unicode()
def encode_bytes(b: bytes) -> str:
    return "".join(byte_encoder[x] for x in b)

# 反向：可读字符串 → bytes
byte_decoder = {v: k for k, v in byte_encoder.items()}
def decode_str(s: str) -> bytes:
    return bytes([byte_decoder[c] for c in s])

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = {}
    token_to_id = {}
    next_id = 0
    merges = []

    t0 = time.perf_counter()
    # 1. 先加special tokens
    for st in special_tokens:
        st_bytes = st.encode("utf-8")
        vocab[next_id] = st_bytes
        token_to_id[st_bytes] = next_id
        next_id += 1

    # # 2. 再加 256 个单字节
    for byte_val in range(256):
        byte_token = bytes([byte_val])
        vocab[next_id] = byte_token
        token_to_id[byte_token] = next_id
        next_id += 1

    # 3. 预分词 + 计数
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    if special_tokens:
        pattern = "|".join(re.escape(st) for st in special_tokens)
        # print(f"Splitting text on special tokens: {pattern}")
        chunks = re.split(pattern, text)
    else:
        chunks = [text]

    # print(f"chunks: {chunks}")
    # 对每个chunk预分词并计数
    pretoken_counts = {}
    for chunk in chunks:
        for match in re.finditer(PAT, chunk):
            word = match.group(0)
            token_tuple = tuple(bytes([b]) for b in word.encode("utf-8"))
            # print(f"Token tuple: {token_tuple}")
            pretoken_counts[token_tuple] = pretoken_counts.get(token_tuple, 0) + 1

    t1 = time.perf_counter()
    print(f"Pre-tokenization and counting took {t1 - t0:.2f} seconds")

    # print(f"origin pretoken_counts: {pretoken_counts}")
    # top10 = sorted(pretoken_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    # print(f"Top 10 pre-token counts: {top10}")
    # for token_tuple, count in top10:
    #     print(f"{b''.join(token_tuple)}: {count}")

    # 4. BPE merge
    t_loop_start = time.perf_counter()
    while len(vocab) < vocab_size:
        # 4.1 统计所有相邻token pair的频率
        pair_counts = {}
        for token_tuple, count in pretoken_counts.items():
            for i in range(len(token_tuple) - 1):
                pair = (token_tuple[i], token_tuple[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + count

        if not pair_counts:
            break

        if max(pair_counts.values()) <= 1:
            break

        # print(f"pair_counts: {pair_counts}")

        # 4.2 找到出现频率最高的pair
        # best_pair = max(pair_counts, key=pair_counts.get) # 错误，没有取字典序最大的
        # 这里同样count的pair，需要按照字典序选择最大的pair
        # best_pair = sorted(pair_counts.items(), key=lambda x: (x[1], x[0]))[-1][0]
        best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0] # 这个更快
        # print(f"Best pair: {best_pair} with count: {best_count}")

        # 4.3 将该pair合并为一个新token
        new_token = b"".join(best_pair)
        vocab[next_id] = new_token
        token_to_id[new_token] = next_id
        next_id += 1

        # 4.4 更新pretoken_counts，替换所有出现的best_pair为new_token
        new_pretoken_counts = {}
        for token_tuple, count in pretoken_counts.items():
            new_token_tuple = []
            i = 0
            while i < len(token_tuple):
                if i < len(token_tuple) - 1 and (token_tuple[i], token_tuple[i + 1]) == best_pair:
                    new_token_tuple.append(new_token)
                    i += 2
                else:
                    new_token_tuple.append(token_tuple[i])
                    i += 1
            # 注意这里更新的是原始的pretoken_counts，而不是pair_counts
            new_pretoken_counts[tuple(new_token_tuple)] = new_pretoken_counts.get(tuple(new_token_tuple), 0) + count

        # print(f"Updated pretoken_counts: {new_pretoken_counts}")
        pretoken_counts = new_pretoken_counts
        merges.append(best_pair)

        if len(merges) % 100 == 0:
            elapsed_time = time.perf_counter() - t_loop_start
            t_loop_start = time.perf_counter()
            print(f"已合并 {len(merges)} 个pair，耗时 {elapsed_time:.2f} 秒"),
    # print(f"Final vocab: {vocab}")
    # print(f"Final merges: {merges}")

    t2 = time.perf_counter()
    print(f"BPE training took {t2 - t1:.2f} seconds")

    # 把vocab和merges写入文件
    with open("results/vocab.json", "w", encoding="utf-8") as f:
        json.dump({k: encode_bytes(v) for k, v in vocab.items()}, f, ensure_ascii=False)
    with open("results/merges.txt", "w", encoding="utf-8") as f:
        for pair in merges:
            f.write(f"{encode_bytes(pair[0])} {encode_bytes(pair[1])}\n")

    # 打印最长的token
    longest_token = max(vocab.values(), key=len)
    print(f"Longest token: {longest_token} with length {len(longest_token)}")
    return vocab, merges


def train_bpe_parallel(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 4,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train BPE in parallel by splitting the input file into chunks.
    """
    t0 = time.perf_counter()
    chunk_num = 200
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, chunk_num, b"<|endoftext|>")
        print(f"Chunk boundaries: {boundaries}")

        # Prepare arguments for each chunk
        args = [(start, end, input_path, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]

        # Use multiprocessing to process chunks in parallel
        final_pretoken_counts = {}
        with Pool(processes=num_processes, maxtasksperchild=50) as pool:
            # Use imap_unordered to process chunks in parallel and collect results as they complete
            # 改成流式消费
            # results = pool.map(process_chunk, args)
            results = list(pool.imap_unordered(process_chunk, args))
            # Merge results from all chunks
            for pretoken_counts in results:
                for token_tuple, count in pretoken_counts.items():
                    final_pretoken_counts[token_tuple] = final_pretoken_counts.get(token_tuple, 0) + count


    t1 = time.perf_counter()
    print(f"Parallel Pre-tokenization and counting took {t1 - t0:.2f} seconds")  
    # Continue with BPE training using final_pretoken_counts
    vocab = {}
    token_to_id = {}
    next_id = 0
    merges = []

    # Add special tokens and single-byte tokens to vocab
    for st in special_tokens:
        st_bytes = st.encode("utf-8")
        vocab[next_id] = st_bytes
        token_to_id[st_bytes] = next_id
        next_id += 1

    for byte_val in range(256):
        byte_token = bytes([byte_val])
        vocab[next_id] = byte_token
        token_to_id[byte_token] = next_id
        next_id += 1

    # 用新的更新pair的方法进行BPE训练
    # 一次 merge 只会影响合并位置附近的 pair, 不希望每次都重新计数

    # 给每个pretoken编号，节省内存，优化点去掉pretoken_to_id
    id_to_pretoken = {idx: token_tuple for idx, token_tuple in enumerate(final_pretoken_counts.keys())}

    # 先在外面记录下初始的 pair_counts
    pair_counts = {}
    pair_to_pretokens = defaultdict(set)
    # for token_tuple, count in final_pretoken_counts.items():
    for idx, token_tuple in id_to_pretoken.items():
        count = final_pretoken_counts[token_tuple]
        for i in range(len(token_tuple) - 1):
            pair = (token_tuple[i], token_tuple[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + count
            pair_to_pretokens[pair].add(idx) # 只记录set，会去重，出现几次得去final_pretoken_counts里查

    # for pair, pretoken_set in pair_to_pretokens.items():
    #     print(f"Pair: {pair}, Affected pretokens: {pretoken_set}")

    t_loop_start = time.perf_counter()
    while len(vocab) < vocab_size:
        # 循环内更新维护pair_counts，而不是每次都重新计算
        if not pair_counts or max(pair_counts.values()) <= 1:
            break

        best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
        # print(f"Best pair: {best_pair} with count: {pair_counts[best_pair]}")

        new_token = b"".join(best_pair)
        vocab[next_id] = new_token
        token_to_id[new_token] = next_id
        next_id += 1

        # 增量bpe核心，维护pair_counts和pair_to_pretokens
        affected_pretokens = set(pair_to_pretokens.get(best_pair, ())) # 只处理受影响的pretoken，创建副本
        for id in affected_pretokens:
            token_tuple = id_to_pretoken[id]
            count = final_pretoken_counts[token_tuple] # 这里记录了原始token tuple出现的次数
            new_token_tuple = []
            i = 0
            while i < len(token_tuple):
                if i < len(token_tuple) - 1 and (token_tuple[i], token_tuple[i + 1]) == best_pair:
                    new_token_tuple.append(new_token)
                    i += 2
                else:
                    new_token_tuple.append(token_tuple[i])
                    i += 1
            new_token_tuple = tuple(new_token_tuple) # 先把所有转换的新结果收集起来
            final_pretoken_counts[new_token_tuple] = final_pretoken_counts.get(new_token_tuple, 0) + count # 应该大多是新的tuple，原来的count说明出现了几次得继续保留

            # 更新 id_to_pretoken
            id_to_pretoken[id] = new_token_tuple
            del final_pretoken_counts[token_tuple] # 删除旧的token_tuple

            # 检查， ID 与 tuple是否一对一对应
            # assert len(id_to_pretoken) == len(set(id_to_pretoken.values()))
            # assert set(id_to_pretoken.values()) == set(final_pretoken_counts)

            # 更新 pair_counts 和 pair_to_pretokens
            # 因为token_tuple被删掉了，所以它的所有pair都要减少计数
            for i in range(len(token_tuple) - 1):
                pair = (token_tuple[i], token_tuple[i + 1])
                pair_counts[pair] -= count
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                # pair_to_pretokens[pair].discard(token_tuple)
                pair_to_pretokens[pair].discard(id)

            # 把new_token_tuple都是更新，全部更新进pair_counts
            for i in range(len(new_token_tuple) - 1):
                pair = (new_token_tuple[i], new_token_tuple[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + count
                # pair_to_pretokens[pair].add(new_token_tuple)
                pair_to_pretokens[pair].add(id) # 这里是新tuple的id

        merges.append(best_pair)
        if len(merges) % 2000 == 0:
            elapsed_time = time.perf_counter() - t_loop_start
            t_loop_start = time.perf_counter()
            print(f"已合并 {len(merges)} 个pair，耗时 {elapsed_time:.2f} 秒")


    t2 = time.perf_counter()
    print(f"BPE training took {t2 - t1:.2f} seconds")

    # 把vocab和merges写入文件
    with open("results/vocab_parallel_ts.json", "w", encoding="utf-8") as f:
        json.dump({k: encode_bytes(v) for k, v in vocab.items()}, f, ensure_ascii=False)
    with open("results/merges_parallel_ts.txt", "w", encoding="utf-8") as f:
        for pair in merges:
            f.write(f"{encode_bytes(pair[0])} {encode_bytes(pair[1])}\n")

    # 打印最长的token
    longest_token = max(vocab.values(), key=len)
    print(f"Longest token: {longest_token} with length {len(longest_token)}")
    return vocab, merges


if __name__ == "__main__":
    # cProfile.run("train_bpe('tests/fixtures/tinystories_sample_5M.txt', 10000, ['<|endoftext|>'])", sort="cumulative", filename="train_bpe_profile.stats")
    # vocab, merges = train_bpe('tests/fixtures/corpus.en', 500, ['<|endoftext|>'])
    # vocab_parallel, merges_parallel = train_bpe_parallel('tests/fixtures/corpus.en', 500, ['<|endoftext|>'], 4)

    # vocab, merges = train_bpe('tests/fixtures/tinystories_sample_5M.txt', 800, ['<|endoftext|>'])
    # vocab_parallel, merges_parallel = train_bpe_parallel('tests/fixtures/tinystories_sample_5M.txt', 800, ['<|endoftext|>'], 4)

    vocab_parallel, merges_parallel = train_bpe_parallel('data/TinyStoriesV2-GPT4-train.txt', 10000, ['<|endoftext|>'], 30)
    # vocab_parallel, merges_parallel = train_bpe_parallel('data/owt_train.txt', 32000, ['<|endoftext|>'], 8)

    # assert set(vocab) == set(vocab_parallel)
    # assert set(merges) == set(merges_parallel)

    # /usr/bin/time -p uv run python cs336_basics/tokenizer.py # 测耗时
    # /usr/bin/time -l uv run python cs336_basics/tokenizer.py # 测内存