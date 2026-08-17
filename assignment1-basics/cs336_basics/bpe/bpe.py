import os
import copy
import regex as re
from pathlib import Path
from typing import BinaryIO
from collections import Counter, defaultdict
from multiprocessing import Pool

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_COMPILED_PAT = re.compile(PAT)

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

def process_chunk(args: tuple[str, int, int, list[str]]) -> dict[str, int]:
    """
    Split Chunk by Special Tokens and Perform Word Count.
    """
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors = "ignore")

    # 1. Split Chunk by Special Tokens
    if len(special_tokens) > 0:
        pattern = "|".join(re.escape(tok) for tok in special_tokens)
        docs = re.split(pattern, chunk)
    else:
        docs = [chunk]

    # 2. Pre-Tokenization and Word Count
    word_counts: dict[str, int] = Counter()
    for doc in docs:
        for match in re.finditer(_COMPILED_PAT, doc):
            # Obtain Word
            token = match.group()

            # Count Tokens
            word_counts[token] += 1

    return word_counts

def merge_pair(
    vocab: dict[int, bytes],
    combined_pretoken_counts: Counter,
    pretoken_to_tokens: defaultdict[str, list[bytes]],
    pair_to_pretokens: defaultdict[tuple[bytes, bytes], set[str]],
    pair_counts: Counter,
    merges: list[tuple[bytes, bytes]],
):
    # 1. Add Most Common Byte Pair into merges
    best_pair = max(
        pair_counts,
        key = lambda pair: (pair_counts[pair], pair),
    )
    merges.append(best_pair)
    
    # 2. Add to vocab
    new_byte = best_pair[0] + best_pair[1]
    vocab[len(vocab)] = new_byte

    # 3. Update
    pretokens_copy = pair_to_pretokens[best_pair].copy()

    for pretoken in pretokens_copy:

        # Update Pretoken to Tokens
        token_list = pretoken_to_tokens[pretoken]
        new_token_list = []

        i = 0
        while i < len(token_list):
            # Case 1: if find match at current byte and next byte
            if i + 1 < len(token_list) and \
                token_list[i] == best_pair[0] and \
                token_list[i + 1] == best_pair[1]:
                new_token_list.append(new_byte)
                i += 2
            # Case 2: if not match
            else:
                new_token_list.append(token_list[i])
                i += 1

        pretoken_to_tokens[pretoken] = new_token_list
        
        # Decrement Pair Count for Old Token List
        old_pair_count = Counter(zip(token_list, token_list[1:]))
        for old_pair, count in old_pair_count.items():
            pair_counts[old_pair] -= count * combined_pretoken_counts[pretoken]
            if pair_counts[old_pair] == 0:
                pair_counts.pop(old_pair, None)

        # Increment Pair Count for New Token List
        new_pair_count = Counter(zip(new_token_list, new_token_list[1:]))
        for new_pair, count in new_pair_count.items():
            pair_counts[new_pair] += count * combined_pretoken_counts[pretoken]

        
        # Delete and Add Pretoken
        old_pairs = set(old_pair_count)
        new_pairs = set(new_pair_count)

        for p in old_pairs - new_pairs:
            pair_to_pretokens[p].remove(pretoken)
            if len(pair_to_pretokens[p]) == 0:
                pair_to_pretokens.pop(p, None)

        for p in new_pairs - old_pairs:
            pair_to_pretokens[p].add(pretoken)
    
    return None

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 4
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # 1. Init Vocab -> dict[int, bytes]
    vocab = {i : bytes([i]) for i in range(256)}

    # 2. Add Special Tokens
    for i in range(len(special_tokens)):
        vocab[len(vocab)] = special_tokens[i].encode("utf-8")

    # 3. Read File
    with open(input_path, "rb") as f:
        # 4. Split into Chunk by Special Tokens

        if not special_tokens:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()

            chunk_boundaries = [0, file_size]
        else:
            split_token = special_tokens[0].encode("utf-8")
            chunk_boundaries = find_chunk_boundaries(
                f, num_processes, split_token
            )

        # 5. Create args for Multiprocessing
        tasks = []

        for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
            tasks.append((input_path, start, end, special_tokens))

        # 6. Multiprocessing Chunks and Combine All Counts
        with Pool(processes = num_processes) as pool:
            chunk_pretoken_counts = pool.map(process_chunk, tasks)

        combined_pretoken_counts = Counter()

        for pretoken_count in chunk_pretoken_counts:
            combined_pretoken_counts.update(pretoken_count)
        
        # 7. Init Create Pair Bytes Set and Count Pair Bytes in Word
        pretoken_to_tokens = defaultdict(list) # token: tuple byte sequence
        pair_to_pretokens = defaultdict(set) # tuple byte pair: set
        pair_counts = Counter() # tuple byte pair: count
        for word, frequency in combined_pretoken_counts.items():
            encoded_word = word.encode("utf-8")

            tokens = list(bytes([byte_value]) for byte_value in encoded_word)

            pretoken_to_tokens[word] = tokens

            for byte_pair in zip(tokens, tokens[1:]):
                pair_to_pretokens[byte_pair].add(word)
                pair_counts[byte_pair] += frequency
        
        # 8. Merge Pair
        merges: list[tuple[bytes, bytes]] = []
        while len(vocab) < vocab_size and pair_counts:
            merge_pair(
                vocab,
                combined_pretoken_counts,
                pretoken_to_tokens,
                pair_to_pretokens,
                pair_counts,
                merges
            )

    return vocab, merges
import cProfile

if __name__ == "__main__":
    
    # train_bpe("dummy_path", 256, [])
    cProfile.run('train_bpe("dummy_path", 500, [])', sort='tottime')