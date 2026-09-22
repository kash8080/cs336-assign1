import os
import json
import time
from collections.abc import Iterable, Iterator
import regex as re
import numpy as np
import multiprocessing
from typing import Optional

# for multiprocessing
# This global exists separately inside every worker process.
_worker_tokenizer: Optional["Tokenizer"] = None

def initialize_worker(
    vocab_path: str,
    merges_path: str,
    special_tokens: list[str],
) -> None:
    """Called once when each worker process starts."""
    global _worker_tokenizer

    _worker_tokenizer = Tokenizer.from_files(
        vocab_path,
        merges_path,
        special_tokens,
    )

def encode_text(text: str) -> list[int]:
    """Called once for each text passed to pool.map()."""
    if _worker_tokenizer is None:
        raise RuntimeError("Worker tokenizer was not initialized")

    return _worker_tokenizer.encode(text)


class Tokenizer:

    def __init__(self, vocab, merges, special_tokens=None):
        # Construct a tokenizer from a given
        # vocabulary, list of merges, and (optionally) a list of special tokens. This function should accept
        # the following parameters:
        # vocab: dict[int, bytes]
        # merges: list[tuple[bytes, bytes]]
        # special_tokens: list[str] | None = None
        self.vocab = vocab
        self.vocab_rev = {v: int(k) for k, v in vocab.items()}
        self.merges = merges
        self.merges_map = {item: idx for idx, item in enumerate(merges)}
        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        # Class method
        # that constructs and returns a Tokenizer from a serialized vocabulary and list of merges (in the
        # same format that your BPE training code output) and (optionally) a list of special tokens.
        # This method should accept the following additional parameters:
        # vocab_filepath: str
        # merges_filepath: str
        # special_tokens: list[str] | None = None

        with open(vocab_filepath, "r") as f:
            vocab_json = f.read()
            vocab = {int(k): v.encode("latin-1") for k, v in json.loads(vocab_json).items()}
            # vocab = json.loads(vocab_json)
            # for key in vocab:
            #     vocab[key] = vocab[key].encode("latin-1")
            # print(vocab)

        with open(merges_filepath, "r") as f:
            merges_json = f.read()
            merges = json.loads(merges_json)["merges"]

            merges = [
                (item[0].encode("latin-1"), item[1].encode("latin-1"))
                for item in merges
            ]
            # print(merges)

        return Tokenizer(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        # Encode an input text into a sequence of token IDs.

        SPLIT_SPECIAL_TOKEN_PATTERN = (
            "(" + "|".join(re.escape(sep) for sep in self.special_tokens) + ")"
        )

        split_texts = (
            re.split(SPLIT_SPECIAL_TOKEN_PATTERN, text)
            if len(self.special_tokens) > 0
            else [text]
        )

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        result = []
        for chunk in split_texts:
            if chunk in self.special_tokens:
                result.append(self.vocab_rev[chunk.encode()])
                continue

            matches = re.finditer(PAT, chunk)

            for match in matches:
                token = match.group()
                # text_bytes = token.encode('UTF-8')

                bytes_list = [bytes([b]) for b in token.encode("UTF-8")]

                # bytes_list = list(text_bytes)
                while True:

                    pair_candidate = None
                    pair_candidate_idx = -1
                    pair_candidate_merges_idx = len(self.merges) + 1
                    for i in range(len(bytes_list) - 1):
                        pair = (bytes_list[i], bytes_list[i + 1])
                        if pair in self.merges_map:
                            idx = self.merges_map[pair]
                            if idx < pair_candidate_merges_idx:
                                pair_candidate = pair
                                pair_candidate_merges_idx = idx
                                pair_candidate_idx = i

                    if pair_candidate is not None:
                        merged_bytes = pair_candidate[0] + pair_candidate[1]
                        bytes_list[pair_candidate_idx] = merged_bytes
                        bytes_list.pop(pair_candidate_idx + 1)
                    else:
                        break

                for item in bytes_list:
                    result.append(self.vocab_rev[item])
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # Given an iterable of
        # strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        # required for memory-efficient tokenization of large files that we cannot directly load into
        # memory.

        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        # Decode a sequence of token IDs into text.

        final_bytes = b""
        for id in ids:
            bytes = self.vocab[id]
            final_bytes += bytes

        return final_bytes.decode("UTF-8", errors="replace")


def calculate_analytics(og_text, encoded_ints, time_in_secs):
    ints = og_text.encode("UTF-8")
    compression_ratio = len(ints) / len(encoded_ints)
    print("compression_ratio : " + str(compression_ratio))

    num_original_bytes = len(ints)
    throughput = num_original_bytes / time_in_secs
    print(f"Tokeniser throughput : {throughput } Bytes per second ")
    compression_ratio = len(ints) / len(encoded_ints)

    target_file_size = 825  # GB
    # average_throughput = 12500 # Bytes per second
    time_in_secs = (target_file_size * 1024 * 1024 * 1024) / throughput
    print(f"Time for 825 GB file : { time_in_secs/(60*60*24) } days")

if __name__ == "__main__":

    # tokeniser = Tokenizer.from_files(
    #     "/Users/rahul/Documents/code/rahul/cs336/assignment1-basics/vocab.json",
    #     "/Users/rahul/Documents/code/rahul/cs336/assignment1-basics/merges.json",
    #     ["<|endoftext|>"],
    # )
    # print(type(tokeniser.encode("hello")[0]))

    with open(
        "/Users/rahul/Documents/code/rahul/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt",
        "r",
    ) as f:

        # can optimise this as well like train_bpe TODO
        text = f.read()

        SPLIT_SPECIAL_TOKEN_PATTERN =  "(" + re.escape("<|endoftext|>") + ")"

        split_texts = re.split(SPLIT_SPECIAL_TOKEN_PATTERN, text)

        start = time.time()
        with multiprocessing.Pool(processes=os.cpu_count(),initializer=initialize_worker,
        initargs=("/Users/rahul/Documents/code/rahul/cs336/assignment1-basics/vocab-train.json",
                    "/Users/rahul/Documents/code/rahul/cs336/assignment1-basics/merges-train.json",
                    ["<|endoftext|>"]),) as pool:
            pool_results = pool.map(
                encode_text,
                split_texts,
            )
        encoded_text = [num for sublist in pool_results for num in sublist]
        end = time.time()
        calculate_analytics(text, encoded_text, end - start)
        print('multi done ', len(encoded_text))


        # # export the encoded values as np array
        # encoded_text = tokeniser.encode(text)
        print(f"total tokens : {len(encoded_text)} " )
        if max(encoded_text) >= 65536:
            raise Exception("Number overflow. vocab is too big for uint16")
        np_uint16 = np.asarray(encoded_text, dtype = np.uint16)
        np.save('training_data_encoded', np_uint16)

        # # some alaytics
        # SPLIT_SPECIAL_TOKEN_PATTERN = re.escape("<|endoftext|>")
        # texts_arr = re.split(SPLIT_SPECIAL_TOKEN_PATTERN, text)
        # for document in texts_arr:
        #     ints = document.encode("UTF-8")
        #     num_original_bytes = len(ints)
        #     encode_start = time.time()
        #     encoded = tokeniser.encode(document)
        #     encode_end = time.time()
        #     throughput = num_original_bytes / (encode_end - encode_start)
        #     print(f"Tokeniser throughput : {throughput } Bytes per second ")
        #     compression_ratio = len(ints) / len(encoded)
        #     print("compression_ratio : " + str(compression_ratio))

        #     target_file_size = 825  # GB
        #     # average_throughput = 12500 # Bytes per second
        #     time_in_secs = (target_file_size * 1024 * 1024 * 1024) / throughput
        #     print(f"Time for 825 GB file : { time_in_secs/(60*60*24) } days")
