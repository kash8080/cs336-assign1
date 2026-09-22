import os
import regex as re
import json
import time
from cs336_basics.pretokenization_example import find_chunk_boundaries
import multiprocessing
from collections import Counter


# must be a top-level function (see gotchas)
def get_token_map_from_str(input_path, special_tokens, start, end): 
    pre_tokens_dict = {}  

    SPLIT_SPECIAL_TOKEN_PATTERN = "|".join(re.escape(sep) for sep in special_tokens)
           
    # pre tokenise all the text in the input file using gpt 2 regex
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        split_texts = re.split(SPLIT_SPECIAL_TOKEN_PATTERN, chunk)
        
    for chunk in split_texts:
        matches = re.finditer(PAT, chunk)
        for match in matches:
            token = match.group()
            if token in pre_tokens_dict:
                pre_tokens_dict[token] += 1
            else:
                pre_tokens_dict[token] = 1

    return Counter(pre_tokens_dict)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the input text file and return the learned
    vocabulary and merges.

    Returns:
        vocab: A dictionary mapping token indices to byte strings.
        merges: A list of tuples representing the learned merges.
    """

    start = time.time()
    vocab = {}
    merges = []

    # add all the bytes to the vocab
    for i in range(256):
        # a length-1 bytes object holding the single byte i → b'-'.
        # You pass an iterable of integers, and each integer becomes one byte.
        vocab[i] = bytes([i])
    # add all special tokens (page end etc) to the vocab
    for st in special_tokens:
        vocab[len(vocab)] = st.encode("utf-8")

    needed_vocab = vocab_size - 256 - len(special_tokens)
    if needed_vocab <= 0:
        return vocab, []

    # split input file on special tokens
    splitted_texts_boundaries = get_splitted_texts_boundaries(input_path, special_tokens)
    phase1time = time.time()
    print(f"time taken for splitting chunks : {phase1time-start}")

    pre_tokens_dict = {}
    with multiprocessing.Pool(processes= os.cpu_count()) as pool:
        pool_results = pool.starmap(get_token_map_from_str, splitted_texts_boundaries)

    # print(pool_results)
    total_counts = Counter()
    for count in pool_results:
        total_counts.update(count)

    # print('total_counts ',total_counts)

    phase2time = time.time()
    print(f"time taken for splitting tokens : {phase2time-phase1time}")
            
    # [word] : (parts, freq)
    # to save the word/token with the individual parts and freq
    token_freq_map = {}

    # ([bytes], [bytes]) : number (freq)
    pair_freq = {}

    # create the initial dict with all words and their parts,freq
    for token in total_counts:
        token_freq_map[token] = ([bytes([b]) for b in token.encode("UTF-8")], total_counts[token])



    pair_freq, pair_token_map = create_pairs(token_freq_map)

    while len(merges) < needed_vocab:
        print(f'vocab size {len(vocab)}')
        # print(f'merge size {len(merges)}')
        
        # fetch most freq byte pair, if tie, then fetch lexigraphically bigger one. 
        max_freq_pair = max(pair_freq, key= lambda k: (pair_freq[k], k))

        merged = False

        # print(f"max freq pair {max_freq_pair} : {pair_freq[max_freq_pair]}")

        # merge this pair and update both token_freq_map and pair_freq
        tokens_for_pair = pair_token_map[max_freq_pair]
        if(not tokens_for_pair):
            print('tokens not available for pair. something is wrong!')
            break

        for token in tokens_for_pair:
            byte_list,freq = token_freq_map[token]

            i=0
            while i < len(byte_list)-1:
                bp = (byte_list[i], byte_list[i+1])
                if(max_freq_pair == bp):
                    new_bytes = byte_list[i] + byte_list[i+1]

                    # update the pair_freq directly here otherwise we will have to recmopute it for every pair
                    # this is optimisation to speed up
                    pair_freq[max_freq_pair] = 0 
                    pair_token_map[max_freq_pair] = set()

                    # now we have a new pair with this new bytes and left/right byte if any
                    if(i>0):
                        # there are bytes to left
                        # remove the current left pair as that is not valid now as curent window is merged
                        old_left_tuple = (byte_list[i-1], byte_list[i])
                        pair_freq[old_left_tuple] -= freq
                        # token can have the multiple pair in it so can't remove it
                        # pair_token_map[old_left_tuple].discard(token)


                        new_left_tuple = (byte_list[i-1], new_bytes)
                        if(new_left_tuple in pair_freq):
                            pair_token_map[new_left_tuple].add(token)
                            pair_freq[new_left_tuple] = pair_freq[new_left_tuple] + freq
                        else:
                            pair_token_map[new_left_tuple] = set([token])
                            pair_freq[new_left_tuple] = freq
                        

                    if(i < len(byte_list) - 2):
                        # there are more bytes to right
                        # remove the current right pair as that is not valid now as curent window is merged
                        old_right_tuple = (byte_list[i+1], byte_list[i+2])
                        pair_freq[old_right_tuple] -= freq
                        # token can have the multiple pair in it so can't remove it
                        # pair_token_map[old_right_tuple].discard(token)

                        new_right_tuple = (new_bytes, byte_list[i+2])
                        if(new_right_tuple in pair_freq):
                            pair_token_map[new_right_tuple].add(token)
                            pair_freq[new_right_tuple] = pair_freq[new_right_tuple] + freq
                        else:
                            pair_token_map[new_right_tuple] = set([token])
                            pair_freq[new_right_tuple] = freq

                    byte_list[i] = new_bytes
                    byte_list.pop(i+1)
                    
                    # print(f"updated { b"".join(token_freq_map[token][0]).decode("UTF-8")}")
                    # print(f"updated {token_freq_map[token][0]}")
                    if not merged:
                        merged=True
                        merges.append(bp)
                        vocab[len(vocab)] = byte_list[i]
                    # print(l[i].decode("UTF-8"))
                    i+=1
                else:
                    i+=1 
            
        if not merged:
            # couldn't find any matches for the most freq pair somehow.
            # something is wrong. 
            print(token_freq_map)
            break





    # print(f"pre_tokens_dict: {pre_tokens_dict}")
    # print(f"token_freq_map: {v}")
    # print(f"pair_freq: {pair_freq}")
    # print(f"max_freq_pair: {max_freq_pair} freq:{pair_freq[max_freq_pair]}" )
    # print(f"vocab: {vocab}" )
    phase3time = time.time()
    print(f"time taken for splitting chunks : {phase1time-start}")
    print(f"time taken for splitting tokens : {phase2time-phase1time}")
    print(f"time taken for creating vocab : {phase3time-phase2time}")

    # Implementation of the BPE training algorithm goes here
    return vocab, merges


def get_splitted_texts(
    input_path: str | os.PathLike, special_tokens: list[str]
) -> list[str]:
    """
    Split the input text file into chunks based on the special tokens.
    Returns a list of strings, each representing a chunk of text.
    """

    # pattern = "|".join(re.escape(sep) for sep in special_tokens)
    # with open(input_path, "r", encoding="utf-8") as f:
    #     text = f.read()
    #     split_text = re.split(pattern, text)
    # return split_text

    l = []
    with open(input_path, "rb") as f:
        num_processes = 100
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            l.append(chunk)
            # Run pre-tokenization on your chunk and store the counts for each pre-token
    return l


def get_splitted_texts_boundaries(
    input_path: str | os.PathLike, special_tokens: list[str]
) -> list[str]:
    """
    Split the input text file into chunks based on the special tokens.
    Returns a list of strings, each representing a chunk of text.
    """

    # pattern = "|".join(re.escape(sep) for sep in special_tokens)
    # with open(input_path, "r", encoding="utf-8") as f:
    #     text = f.read()
    #     split_text = re.split(pattern, text)
    # return split_text

    l = []
    with open(input_path, "rb") as f:
        num_processes = 100
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            l.append((input_path,special_tokens, start,end))
    return l

def create_pairs(token_freq_map) -> dict[(bytes,bytes) : int]:
    pair_freq={}
    pair_token_map={}

    # create the initial freq of adjacent pairs as well
    for pretoken in token_freq_map:
        l,freq = token_freq_map[pretoken]
        for i in range(0, len(l) - 1):
            key = (l[i], l[i+1])

            if(key in pair_freq): 
                pair_freq[key] += freq
            else:
                pair_freq[key] = freq

            if key in pair_token_map:
                pair_token_map[key].add(pretoken)
            else:
                pair_token_map[key] = set([pretoken])

    return pair_freq, pair_token_map


def train_bpe_tinystories():   
    start = time.time()

    vocab, merges = train_bpe(
        "/Users/rahul/Documents/code/rahul/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt",
        10000, # 7 secs for 5000
        ["<|endoftext|>"],
    )
    end = time.time()
    print(f"time taken {end-start}")

    # max_len_vocab = max(vocab, key = lambda k: len(vocab[k]))
    # for key in vocab:
    #     print(vocab[key])
    # print(f"max_len_vocab {vocab[max_len_vocab]}")
    print('training done--')


    op_json = {}
    for vb in vocab:
        # Latin-1 maps bytes 0–255 ↔ code points 0–255 one-to-one
        op_json[vb] = vocab[vb].decode('latin-1')
    
    op_merges = {"merges": []}
    for merge in merges:
        op_merges["merges"].append([merge[0].decode('latin-1'),merge[1].decode('latin-1')])

    # Save the dictionary to a file
    with open("vocab.json", "w") as file:
        json.dump(op_json, file, indent=4)

    # Save the dictionary to a file
    with open("merges.json", "w") as file:
        json.dump(op_merges, file, indent=4)


if __name__ == "__main__":
    train_bpe_tinystories()
    
# train_bpe(
#     "/Users/rahul/Documents/code/rahul/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt",
#     1000,
#     ["<|endoftext|>"],
# )
