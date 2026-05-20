import os
import time
import argparse
from tqdm import tqdm
from collections import Counter
from multiprocessing import Pool, cpu_count
from bpe_tokenizer import BPETokenizer

# Global worker variable
worker_tokenizer = None


def init_worker(tokenizer_path):
    """Runs once per CPU process to load the tokenizer into RAM."""
    global worker_tokenizer
    worker_tokenizer = BPETokenizer()
    worker_tokenizer.load(tokenizer_path)


def encode_sentence_worker(sentence):
    """Encodes using the persistent worker_tokenizer."""
    return worker_tokenizer.encode(sentence)


def main(args):
    corpus = []
    with open(args.input_corpus_path, 'r', encoding='utf-8') as f:
        corpus = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(corpus)} sentences.")
    num_processes = args.num_processes or cpu_count()

    # 1. PARALLEL ENCODING
    print(f"Encoding on {num_processes} CPUs...")
    start_time = time.time()
    with Pool(processes=num_processes, initializer=init_worker, initargs=(args.tokenizer_path,)) as pool:
        encoded_corpus = list(tqdm(
            pool.imap(encode_sentence_worker, corpus, chunksize=250),
            total=len(corpus),
            desc="Encoding"
        ))
    print(f"Encoding Finished in {time.time() - start_time:.2f}s")

    # 2. LOAD LOCAL TOKENIZER FOR DECODING/METRICS
    tokenizer = BPETokenizer()
    tokenizer.load(args.tokenizer_path)
    unk_id = tokenizer.get_unk_id()

    # 3. CONSISTENCY TEST (Optimized Decode)
    print("Running consistency test...")
    inconsistent_count = 0
    for i in range(len(corpus)):
        reconstructed = tokenizer.decode(encoded_corpus[i])
        if corpus[i] != reconstructed:
            inconsistent_count += 1
            if inconsistent_count < 5:  # Only print first 5 errors to avoid spam
                print(f"Mismatch at {i}:\nOrig: {corpus[i]}\nDeco: {reconstructed}")

    # 4. COMPRESSION RATIO
    total_original_len = sum(len(s) for s in corpus)
    # Subtract 1 per sentence to exclude the <END> token from the ratio if desired
    total_encoded_len = sum(len(ids) - 1 for ids in encoded_corpus)
    compression_ratio = total_encoded_len / total_original_len if total_original_len > 0 else 0

    # 5. OOV RATE & FREQUENCY ANALYSIS
    token_counts = Counter()
    oov_tokens = 0
    for ids in encoded_corpus:
        token_counts.update(ids)
        oov_tokens += ids.count(unk_id)

    total_tokens = sum(token_counts.values())
    oov_rate = oov_tokens / total_tokens if total_tokens > 0 else 0

    # 6. LONG-TAIL ANALYSIS (Threshold = 5)
    threshold = 5
    rare_tokens = [t for t, count in token_counts.items() if count < threshold]
    long_tail_score = len(rare_tokens) / len(token_counts) if token_counts else 0

    # --- OUTPUT RESULTS ---
    print("\n" + "=" * 30)
    print(f"Consistency: {'PASSED' if inconsistent_count == 0 else 'FAILED'}")
    if inconsistent_count > 0:
        print(f"Inconsistent Sentences: {inconsistent_count}")
    print(f"Compression Ratio: {compression_ratio:.4f}")
    print(f"OOV Rate: {oov_rate:.4f}")
    print(f"Long-tail Score (<{threshold} occurrences): {long_tail_score:.4f}")
    print("=" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_corpus_path', type=str, required=True)
    parser.add_argument('--tokenizer_path', type=str, required=True)
    parser.add_argument('--num_processes', type=int, default=None)
    main(parser.parse_args())