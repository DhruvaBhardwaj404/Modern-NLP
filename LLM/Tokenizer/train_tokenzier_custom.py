import os
import time
from partb.bpe_tokenizer import BPETokenizer
import tqdm


def test_tokenizer_consistency(tokenizer, corpus, encoded_corpus):
    consistent = True
    inconsistent_sentences = 0
    for i in range(len(corpus)):
        original_sentence = corpus[i]
        encoded_tokens = encoded_corpus[i]
        reconstructed_sentence = tokenizer.decode(encoded_tokens)
        # assert original_sentence == reconstructed_sentence, f"Decoded text does not match original for sentence {i}!"
        if original_sentence != reconstructed_sentence:
            consistent = False
            inconsistent_sentences += 1

    return consistent, inconsistent_sentences


# Evaluate the tokenizer in terms of compression ratio
def calculate_compression_ratio(corpus, encoded_corpus):
    total_original_length = sum(len(sentence) for sentence in corpus)
    total_encoded_length = sum(len(tokens) for tokens in encoded_corpus) - len(
        corpus)  # Subtract 1 for each sentence to account for the end token
    compression_ratio = total_encoded_length / total_original_length
    return compression_ratio


# Compute out of vocabulary (OOV) rate
def calculate_oov_rate(encoded_corpus, unk_id):
    total_tokens = 0
    oov_tokens = 0
    for encoded_tokens in encoded_corpus:
        total_tokens += len(encoded_tokens)
        if unk_id is not None:
            oov_tokens += encoded_tokens.count(unk_id)
    oov_rate = oov_tokens / total_tokens if total_tokens > 0 else 0
    return oov_rate


# Analyze token frequency distribution and calculate a score based on the long tail of the distribution
def analyze_token_frequency(encoded_corpus, threshold=5):
    token_freq = {}
    for encoded_tokens in encoded_corpus:
        for token_id in encoded_tokens:
            token_freq[token_id] = token_freq.get(token_id, 0) + 1

    # Sort tokens by frequency
    sorted_tokens = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)

    # Score based on how many tokens are in the long tail (e.g., tokens that appear less than a certain threshold)
    long_tail_tokens = [token for token, freq in sorted_tokens if freq < threshold]
    long_tail_score = len(long_tail_tokens) / len(token_freq) if token_freq else 0
    return long_tail_score


def main(args):
    corpus = []
    with open(args.input_corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            corpus.append(line.strip())

    if args.train_path is not None:
        print(f"Loading training data from {args.train_path}...")
        with open(args.train_path, 'r', encoding='utf-8') as f:
            for line in f:
                corpus.append(line.strip())

    print(f"Loaded {len(corpus)} sentences from the dataset.")
    tokenizer = BPETokenizer(args.vocab_size)
    st = time.time()
    tokenizer.train(corpus)
    en = time.time()

    print(f"Training completed in {en - st:.2f} seconds.")
    os.makedirs(args.output_tokenizer_path, exist_ok=True)
    tokenizer.save(args.output_tokenizer_path)

    for c in tqdm.tqdm(corpus):
        # print(f"Sentence: {c}")
        tokenizer.encode(c)
        tokenizer.decode(tokenizer.encode(c))

    print("done")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train a BPE tokenizer on the provided dataset')
    parser.add_argument('--input_corpus_path', type=str, required=True, help='Path to the input corpus text file')
    parser.add_argument('--train_path', type=str, default=None, required=False, help='Path to the training data text file. Only used in Part C of the assignment.')
    parser.add_argument('--vocab_size', type=int, default=50000, help='Vocabulary size for the BPE tokenizer')
    parser.add_argument('--output_tokenizer_path', type=str, required=True, help='Path to save the trained tokenizer')
    args = parser.parse_args()

    main(args)
