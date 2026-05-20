import os.path
from collections import Counter, defaultdict
import re
import heapq
import pickle
import unicodedata


class BPETokenizer:
    HINDI_REGEX = r"Ð[\u0900-\u097F\w]+|[\u0900-\u097F\w]+|\S"
    MIN_OCC_TOK = 15
    class CharNode:
        
        def __init__(self, char, prev, next, freq, in_vocab=True):
            self.char = char
            self.prev: BPETokenizer.CharNode = prev
            self.next: BPETokenizer.CharNode = next
            self.freq = freq
            self.in_vocab = in_vocab
            self.active = True

        def merge(self, char, next, next_next):
            self.char = self.char + char
            self.next = next_next

            if next_next:
                next_next.prev = self

    def __init__(self, vocab_size=50000, special_tokens=None):
        self.vocab_size = vocab_size
        if special_tokens is None:
            self.special_tokens = ["<PAD>","<STA>","<END>","<UNK>"]
        self.vocab = {}
        self.id2token = {2:"<STA>",3:"<END>",1:"<UNK>","<PAD>":0}
        self.token2id = {"<STA>":2,"<END>":3,"<UNK>":1,"<PAD>":0}
        self.token_id = len(self.token2id)
        self.word_dll = {}
        self.word_dict = Counter()
        self.pair_count = defaultdict(int)
        self.pair_ptr = defaultdict(set)
        self.merges = {}
        
    def create_dll_word(self, word, count=0):
        first = None
        prev = None
        next = None
        for char in word:
            temp = self.CharNode(char, prev, next, count)
            if first is None:
                first = temp
            if prev is not None:
                prev.next = temp
            prev = temp
        return first


    def create_word_dict(self, corpus):
        for line in corpus:
            #line = unicodedata.normalize('NFC', line)
            line = line.replace(" ", " Ð")
            self.word_dict.update(re.findall(BPETokenizer.HINDI_REGEX, line, re.UNICODE))
        # print(self.word_dict.keys())
        for word, count in self.word_dict.items():
            self.word_dll[word] = self.create_dll_word(word, count)


    def get_initial_bytes(self, corpus):
        vocab = Counter()
        for word in self.word_dict.keys():
            vocab.update(word)
        self.vocab = vocab

        for tok, count in self.vocab.items():
            self.token2id[tok] = self.token_id
            self.id2token[self.token_id] = tok
            self.token_id+=1


    def train(self, corpus):
        #print("Starting tokenizer Training")
        self.create_word_dict(corpus)
        self.get_initial_bytes(corpus)

        self.pair_count = Counter()
        self.pair_ptr = defaultdict(set)

        for word, count in self.word_dict.items():
            word_dll = self.word_dll[word]

            cur: BPETokenizer.CharNode = word_dll
            while cur.next:
                self.pair_count[(cur.char,cur.next.char)] += count
                self.pair_ptr[(cur.char,cur.next.char)].add(cur)
                cur = cur.next

        maxheap = [(-count, pair) for pair, count in self.pair_count.items()]
        heapq.heapify(maxheap)

        while self.token_id < self.vocab_size and maxheap:
            count, pair = heapq.heappop(maxheap)
            count = -count

            if count != self.pair_count[pair]:
                continue

            if count < BPETokenizer.MIN_OCC_TOK:
                break

            pair_ptr = self.pair_ptr[pair]
            cur_char, next_char = pair
            self.token2id[cur_char+next_char]=self.token_id
            self.id2token[self.token_id] = cur_char+next_char
            self.merges[(cur_char, next_char)] = self.token_id
            self.token_id+=1

            mod_pairs = set()

            for ptr in list(pair_ptr):
                if (not ptr.active) or not ptr.next or (not ptr.next.active):
                    continue

                local_count = ptr.freq
                cur_next = ptr.next

                if ptr.prev is not None:
                    p = (ptr.prev.char,cur_char)
                    self.pair_count[p] -= local_count
                    self.pair_ptr[p].discard(ptr.prev)
                    mod_pairs.add(p)

                if ptr.next.next is not None:
                    p= (next_char, ptr.next.next.char)
                    self.pair_count[p] -= local_count
                    self.pair_ptr[p].discard(ptr.next)
                    mod_pairs.add(p)

                ptr.merge(next_char, ptr.next, ptr.next.next)
                cur_next.active = False

                cur_next.prev = None
                cur_next.next = None

                if ptr.prev:
                    new_l = (ptr.prev.char, ptr.char)
                    self.pair_count[new_l] += local_count
                    self.pair_ptr[new_l].add(ptr.prev)
                    mod_pairs.add(new_l)

                if ptr.next:
                    new_r = (ptr.char, ptr.next.char)
                    self.pair_count[new_r] += local_count
                    self.pair_ptr[new_r].add(ptr)
                    mod_pairs.add(new_r)

            for p in mod_pairs:
                if self.pair_count[p] > 0:
                    heapq.heappush(maxheap, (-self.pair_count[p], p))

            del self.pair_ptr[pair]
            self.pair_count[pair] = 0
        #print("Tokenizer Training Done")
        print(self.get_vocab_size())
        self.word_dll.clear()
        self.word_dict.clear()
        self.pair_count.clear()
        self.pair_ptr.clear()

    def encode(self, text):
        #print("Encoding Text")
        #text = unicodedata.normalize('NFC', text)
        text = text.replace(" ", " Ð")
        words = re.findall(BPETokenizer.HINDI_REGEX, text, re.UNICODE)

        tokenized_text = [self.token2id["<STA>"]]

        for word in words:

            word_first = self.create_dll_word(word)
            cur = word_first

            while cur:
                if self.token2id.get(cur.char, None) is None:
                    cur.in_vocab = False
                cur = cur.next

            cur: BPETokenizer.CharNode = word_first
            rank_heap = []
            while cur and cur.next:
                if cur.in_vocab and cur.next.in_vocab:
                    rank = self.merges.get((cur.char,cur.next.char), None)
                    if rank is not None:
                        heapq.heappush(rank_heap,(rank,id(cur),(cur,cur.next)))
                cur = cur.next

            while rank_heap:
                rank, _ , pair_ptr = heapq.heappop(rank_heap)

                p1, p2 = pair_ptr

                if not(p1.active and p2.active):
                    continue

                p1.active = False
                p2.active = False

                new_ptr = self.CharNode(p1.char+p2.char,p1.prev,p2.next,0,True)
                if p1 == word_first:
                    word_first = new_ptr

                if p1.prev is not None:
                    p1.prev.next = new_ptr
                    rank = self.merges.get((p1.prev.char,new_ptr.char), None)
                    if rank is not None:
                        heapq.heappush(rank_heap,(rank,id(p1.prev),(p1.prev,new_ptr)))

                if p2.next is not None:
                    p2.next.prev = new_ptr
                    rank = self.merges.get((new_ptr.char, p2.next.char), None)
                    if rank is not None:
                        heapq.heappush(rank_heap,(rank,id(new_ptr),(new_ptr,p2.next)))

            cur = word_first
            unknown_id = self.get_unk_id()
            while cur:
                tokenized_text.append(self.token2id.get(cur.char,unknown_id))
                cur = cur.next
        tokenized_text.append(self.token2id["<END>"])
        #print("Encoding Done")
        return tokenized_text

    def decode(self, token_ids):
        #print("Decoding Text")
        # text = ""
        # for id in token_ids:
        #     if id == 0 or id == 2 or id == 3:
        #         continue
        #     else:
        #         text = text + self.id2token[id].replace("Ð", " ")
        # #print("Decoding Done")
        # #text = unicodedata.normalize('NFC', text)
        # return text

        tokens = []
        for id in token_ids:
            if id in [0, 2, 3]:
                continue
            token = self.id2token.get(id, "<UNK>")
            tokens.append(token.replace("Ð", " "))

        return "".join(tokens)


    def save(self, filepath):
        with open(os.path.join(filepath,"tokenizer.pkl"), 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        #print(f"Saved")


    def load(self,filepath):
        class UnpicklerCustom(pickle.Unpickler):
            def find_class(self, module, name):
                if name == 'BPETokenizer':
                    return BPETokenizer
                if name == 'CharNode':
                    return BPETokenizer.CharNode
                return super().find_class(module, name)

        with open(os.path.join(filepath, "tokenizer.pkl"), 'rb') as f:
            temp_obj = UnpicklerCustom(f).load()

        self.__dict__.update(temp_obj.__dict__)
        #print(f"Loaded")
    
    def get_vocab_size(self):
        return len(self.token2id)
    
    def get_unk_id(self):
        return self.token2id["<UNK>"]
