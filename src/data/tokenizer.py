import json

class Tokenizer:
    def __init__(self, vocab_file, merges_file):
        with open(vocab_file, 'r') as vf, open(merges_file, 'r') as mf:
            self.vocab = json.load(vf)
            self.merges = mf.read().splitlines()

    def tokenize(self, text):
        # Placeholder for tokenization logic (e.g., BPE tokenization)
        return [self.vocab.get(token, self.vocab['<unk>']) for token in text.split()]