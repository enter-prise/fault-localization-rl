# utils/embedding.py

class SimpleEmbedder:
    def __init__(self):
        self.vocab = {}

    def build_vocab(self, texts):
        idx = 0
        for text in texts:
            for word in text.lower().split():
                if word not in self.vocab:
                    self.vocab[word] = idx
                    idx += 1

    def encode(self, text):
        vec = [0] * len(self.vocab)

        for word in text.lower().split():
            if word in self.vocab:
                vec[self.vocab[word]] += 1

        return vec