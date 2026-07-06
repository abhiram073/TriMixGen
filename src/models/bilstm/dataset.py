import torch
from torch.utils.data import Dataset
from collections import Counter

class Vocabulary:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.token2id = {"<PAD>": 0, "<UNK>": 1}
        self.id2token = {0: "<PAD>", 1: "<UNK>"}
        self.label2id = {"<PAD>": 0, "En": 1, "Te": 2, "Mixed": 3, "NE": 4, "Univ": 5}
        self.id2label = {0: "<PAD>", 1: "En", 2: "Te", 3: "Mixed", 4: "NE", 5: "Univ"}
        
    def build(self, sentences, labels):
        # Build token vocab
        word_counts = Counter(token.lower() for seq in sentences for token in seq)
        for word, count in word_counts.items():
            if count >= self.min_freq:
                idx = len(self.token2id)
                self.token2id[word] = idx
                self.id2token[idx] = word
                
        # Build label vocab
        label_set = set(label for seq in labels for label in seq)
        for label in sorted(list(label_set)):
            if label not in self.label2id:
                idx = len(self.label2id)
                self.label2id[label] = idx
                self.id2label[idx] = label
                
    def encode_tokens(self, tokens):
        return [self.token2id.get(token.lower(), self.token2id["<UNK>"]) for token in tokens]
        
    def encode_labels(self, labels):
        return [self.label2id[label] for label in labels]

class SequenceDataset(Dataset):
    def __init__(self, sentences, labels, vocab, max_length=None):
        self.sentences = sentences
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.sentences)
        
    def __getitem__(self, idx):
        tokens = self.sentences[idx]
        tags = self.labels[idx]
        
        if self.max_length:
            tokens = tokens[:self.max_length]
            tags = tags[:self.max_length]
            
        token_ids = self.vocab.encode_tokens(tokens)
        label_ids = self.vocab.encode_labels(tags)
        
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label_ids, dtype=torch.long)

def collate_fn(batch):
    sequences, labels = zip(*batch)
    
    # Pad sequences
    seq_lengths = torch.tensor([len(s) for s in sequences])
    padded_seqs = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    
    return padded_seqs, padded_labels, seq_lengths
