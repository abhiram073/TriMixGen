import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
import logging

logger = logging.getLogger(__name__)

class MBERTLIDDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer: BertTokenizerFast, label2id, max_length=256):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.id2label = {v: k for k, v in label2id.items()}
        
    def __len__(self):
        return len(self.sentences)
        
    def __getitem__(self, idx):
        words = self.sentences[idx]
        tags = self.labels[idx]
        
        # FastTokenizer gives us word_ids which we use to align labels to subwords!
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )
        
        word_ids = encoding.word_ids(batch_index=0)
        
        label_ids = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens like [CLS] and [SEP]
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # First subword of a new word
                # If truncation chopped off tags, handle it
                if word_idx < len(tags):
                    label_ids.append(self.label2id.get(tags[word_idx], -100))
                else:
                    label_ids.append(-100)
            else:
                # Subsequent subwords of the same word get -100 so they don't contribute to loss
                label_ids.append(-100)
                
            previous_word_idx = word_idx
            
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label_ids, dtype=torch.long)
        return item
