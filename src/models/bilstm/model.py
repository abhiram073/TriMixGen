import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiLSTM_LID(nn.Module):
    """
    Standard Bidirectional LSTM for Token-Level Sequence Labeling.
    No attention mechanism. Pure recurrent state representation.
    """
    def __init__(self, vocab_size, num_classes, embedding_dim=128, hidden_dim=256, num_layers=1, dropout=0.3, padding_idx=0):
        super(BiLSTM_LID, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Multiply by 2 because it's bidirectional
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x, lengths):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x) # (batch_size, seq_len, emb_dim)
        embedded = self.dropout(embedded)
        
        # Pack sequence for efficient RNN computation (ignores padded elements in the recurrent state)
        packed_embedded = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        packed_output, _ = self.lstm(packed_embedded)
        
        # Unpack sequence back to (batch_size, seq_len, hidden_dim*2)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out) # (batch_size, seq_len, num_classes)
        
        return logits
