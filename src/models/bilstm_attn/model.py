import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Attention(nn.Module):
    """
    Standard Luong-style dot-product attention over BiLSTM hidden states.
    Allows the model to weight different tokens differently for classification.
    """
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        # Project hidden state to calculate attention scores
        self.attention = nn.Linear(hidden_dim * 2, 1, bias=False)

    def forward(self, lstm_out, lengths):
        # lstm_out: (batch_size, seq_len, hidden_dim * 2)
        
        # Calculate attention scores: (batch_size, seq_len, 1)
        scores = self.attention(lstm_out)
        scores = scores.squeeze(-1) # (batch_size, seq_len)
        
        # Mask out padded positions
        mask = torch.arange(lstm_out.size(1), device=lstm_out.device)[None, :] >= lengths[:, None]
        scores.masked_fill_(mask, -1e9)
        
        # Softmax to get probability distribution over the sequence
        weights = F.softmax(scores, dim=-1) # (batch_size, seq_len)
        
        # Apply weights to lstm outputs. 
        # Here we use self-attention for each token in sequence tagging, meaning 
        # we fuse the token's own hidden state with the sequence context. 
        # Wait, for sequence labeling, standard attention is usually local attention 
        # or contextualized self-attention, because we need an output for EVERY token.
        
        # Context vector for each token by attending to all other tokens
        # context: (batch, seq_len, hidden_dim * 2)
        # weights: (batch, seq_len) -> we need a weights matrix for (batch, seq_len, seq_len)
        
        # Alternatively, for token-level classification, attention is often implemented 
        # as a feed-forward attention that simply projects the BiLSTM state into a 
        # richer representation, OR we use standard self-attention (Q=K=V).
        pass

class SelfAttention(nn.Module):
    """
    Contextual Self-Attention mechanism for sequence labeling.
    Computes an attention matrix that allows every token to attend to every other token.
    """
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_dim * 2, hidden_dim)
        self.key = nn.Linear(hidden_dim * 2, hidden_dim)
        self.value = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        
    def forward(self, lstm_out, lengths):
        # lstm_out: (batch, seq_len, hidden_dim * 2)
        Q = self.query(lstm_out) # (batch, seq_len, hidden_dim)
        K = self.key(lstm_out)   # (batch, seq_len, hidden_dim)
        V = self.value(lstm_out) # (batch, seq_len, hidden_dim*2)
        
        # Attention scores: (batch, seq_len, seq_len)
        scores = torch.bmm(Q, K.transpose(1, 2)) / (Q.size(-1) ** 0.5)
        
        # Masking padding tokens
        max_len = lstm_out.size(1)
        mask = torch.arange(max_len, device=lstm_out.device)[None, :] >= lengths[:, None]
        # mask is (batch, seq_len). We need to broadcast it to (batch, seq_len, seq_len)
        mask = mask.unsqueeze(1).expand_as(scores)
        
        scores.masked_fill_(mask, -1e9)
        weights = F.softmax(scores, dim=-1)
        
        # Context vectors: (batch, seq_len, hidden_dim*2)
        context = torch.bmm(weights, V)
        return context, weights

class BiLSTMAttention_LID(nn.Module):
    def __init__(self, vocab_size, num_classes, embedding_dim=128, hidden_dim=256, num_layers=1, dropout=0.3, padding_idx=0):
        super(BiLSTMAttention_LID, self).__init__()
        
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
        
        self.attention = SelfAttention(hidden_dim)
        
        # Fully connected layer takes the concatenated LSTM output and Attention context
        self.fc = nn.Linear(hidden_dim * 2 * 2, num_classes)
        
    def forward(self, x, lengths):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        packed_embedded = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_embedded)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Apply Self-Attention
        context, attn_weights = self.attention(lstm_out, lengths)
        
        # Combine recurrent context with global sequence attention context
        combined = torch.cat([lstm_out, context], dim=-1)
        combined = self.dropout(combined)
        
        logits = self.fc(combined)
        return logits, attn_weights
