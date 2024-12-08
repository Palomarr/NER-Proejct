import torch
import torch.nn as nn
from torchcrf import CRF

class ClassicBiLSTMNER(nn.Module):
    def __init__(self, num_labels, embedding_dim=300, hidden_dim=256, dropout=0.1, vocab_size=10000):
        super(ClassicBiLSTMNER, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)  # Adjust vocab_size and padding_idx
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
    
    def forward(self, input_ids, attention_mask, labels=None):
        embeds = self.embedding(input_ids)
        lstm_out, _ = self.bilstm(embeds)
        lstm_out = self.dropout(lstm_out)
        emissions = self.classifier(lstm_out)
        
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attention_mask, reduction='mean')
            return loss
        else:
            return self.crf.decode(emissions, mask=attention_mask)
