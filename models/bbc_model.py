import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel
from config import BERT_MODEL

# Define constants
EMBEDDING_DIM = 768  # Embedding dimension (BERT default)
HIDDEN_SIZE = 256  # LSTM hidden size



class Model(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL)
        self.lstm = nn.LSTM(
            EMBEDDING_DIM,
            HIDDEN_SIZE,
            batch_first=True,
            bidirectional=True,
        )
        self.linear = nn.Linear(2 * HIDDEN_SIZE, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def _get_lstm_feature(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        lstm_output, _ = self.lstm(sequence_output)
        emissions = self.linear(lstm_output)
        return emissions

    def forward(self, input_ids, attention_mask, labels=None):
        emissions = self._get_lstm_feature(input_ids, attention_mask)
        if labels is not None:
            # Compute loss
            loss = -self.crf(emissions, labels, mask=attention_mask, reduction='mean')
            return loss
        else:
            # Decode tags
            predictions = self.crf.decode(emissions, mask=attention_mask)
            return predictions


