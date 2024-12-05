import torch.nn as nn
from torchcrf import CRF
import torch
from transformers import BertModel
from transformers import BertTokenizer

# Define constants
EMBEDDING_DIM = 768  # Embedding dimension (BERT default)
HIDDEN_SIZE = 256  # LSTM hidden size
TARGET_SIZE = 31  # Number of target classes (adjust as needed)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.lstm = nn.LSTM(
            EMBEDDING_DIM,
            HIDDEN_SIZE,
            batch_first=True,
            bidirectional=True,
        )
        self.linear = nn.Linear(2 * HIDDEN_SIZE, TARGET_SIZE)
        self.crf = CRF(TARGET_SIZE)

    def _get_lstm_feature(self, input, mask):
        # Pass input and mask to BERT to get embeddings
        out = self.bert(input, attention_mask=mask)[0]
        out, _ = self.lstm(out)
        return self.linear(out)

    def forward(self, input, mask):
        # Generate features and decode using CRF
        out = self._get_lstm_feature(input, mask)
        print(f"Mask dtype before CRF: {mask.dtype}")
        return self.crf.decode(out, mask)

    def loss_fn(self, input, target, mask):
        # Calculate negative log-likelihood loss using CRF
        y_pred = self._get_lstm_feature(input, mask)
        return -self.crf(y_pred, target, mask, reduction='mean')


if __name__ == '__main__':
    # Initialize the model
    model = Model()
    
    # Use the tokenizer to generate input IDs
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    texts = ["你好，世界！"] * 100  # Sample texts
    encoding = tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=50
    )

    input = encoding['input_ids']  # Shape: (batch_size, seq_len)
    mask = encoding['attention_mask'].bool()  # Shape: (batch_size, seq_len)

    # Target tensor for CRF (batch_size, seq_len)
    seq_len = input.shape[1]
    target = torch.randint(0, TARGET_SIZE, (input.shape[0], seq_len))

    # Forward pass
    output = model(input, mask)
    print(f"Output length: {len(output)}, Sequence length: {len(output[0])}")

    # Calculate loss
    loss = model(input, mask, labels=target)
    print(f"Loss: {loss.item()}")
