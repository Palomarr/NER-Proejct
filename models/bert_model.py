import torch
import torch.nn as nn
from transformers import BertModel

class ClassicBERTNER(nn.Module):
    def __init__(self, num_labels):
        super(ClassicBERTNER, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100) 
            # Reshape for loss computation
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, logits.shape[-1])[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            return loss
        else:
            return torch.argmax(logits, dim=-1)
