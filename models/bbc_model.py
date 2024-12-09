import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel, BertTokenizer
import torch
from typing import Optional, List, Dict



class ImprovedBertBiLSTMCRF(nn.Module):
    def __init__(
        self, 
        num_labels: int,
        bert_model_name: str = 'bert-base-uncased',
        embedding_dim: int = 768,
        hidden_size: int = 256,
        num_lstm_layers: int = 2,  
        dropout: float = 0.5,     
        freeze_bert: bool = False,  
    ):
        super().__init__()
        
        # BERT layer
        self.bert = BertModel.from_pretrained(bert_model_name)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
                
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # Linear layer
        self.linear = nn.Linear(2 * hidden_size, num_labels)
        
        # CRF layer
        self.crf = CRF(num_labels, batch_first=True)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)


    def _get_lstm_features(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Get BERT outputs
        bert_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        # Apply layer normalization and dropout to BERT outputs
        sequence_output = self.layer_norm(bert_outputs.last_hidden_state)
        sequence_output = self.dropout(sequence_output)
        
        # BiLSTM
        lstm_output, _ = self.lstm(sequence_output)
        lstm_output = self.dropout(lstm_output)
        
        # Linear projection
        emissions = self.linear(lstm_output)
        return emissions
    

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        emissions = self._get_lstm_features(input_ids, attention_mask, token_type_ids)
        
        if labels is not None:
            # Training mode
            loss = -self.crf(emissions, labels, mask=attention_mask, reduction='mean')
            predictions = self.crf.decode(emissions, mask=attention_mask)
            return {
                'loss': loss,
                'predictions': predictions,
                'emissions': emissions
            }
        else:
            # Inference mode
            predictions = self.crf.decode(emissions, mask=attention_mask)
            return {
                'predictions': predictions,
                'emissions': emissions
            }
    
    
    def decode(
        self,
        emissions: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> List[List[int]]:
        """Decode the best tag sequence using Viterbi algorithm"""
        return self.crf.decode(emissions, mask=mask)


