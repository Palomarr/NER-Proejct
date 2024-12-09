import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data
import pandas as pd
from typing import Dict, List, Tuple
from seqeval.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from transformers import BertTokenizerFast
import logging as python_logging
from transformers import logging as transformers_logging
import os
from config import *

# Configure transformers logging
transformers_logging.set_verbosity_warning()

# Configure Python logging
python_logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=python_logging.INFO
)

# Initialize the tokenizer with English BERT
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Ensure the pad token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = '[PAD]'

WORD_PAD_ID = tokenizer.pad_token_id

def evaluate(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device) -> Dict[str, float]:
    """Evaluate the model and return metrics"""
    model.eval()
    _, _, id2label = get_label()
    
    total_loss = 0
    all_predictions = []
    all_labels = []
    all_masks = []
    
    with torch.no_grad():
        for input_ids, labels, attention_mask in data_loader:
            # Move data to device
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)
            
            # Get model outputs
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            predictions = outputs['predictions']
            
            total_loss += loss.item()
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())
            all_masks.extend(attention_mask.cpu().numpy())
    
    # Convert predictions to labels
    true_labels, pred_labels = convert_predictions_to_labels(
        all_predictions, all_labels, all_masks, id2label
    )
    
    # Calculate metrics with zero_division parameter to handle undefined cases
    try:
        f1 = f1_score(true_labels, pred_labels, zero_division=0)
        report = classification_report(true_labels, pred_labels, zero_division=0)
    except Exception as e:
        python_logging.warning(f"Error calculating metrics: {str(e)}")
        f1 = 0.0
        report = "Could not generate classification report"
    
    # Calculate metrics
    metrics = {
        'loss': total_loss / len(data_loader),
        'f1': f1
    }
    
    # Print detailed classification report
    python_logging.info(f"\nClassification Report:\n{report}")
    
    return metrics

def convert_predictions_to_labels(
    predictions: List[List[int]], 
    labels: List[List[int]], 
    mask: List[List[bool]], 
    id2label: Dict[int, str]
) -> Tuple[List[List[str]], List[List[str]]]:
    """Convert model predictions to label sequences"""
    y_pred = []
    y_true = []
    
    for pred_seq, label_seq, mask_seq in zip(predictions, labels, mask):
        pred_labels = [id2label[p] for p, m in zip(pred_seq, mask_seq) if m]
        true_labels = [id2label[l.item()] for l, m in zip(label_seq, mask_seq) if m]
        
        # Only add sequences that are not empty
        if pred_labels and true_labels:
            y_pred.append(pred_labels)
            y_true.append(true_labels)
    
    return y_true, y_pred

def get_vocab():
    """Load vocabulary and word-to-id mapping from the vocab file."""
    vocab_path = VOCAB_PATH
    
    try:
        python_logging.info(f"Loading vocabulary from {vocab_path}")
        
        # Read vocabulary first without adding PAD token
        vocab = []
        word2id = {}
        
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                word = line.strip()
                if word:
                    vocab.append(word)
                    word2id[word] = i
        
        # Check if [PAD] is already in vocabulary
        if '[PAD]' not in word2id:
            vocab = ['[PAD]'] + vocab
            # Rebuild word2id with updated indices
            word2id = {word: idx for idx, word in enumerate(vocab)}
        
        python_logging.info(f"Loaded vocabulary with size: {len(vocab)}")
        
        return vocab, word2id
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}")
    except Exception as e:
        raise Exception(f"Error loading vocabulary: {str(e)}")

class Dataset(data.Dataset):
    def __init__(self, type='train'):
        super().__init__()
        file_paths = {
            'train': TRAIN_SAMPLE_PATH,
            'val': VAL_SAMPLE_PATH,
            'test': TEST_SAMPLE_PATH
        }
        if type not in file_paths:
            raise ValueError(f"Invalid dataset type. Must be one of {list(file_paths.keys())}")
            
        sample_path = file_paths[type]
        self.samples = self.load_samples(sample_path)
        self.tokenizer = tokenizer
        _, self.label2id, _ = get_label()

    def __getitem__(self, index):
        words, labels = self.samples[index]
        
        # Tokenize words and get word IDs mapping
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            padding=False,
            truncation=True,
            max_length=MAX_POSITION_EMBEDDINGS,
            return_tensors=None,
        )

        # Get word IDs to align labels with tokens
        word_ids = encoding.word_ids()
        
        input_ids = torch.tensor(encoding['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(encoding['attention_mask'], dtype=torch.bool)

        # Create label_ids with proper handling of subwords
        label_ids = []
        prev_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens ([CLS], [SEP], [PAD])
                label_ids.append(self.label2id['O'])
            elif word_idx != prev_word_idx:
                # First token of the word
                label_ids.append(self.label2id.get(labels[word_idx], self.label2id['O']))
            else:
                # Continuation of the same word
                label_ids.append(label_ids[-1])
            prev_word_idx = word_idx

        label_ids = torch.tensor(label_ids, dtype=torch.long)

        # Debug information
        assert len(input_ids) == len(label_ids) == len(attention_mask), \
            f"Length mismatch: input_ids={len(input_ids)}, label_ids={len(label_ids)}, attention_mask={len(attention_mask)}"

        return input_ids, label_ids, attention_mask

    def load_samples(self, sample_path):
        """Load samples from the dataset file."""
        samples = []
        with open(sample_path, 'r', encoding='utf-8') as f:
            words = []
            labels = []
            for line in f:
                if line.strip() == '':
                    if words:
                        samples.append((words, labels))
                        words = []
                        labels = []
                else:
                    try:
                        word, label = line.strip().split('\t')
                        words.append(word)
                        labels.append(label)
                    except ValueError as e:
                        print(f"Warning: Skipping malformed line: {line.strip()}")
                        continue
        if words:
            samples.append((words, labels))
        return samples

    def __len__(self):
        return len(self.samples)

def collate_fn(batch):
    input_ids, label_ids, attention_masks = zip(*batch)
    
    # Pad sequences
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=WORD_PAD_ID)
    label_ids = pad_sequence(label_ids, batch_first=True, padding_value=LABEL_O_ID)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    
    # Debug information
    assert input_ids.size() == label_ids.size(), \
        f"Size mismatch after padding: input_ids={input_ids.size()}, label_ids={label_ids.size()}"
    
    return input_ids, label_ids, attention_masks

def get_label():
    df = pd.read_csv(LABEL_PATH, names=['label', 'id'])
    label_list = list(df['label'])
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for idx, label in enumerate(label_list)}
    return label_list, label2id, id2label

def extract(label, text):
    i = 0
    res = []
    while i < len(label):
        if label[i] != 'O':
            prefix, name = label[i].split('-')
            start = end = i
            i += 1
            while i < len(label) and label[i] == 'I-' + name:
                end = i
                i += 1
            res.append([name, text[start:end + 1]])
        else:
            i += 1
    return res

def report(y_true, y_pred):
    return classification_report(y_true, y_pred), accuracy_score(y_true, y_pred)