import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data
from config import *
import pandas as pd
from seqeval.metrics import classification_report

from transformers import BertTokenizerFast, logging
 

logging.set_verbosity_warning()

tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL)

# Ensure the pad token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = '[PAD]'

WORD_PAD_ID = tokenizer.pad_token_id


def get_vocab():
    df = pd.read_csv(VOCAB_PATH, names=['word', 'id'])
    return list(df['word']), dict(df.values)


def get_label():
    df = pd.read_csv(LABEL_PATH, names=['label', 'id'])
    label_list = list(df['label'])
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for idx, label in enumerate(label_list)}
    return label_list, label2id, id2label


_, label2id, _ = get_label()
LABEL_O_ID = label2id['O']


class Dataset(data.Dataset):
    def __init__(self, type='train'):
        super().__init__()
        sample_path = TRAIN_SAMPLE_PATH if type == 'train' else TEST_SAMPLE_PATH
        self.samples = self.load_samples(sample_path)
        self.tokenizer = tokenizer
        _, self.label2id, _ = get_label()


    def __getitem__(self, index):
        words, labels = self.samples[index]
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_POSITION_EMBEDDINGS,
            return_special_tokens_mask=True,
        )
        input_ids = torch.tensor(encoding['input_ids'], dtype=torch.long)
        word_ids = encoding.word_ids()

        label_ids = []
        mask = []

        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                # Special tokens ([CLS], [SEP], [PAD])
                label_ids.append(LABEL_O_ID)  # Assign 'O' label
                mask.append(0)  # Mask out
            else:
                label = labels[word_idx]
                label_id = self.label2id.get(label, LABEL_O_ID)
                label_ids.append(label_id)
                mask.append(1)  # Valid token

        label_ids = torch.tensor(label_ids, dtype=torch.long)
        mask = torch.tensor(mask, dtype=torch.bool)
        # Ensure the first timestep of the mask is always on (1)
        mask[0] = 1

        return input_ids, label_ids, mask

    
    def load_samples(self, sample_path):
        """Load samples from the dataset file."""
        samples = []
        with open(sample_path, 'r', encoding='utf-8') as f:
            words = []
            labels = []
            for line in f:
                if line.strip() == '':
                    if words:  # End of a sample
                        samples.append((words, labels))
                        words = []
                        labels = []
                else:
                    word, label = line.strip().split('\t')
                    words.append(word)
                    labels.append(label)
        if words:  # Handle the last sample
            samples.append((words, labels))
        return samples


    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)



def collate_fn(batch):
    inputs, labels, masks = zip(*batch)
    inputs = pad_sequence(inputs, batch_first=True, padding_value=WORD_PAD_ID)
    labels = pad_sequence(labels, batch_first=True, padding_value=LABEL_O_ID)
    masks = pad_sequence(masks, batch_first=True, padding_value=0)

    masks[:, 0] = 1

    return inputs, labels, masks



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
    return classification_report(y_true, y_pred)
    
