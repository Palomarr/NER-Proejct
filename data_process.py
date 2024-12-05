import os
import pandas as pd
from datasets import load_dataset
from config import *


# Load the dataset
dataset = load_dataset("DFKI-SLT/cross_ner", "conll2003", cache_dir="./input/cross")
print(dataset)

# Get label names
ner_feature = dataset['train'].features['ner_tags']
label_names = ner_feature.feature.names
id_to_label = {i: label for i, label in enumerate(label_names)}

# Clear existing sample files
open(TRAIN_SAMPLE_PATH, 'w', encoding='utf-8').close()
open(TEST_SAMPLE_PATH, 'w', encoding='utf-8').close()

# Function to process and save a split
def process_and_save_split(dataset_split, output_file):
    with open(output_file, 'a', encoding='utf-8') as f:
        for example in dataset_split:
            tokens = example['tokens']
            ner_tags = example['ner_tags']
            for token, tag_id in zip(tokens, ner_tags):
                label = id_to_label[tag_id]
                f.write(f"{token}\t{label}\n")
            f.write('\n')  # Empty line to separate sentences

# Process train and validation splits and save to train_sample.txt
process_and_save_split(dataset['train'], TRAIN_SAMPLE_PATH)
process_and_save_split(dataset['validation'], TRAIN_SAMPLE_PATH)

# Process test split and save to test_sample.txt
process_and_save_split(dataset['test'], TEST_SAMPLE_PATH)

# Generate vocab
def generate_vocab():
    vocab_set = set()
    with open(TRAIN_SAMPLE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == '':
                continue
            token, _ = line.strip().split('\t')
            vocab_set.add(token)
    vocab_list = ['[PAD]', '[UNK]'] + sorted(vocab_set)
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab_df = pd.DataFrame(list(vocab_dict.items()))
    vocab_df.to_csv(VOCAB_PATH, header=None, index=None)

# Generate label
def generate_label():
    label_list = label_names
    label_dict = {v: k for k, v in enumerate(label_list)}
    label_df = pd.DataFrame(list(label_dict.items()))
    label_df.to_csv(LABEL_PATH, header=None, index=None)

# Generate vocab and label files
generate_vocab()
generate_label()
