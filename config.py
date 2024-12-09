ORIGIN_DIR = './BBC/input/origin/'
ANNOTATION_DIR = './BBC/output/annotation/'

TRAIN_SAMPLE_PATH = './BBC/output/train_sample.txt'
VAL_SAMPLE_PATH = './BBC/output/val_sample.txt'
TEST_SAMPLE_PATH = './BBC/output/test_sample.txt'

VOCAB_PATH = './BBC/output/vocab.txt'
LABEL_PATH = './BBC/output/label.txt'

WORD_PAD = '<PAD>'
WORD_UNK = '<UNK>'

WORD_PAD_ID = 0
WORD_UNK_ID = 1
LABEL_O_ID = 0

VOCAB_SIZE = 3000
HIDDEN_SIZE = 256

BATCH_SIZE = 100
LR = 5e-5
EPOCH = 50

GRAD_CLIP=1.0

MODEL_DIR = './BBC/output/model/'
DATA_DIR = './BBC/output/'

import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# bert
BERT_MODEL = "bert-base-uncased"
MAX_POSITION_EMBEDDINGS = 512

