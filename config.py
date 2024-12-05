ORIGIN_DIR = './input/origin/'
ANNOTATION_DIR = './output/cross/annotation/'

TRAIN_SAMPLE_PATH = './output/cross/train_sample.txt'
TEST_SAMPLE_PATH = './output/cross/test_sample.txt'

VOCAB_PATH = './output/cross/vocab.txt'
LABEL_PATH = './output/cross/label.txt'

WORD_PAD = '<PAD>'
WORD_UNK = '<UNK>'

WORD_PAD_ID = 0
WORD_UNK_ID = 1
LABEL_O_ID = 0

VOCAB_SIZE = 3000
HIDDEN_SIZE = 256
BATCH_SIZE = 100
LR = 1e-4
EPOCH = 50
GRAD_CLIP=1.0

MODEL_DIR = './output/cross/model/'

import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# bert
BERT_MODEL = 'bert-base-cased'
EMBEDDING_DIM = 768
MAX_POSITION_EMBEDDINGS = 512

