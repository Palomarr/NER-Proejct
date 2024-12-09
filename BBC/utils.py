import os
from BBC.config import DATA_DIR

def get_vocab():
    """
    Load vocabulary and word-to-id mapping from the vocab file.
    
    Returns:
        tuple: (vocab_list, word2id_dict) where
            - vocab_list is a list of words in the vocabulary
            - word2id_dict is a dictionary mapping words to their indices
    """
    vocab_path = os.path.join(DATA_DIR, 'vocab.txt')
    
    try:
        vocab = ['[PAD]']  # Add padding token as first token (index 0)
        word2id = {'[PAD]': 0}
        
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, start=1):  # Start from 1 since PAD is 0
                word = line.strip()
                if word:  # Skip empty lines
                    vocab.append(word)
                    word2id[word] = i
        
        return vocab, word2id
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}")
    except Exception as e:
        raise Exception(f"Error loading vocabulary: {str(e)}")