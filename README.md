# Pytorch BiLSTM_CRF Medical Named Entity Recognition (NER)

This project focuses on medical named entity recognition (NER) using combinations of **BERT**, **BiLSTM**, and **CRF** models. The task involves identifying medical entities from text based on annotated data provided by the **Tianchi competition** hosted by Alibaba.

## Overview

Named Entity Recognition (NER) is a foundational task in Natural Language Processing (NLP). Key concepts include:
- **BIO tagging**: A scheme where tokens are tagged as Beginning (`B-`), Inside (`I-`), or Outside (`O`) of an entity.
- **BERT**: A pre-trained transformer model providing contextualized embeddings.
- **BiLSTM**: A bidirectional recurrent neural network for capturing sequential data patterns.
- **CRF**: Conditional Random Field for modeling dependencies between output labels.


---

## Directory Structure

```
├── README.md
├── input
├── output
│ ├── model
│ │ ├── bbc
│ │ ├── bert
│ │ └── bilstm
│ ├── label.txt
│ ├── test_sample.txt
│ ├── train_sample.txt
│ └── vocab.txt
├── models
│ ├── bbc_model.py
│ ├── bert_model.py
│ └── bilstm_model.py
├── training
│ ├── train_bbc.py
│ ├── train_bert.py
│ └── train_bilstm.py
├── eval
│ ├── evaluate_bbc.py
│ ├── evaluate_bert.py
│ └── evaluate_bilstm.py
├── predict.py
├── data_process.py
├── utils.py
├── config.py
└── environment.yml
```


### Directory Details

#### `input/`
Contains input datasets for training and prediction.

#### `output/`
Stores processed files and model outputs:
- **`train_sample.txt`** and **`test_sample.txt`**: Pre-processed samples for training and testing.
- **`vocab.txt`**: Character vocabulary and corresponding indices.
- **`label.txt`**: Entity type dictionary for NER.
- **`model/`**: Subdirectories for saved model checkpoints (e.g., `bbc/`, `bert/`, `bilstm/`).

#### `models/`
Contains model architectures:
- **`bert_model.py`**: BERT-based NER model definition.
- **`bilstm_model.py`**: BiLSTM-CRF based NER model definition.
- **`bbc_model.py`**: Combined BERT + BiLSTM + CRF model definition.

#### `training/`
Training scripts for each model:
- **`train_bbc.py`**: Trains the BERT-BiLSTM-CRF model.
- **`train_bert.py`**: Trains the BERT-based NER model.
- **`train_bilstm.py`**: Trains the BiLSTM-CRF based NER model.

#### `eval/`
Evaluation scripts:
- **`evaluate_bbc.py`**: Evaluates the BERT-BiLSTM-CRF model.
- **`evaluate_bert.py`**: Evaluates the BERT-based NER model.
- **`evaluate_bilstm.py`**: Evaluates the BiLSTM-CRF based NER model.

---

## Scripts Overview

- **`config.py`**: Configures file paths, training parameters, and constants.
- **`data_process.py`**: Pre-processes raw data into training and testing samples, and generates `vocab.txt` and `label.txt`.
- **`utils.py`**: Provides utilities for loading vocabularies, initializing models, and tokenization.
- **`predict.py`**: Uses a trained model to predict entities from user-provided text.

---

## Execution Workflow

1. **`config.py`**  
   Set file paths and training parameters.

2. **`data_process.py`**  
   Preprocess data to produce `train_sample.txt`, `test_sample.txt`, `vocab.txt`, and `label.txt` in `output/`.

3. **`utils.py`**  
   Load vocab, initialize tokenizers, and set up models.

4. **`models/`**  
   Defines architectures for BERT, BiLSTM, and BERT+BiLSTM+CRF.

5. **`training/train_*.py`**  
   Train the chosen model (BERT, BiLSTM, or BBC). Checkpoints saved under `output/model/`.

6. **`eval/evaluate_*.py`**  
   Evaluate trained models to get metrics like precision, recall, and F1 score.

7. **`predict.py`**  
   Predict entities in new text using a trained model.

---

## Additional Notes

- Models can be trained locally or on platforms like Kaggle.
- Checkpoints are saved for resuming or comparing models.
- NER metrics (precision, recall, F1) help measure model performance.

This project demonstrates a complete end-to-end NER pipeline from data preprocessing to model training, evaluation, and prediction on medical texts.
