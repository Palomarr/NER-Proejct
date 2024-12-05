# Pytorch BiLSTM_CRF Medical Named Entity Recognition (NER)

This project focuses on medical named entity recognition (NER) using a combination of **BERT**, **BiLSTM**, and **CRF** models. The task involves identifying medical entities from text based on annotated data provided by the **Tianchi competition** hosted by Alibaba.

## Overview

Named Entity Recognition (NER) is a foundational task in Natural Language Processing (NLP). Key concepts include:
- **BIO tagging**: A scheme where tokens are tagged as Beginning (`B-`), Inside (`I-`), or Outside (`O`) of an entity.
- **BERT**: A pre-trained transformer model providing contextualized embeddings.
- **BiLSTM**: A bidirectional recurrent neural network for capturing sequential data patterns.
- **CRF**: Conditional Random Field for modeling dependencies between output labels.

The dataset comes from the **Tianchi competition**: [MMC Knowledge Graph Competition by Ruijin Hospital](https://tianchi.aliyun.com/competition/entrance/231687/information).

---

## Directory Structure




├── README.md
├── input
├── output
├── config.py
├── data_process.py
├── environment.yml
├── model.py
├── predict.py
├── test.py
├── train.py
└── utils.py


### Directory Details

#### `input/`
Contains input datasets for training and prediction:
- **`origin/`**: Original files, including:
  - `xx.txt`: Raw text files.
  - `xx.ann`: Annotation files containing positions, entity types, and details of positive samples.

#### `output/`
Stores processed files and outputs:
- **`annotation/`**: Annotated `.txt` files with marked entities.
- **`train_sample.txt`**: Pre-processed training set for model training.
- **`test_sample.txt`**: Pre-processed test set for evaluation (e.g., precision, recall, F1 score).
- **`label.txt`**: Dictionary of entity types for NER.
- **`vocab.txt`**: Character vocabulary with corresponding indices.

---

## Scripts

- **`config.py`**: Configures file paths, training parameters, and other constants.
- **`data_process.py`**: Pre-processes data to generate output files for training and evaluation.
- **`utils.py`**: Provides utility functions for loading vocabularies, initializing BERT, etc.
- **`model.py`**: Implements the BERT + BiLSTM + CRF model.
- **`train.py`**: Trains the model. Can run locally or on Kaggle.
- **`test.py`**: Evaluates the trained model with metrics like precision, recall, and F1 score.
- **`predict.py`**: Predicts entities and labels from user-provided text.

---

## Execution Workflow

1. **`config.py`**  
   Configure file paths and training parameters.

2. **`data_process.py`**  
   Preprocess data:
   - Generate files such as `train_sample.txt`, `test_sample.txt`, `label.txt`, and `vocab.txt` in the `output/` directory.

3. **`utils.py`**  
   Load vocabularies, prepare tokenizers, and initialize the pre-trained BERT model.

4. **`model.py`**  
   Define the model architecture using BERT, BiLSTM, and CRF components.

5. **`train.py`**  
   Train the model:
   - Supports local or cloud training (e.g., Kaggle).
   - Includes checkpointing for interrupted training.

6. **`test.py`**  
   Evaluate the model on the test set:
   - Generates performance metrics such as precision, recall, and F1 score.

7. **`predict.py`**  
   Use the trained model for predictions:
   - Input custom text and identify entities with their labels.

---

## Notes on Training

- **Checkpointing**:  
   To save space and ensure training can resume efficiently, checkpointing is implemented.

---

This project demonstrates a complete implementation of an NER system for medical texts using advanced NLP techniques.
