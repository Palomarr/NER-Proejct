import torch
from models.bbc_model import Model
from utils import get_label
from config import *
from transformers import BertTokenizerFast
import argparse
import os

def extract_entities(labels, text):
    entities = []
    current_entity = []
    current_label = None

    for token, label in zip(text, labels):
        if label.startswith('B-'):
            if current_entity:
                entities.append((current_label, ''.join(current_entity)))
                current_entity = []
            current_label = label[2:]
            current_entity.append(token)
        elif label.startswith('I-') and current_label == label[2:]:
            current_entity.append(token)
        else:
            if current_entity:
                entities.append((current_label, ''.join(current_entity)))
                current_entity = []
            current_label = None

    if current_entity:
        entities.append((current_label, ''.join(current_entity)))

    return entities

def main():
    parser = argparse.ArgumentParser(description="NER Prediction Script")
    parser.add_argument('--text', type=str, required=True, help="Input text for NER prediction")
    parser.add_argument('--model_epoch', type=int, default=50, help="Model epoch to load")
    args = parser.parse_args()

    text = args.text
    model_epoch = args.model_epoch

    # Load label information
    label_list, label2id, id2label = get_label()
    num_labels = len(label_list)

    # Initialize tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL)

    # Prepare inputs
    encoding = tokenizer.encode_plus(
        text=list(text),
        add_special_tokens=True,
        return_tensors='pt',
        is_split_into_words=True
    )
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)
    attention_mask = attention_mask.bool()

    # Instantiate and load the model
    model = Model(num_labels=num_labels).to(DEVICE)
    model_path = f"{MODEL_DIR}/model_epoch_{model_epoch}.pth"
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        return
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    # Make predictions
    with torch.no_grad():
        y_pred = model(input_ids, attention_mask)  

    # Handle the output correctly
    label_ids = y_pred[0]
    labels = [id2label[label_id] for label_id in label_ids]

    print("Input Text:")
    print(text)
    print("\nPredicted Labels:")
    print(labels)

    # Extract entities
    entities = extract_entities(labels, text)
    print("\nExtracted Entities:")
    for entity in entities:
        print(f"Entity: {entity[1]}, Type: {entity[0]}")

if __name__ == '__main__':
    main()
