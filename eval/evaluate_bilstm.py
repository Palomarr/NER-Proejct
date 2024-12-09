import os
import torch
from torch.utils.data import DataLoader
from seqeval.metrics import classification_report, accuracy_score

from BBC.models.bilstm_model import ClassicBiLSTMNER
from BBC.utils import Dataset, collate_fn, get_label, get_vocab
from BBC.config import *


def evaluate_classic_bilstm(model_path=None, model_epoch=None):
    """
    Evaluate the Classic BiLSTM-CRF NER model on the test set.
    
    Args:
        model_path (str): Path to the model file. If None, uses the best model.
        model_epoch (int): Epoch number of the model to evaluate. If None, uses the best model.
    """
    if model_path is None and model_epoch is not None:
        model_path = f"{MODEL_DIR}/bilstm/model_epoch_{model_epoch}.pth"
    elif model_path is None:
        model_path = f"{MODEL_DIR}/bilstm/best_model.pth"
    
    # Load label information
    label_list, label2id, id2label = get_label()
    num_labels = len(label_list)
    
    # Load the vocabulary from the saved model's epoch
    vocab, word2id = get_vocab()
    vocab_size = len(vocab)
    
    print(f"Vocabulary size: {vocab_size}")
    
    # Load model vocabulary size from the saved state
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        return
    
    saved_state = torch.load(model_path, map_location=DEVICE)
    saved_vocab_size = saved_state['embedding.weight'].shape[0]
    
    # Initialize the model with saved vocabulary size
    model = ClassicBiLSTMNER(num_labels=num_labels, vocab_size=saved_vocab_size).to(DEVICE)
    
    # Load model state
    model.load_state_dict(saved_state)
    model.eval()
    print(f"Loaded Classic BiLSTM model from {model_path}")
    
    # Initialize test dataset and dataloader
    test_dataset = Dataset(type='test')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    print(f"Number of Test Samples: {len(test_dataset)}")
    
    y_true_list = []
    y_pred_list = []
    
    with torch.no_grad():
        for batch_idx, (input_ids, labels, attention_mask) in enumerate(test_loader):
            # Clamp input_ids to valid vocabulary indices
            input_ids = torch.clamp(input_ids, min=0, max=saved_vocab_size-1)
            
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Get predictions (list of lists)
            predictions = model(input_ids, attention_mask)
            
            # Convert to label names
            for i, pred_labels in enumerate(predictions):
                true_labels = labels[i].cpu().tolist()
                mask = attention_mask[i].cpu().tolist()
                
                pred_labels_filtered = [id2label[p] for p, m in zip(pred_labels, mask) if m]
                true_labels_filtered = [id2label[t] for t, m in zip(true_labels, mask) if m]
                
                y_pred_list.append(pred_labels_filtered)
                y_true_list.append(true_labels_filtered)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1} batches.")
    
    # Compute metrics
    accuracy = accuracy_score(y_true_list, y_pred_list)
    report = classification_report(y_true_list, y_pred_list, digits=4)
    
    print("\n=== Evaluation Metrics for Classic BiLSTM Model ===\n")
    print(f"Accuracy: {accuracy:.4f}\n")
    print("Classification Report:")
    print(report)
    
    # Save evaluation report
    report_path = f"{MODEL_DIR}/bilstm/evaluation_report_epoch_{model_epoch}.txt"
    with open(report_path, "w") as f:
        f.write("=== Evaluation Metrics for Classic BiLSTM Model ===\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    print(f"Saved evaluation report to {report_path}")


if __name__ == '__main__':
    evaluate_classic_bilstm()
