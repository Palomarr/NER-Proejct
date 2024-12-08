import torch
from torch.utils.data import DataLoader
from seqeval.metrics import classification_report, accuracy_score
from models.bilstm_model import ClassicBiLSTMNER
from utils import Dataset, collate_fn, get_label, get_vocab
from config import *
import os

def evaluate_classic_bilstm(model_epoch=50):
    # Load label information
    label_list, label2id, id2label = get_label()
    num_labels = len(label_list)
    
    # Initialize model
    vocab_size = len(get_vocab()[1])  # Assuming get_vocab() returns (id2word, word2id)
    model = ClassicBiLSTMNER(num_labels=num_labels, vocab_size=vocab_size).to(DEVICE)
    model_path = f"{MODEL_DIR}/classic_bilstm/model_epoch_{model_epoch}.pth"
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
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
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Get predictions
            predictions = model(input_ids, attention_mask)  # List of lists
            
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
    
    # Save report
    report_path = f"{MODEL_DIR}/classic_bilstm/evaluation_report_epoch_{model_epoch}.txt"
    with open(report_path, "w") as f:
        f.write("=== Evaluation Metrics for Classic BiLSTM Model ===\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"Saved evaluation report to {report_path}")

if __name__ == '__main__':
    evaluate_classic_bilstm(model_epoch=50)
