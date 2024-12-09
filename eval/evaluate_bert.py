import torch
from torch.utils.data import DataLoader
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score
from seqeval.metrics import classification_report
import os

from BBC.models.bert_model import ClassicBERTNER
from BBC.utils import Dataset, collate_fn, get_label
from BBC.config import *
from BBC.training.train_bert import evaluate_model

def convert_ids_to_tags(predictions, labels, mask, id2label):
    """Convert predicted and true label ids to tag sequences"""
    y_pred = []
    y_true = []
    
    for pred_seq, label_seq, mask_seq in zip(predictions, labels, mask):
        # Handle predictions
        pred_tags = []
        true_tags = []
        for p, l, m in zip(pred_seq, label_seq, mask_seq):
            if m:  # Only process tokens that are not padding
                pred_tags.append(id2label[p.item()])  # Convert tensor to int
                true_tags.append(id2label[l.item()])  # Convert tensor to int
        
        y_pred.append(pred_tags)
        y_true.append(true_tags)
        
    return y_true, y_pred

def evaluate(model, data_loader):
    """Evaluate the model and return metrics"""
    model.eval()
    
    _, _, id2label = get_label()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for input_ids, labels, attention_mask in data_loader:
            input_ids = input_ids.to(DEVICE)
            labels = labels.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            
            # Get predictions
            outputs = model(input_ids, attention_mask)
            predictions = torch.argmax(outputs, dim=-1)  # [batch_size, seq_len]
            
            # Convert to label sequences
            true_tags, pred_tags = convert_ids_to_tags(
                predictions, labels, attention_mask, id2label
            )
            
            all_predictions.extend(pred_tags)
            all_labels.extend(true_tags)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_predictions),
        'precision': precision_score(all_labels, all_predictions),
        'recall': recall_score(all_labels, all_predictions),
        'f1': f1_score(all_labels, all_predictions)
    }
    
    return metrics, all_labels, all_predictions

def main(model_epoch=50):
    """Evaluate a trained model"""
    model_path = f"{MODEL_DIR}/bert/model_epoch_{model_epoch}.pth"
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        return
    
    # Use the imported evaluate_model function
    metrics = evaluate_model(
        model_path=model_path,
        data_type='test',
        batch_size=BATCH_SIZE
    )
    
    # Get detailed classification report
    label_list, label2id, id2label = get_label()
    num_labels = len(label_list)
    
    model = ClassicBERTNER(num_labels=num_labels).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    
    test_dataset = Dataset(type='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Get predictions for classification report
    _, all_labels, all_predictions = evaluate(model, test_loader)
    
    # Generate and save detailed report
    report = classification_report(all_labels, all_predictions)
    save_path = os.path.join(os.path.dirname(model_path), f'evaluation_results_epoch_{model_epoch}.txt')
    
    with open(save_path, 'w') as f:
        f.write("=== Evaluation Results ===\n")
        for metric_name, value in metrics.items():
            f.write(f"{metric_name}: {value:.4f}\n")
        f.write("\n=== Detailed Classification Report ===\n\n")
        f.write(report)
    
    # Print results to console
    print("\n=== Evaluation Results ===")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    print("\n=== Detailed Classification Report ===\n")
    print(report)

if __name__ == '__main__':
    main(model_epoch=50)
