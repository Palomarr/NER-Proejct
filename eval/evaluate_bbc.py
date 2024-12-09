import torch
from torch.utils.data import DataLoader
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score
from seqeval.metrics import classification_report
import logging
from typing import Dict, List
import os

from BBC.utils import Dataset, collate_fn, get_label
from BBC.models.bbc_model import ImprovedBertBiLSTMCRF
from BBC.config import *
from BBC.training.train_bbc import MODEL_DIR

def setup_logging():
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler('evaluation.log'),
            logging.StreamHandler()
        ]
    )

def convert_ids_to_tags(predictions: List[List[int]], 
                       labels: torch.Tensor, 
                       mask: torch.Tensor,
                       id2label: Dict[int, str]) -> tuple:
    """Convert predicted and true label ids to tag sequences"""
    y_pred = []
    y_true = []
    
    for pred_seq in predictions:
        y_pred.append([id2label[i] for i in pred_seq])
        
    for label_seq, mask_seq in zip(labels, mask):
        true_seq = [id2label[label_seq[i].item()] for i in range(len(mask_seq)) if mask_seq[i]]
        y_true.append(true_seq)
        
    return y_true, y_pred

def evaluate(model: torch.nn.Module, data_loader: DataLoader) -> Dict[str, float]:
    """Evaluate the model and return metrics"""
    model.eval()
    
    _, _, id2label = get_label()
    
    all_predictions = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch_idx, (input_ids, labels, attention_mask) in enumerate(data_loader):
            # Move data to device
            input_ids = input_ids.to(DEVICE)
            labels = labels.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            
            # Get model outputs
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            predictions = outputs['predictions']
            
            # Convert ids to tags
            true_tags, pred_tags = convert_ids_to_tags(
                predictions, labels, attention_mask, id2label
            )
            
            all_predictions.extend(pred_tags)
            all_labels.extend(true_tags)
            total_loss += loss.item()
    
    # Calculate metrics
    metrics = {
        'loss': total_loss / len(data_loader),
        'accuracy': accuracy_score(all_labels, all_predictions),
        'precision': precision_score(all_labels, all_predictions),
        'recall': recall_score(all_labels, all_predictions),
        'f1': f1_score(all_labels, all_predictions)
    }
    
    return metrics

def main():
    # Setup logging
    setup_logging()
    
    # Initialize the test dataset and data loader
    test_dataset = Dataset(type='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Get label mappings
    label_list, label2id, id2label = get_label()
    num_labels = len(label_list)

    # Initialize model with the same configuration as training
    model_config = {
        'num_labels': num_labels,
        'bert_model_name': 'bert-base-uncased',
        'hidden_size': 256,
        'num_lstm_layers': 2,
        'dropout': 0.5,
        'freeze_bert': False
    }
    
    model = ImprovedBertBiLSTMCRF(**model_config).to(DEVICE)
    
    # Load the best model
    try:
        state_dict = torch.load(f"{MODEL_DIR}/best_model.pth", map_location=DEVICE)
        model.load_state_dict(state_dict)
        logging.info("Loaded the best model successfully")
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return

    # Evaluate
    metrics = evaluate(model, test_loader)
    
    # Print results
    print("\n=== Evaluation Results ===")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    # Generate detailed classification report
    _, _, id2label = get_label()
    all_predictions = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for input_ids, labels, attention_mask in test_loader:
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = outputs['predictions']
            
            true_tags, pred_tags = convert_ids_to_tags(
                predictions, labels, attention_mask, id2label
            )
            
            all_predictions.extend(pred_tags)
            all_labels.extend(true_tags)
    
    # Get classification report
    report = classification_report(all_labels, all_predictions)
    print("\n=== Detailed Classification Report ===\n")
    print(report)

    # Save results to file
    save_path = os.path.join(os.path.dirname(f"{MODEL_DIR}/best_model.pth"), 'evaluation_results.txt')
    with open(save_path, 'w') as f:
        f.write("=== Evaluation Results ===\n")
        for metric_name, value in metrics.items():
            f.write(f"{metric_name}: {value:.4f}\n")
        f.write("\n=== Detailed Classification Report ===\n\n")
        f.write(report)

if __name__ == '__main__':
    main()