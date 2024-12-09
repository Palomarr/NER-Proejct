import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import logging
import numpy as np
from seqeval.metrics import classification_report, f1_score
from BBC.utils import *
from BBC.models.bert_model import ClassicBERTNER
from BBC.config import *

def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    # Get label mappings
    _, _, id2label = get_label()
    
    try:
        with torch.no_grad():
            for inputs, targets, masks in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                masks = masks.to(device)
                
                # Get model predictions
                outputs = model(inputs, masks)  # Get logits from model
                
                # Handle both training mode (returns loss) and eval mode (returns logits)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                # Get predictions for each token
                predictions = torch.argmax(outputs, dim=-1)  # [batch_size, seq_len]
                
                # Convert to numpy and mask out padding
                batch_predictions = predictions.cpu().numpy()
                batch_targets = targets.cpu().numpy()
                batch_masks = masks.cpu().numpy()
                
                # Process each sequence in the batch
                for pred, target, mask in zip(batch_predictions, batch_targets, batch_masks):
                    valid_indices = mask.astype(bool)
                    valid_pred = [id2label[p] for p in pred[valid_indices]]
                    valid_target = [id2label[t] for t in target[valid_indices]]
                    
                    all_predictions.append(valid_pred)
                    all_labels.append(valid_target)
        
        if len(all_predictions) == 0:
            logging.warning("No valid predictions found!")
            return {
                'accuracy': 0.0
            }
        
        # Calculate metrics using string labels
        metrics = {
            'accuracy': accuracy_score(all_labels, all_predictions)
        }
        
        return metrics
        
    except Exception as e:
        logging.error(f"Error in evaluate function: {str(e)}")
        logging.error(f"Shapes at error:")
        logging.error(f"inputs: {inputs.shape}")
        logging.error(f"targets: {targets.shape}")
        logging.error(f"masks: {masks.shape}")
        logging.error(f"outputs: {outputs.shape}")
        logging.error(f"predictions: {predictions.shape}")
        logging.error(f"Sample prediction: {all_predictions[:1] if all_predictions else 'empty'}")
        logging.error(f"Sample label: {all_labels[:1] if all_labels else 'empty'}")
        raise e

def train_epoch(model, train_loader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCH}')
    for batch in progress_bar:
        input_ids, labels, attention_mask = [b.to(device) for b in batch]
        
        # Forward pass
        loss = model(input_ids, attention_mask, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })
    
    return total_loss / len(train_loader)

def evaluate_model(model_path, data_type='test', batch_size=BATCH_SIZE):
    """
    Evaluate a trained BERT model on a specified dataset.
    
    Args:
        model_path (str): Path to the saved model state dict
        data_type (str): Type of dataset to evaluate on ('train', 'val', or 'test')
        batch_size (int): Batch size for evaluation
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load label information
        label_list, _, _ = get_label()
        num_labels = len(label_list)
        
        # Initialize model and load weights
        model = ClassicBERTNER(num_labels=num_labels).to(device)
        model.load_state_dict(torch.load(model_path))
        
        # Prepare dataset and dataloader
        eval_dataset = Dataset(type=data_type)
        eval_loader = DataLoader(
            eval_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=collate_fn
        )
        
        # Evaluate
        metrics = evaluate(model, eval_loader, device)
        
        # Log results
        logging.info(f"\nEvaluation results on {data_type} dataset:")
        for metric, value in metrics.items():
            logging.info(f"{metric}: {value:.4f}")
            
        return metrics
        
    except Exception as e:
        logging.error(f"Evaluation failed: {str(e)}")
        raise e

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data loaders
    train_dataset = Dataset(type='train')
    val_dataset = Dataset(type='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Model
    num_labels = len(get_label()[0])
    model = ClassicBERTNER(num_labels=num_labels).to(device)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    num_training_steps = len(train_loader) * EPOCH
    num_warmup_steps = num_training_steps // 10
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Training loop
    best_accuracy = 0
    patience = 5
    patience_counter = 0
    
    for epoch in range(EPOCH):
        # Train
        avg_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        
        # Quick evaluation for model saving every epoch
        val_metrics = evaluate(model, val_loader, device)
        current_accuracy = val_metrics['accuracy']
        
        # Save best model
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), f'{MODEL_DIR}/bert/best_model.pth')
            logging.info(f"New best model saved with accuracy: {best_accuracy:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f'Early stopping triggered after {epoch + 1} epochs')
                break
        
        # Detailed logging every 10 epochs or on final epoch
        if (epoch + 1) % 10 == 0 or epoch == EPOCH - 1:
            logging.info(f'Epoch [{epoch+1}/{EPOCH}] - Average Loss: {avg_loss:.4f}')
            logging.info(f'Validation metrics: {val_metrics}')

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise