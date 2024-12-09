import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

from BBC.models.bilstm_model import ClassicBiLSTMNER
from BBC.utils import Dataset, collate_fn, get_label, get_vocab
from BBC.config import *

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler('bilstm_training.log'),
            logging.StreamHandler()
        ]
    )

def save_checkpoint(model, optimizer, epoch, best_val_f1, vocab_size):
    """Save model checkpoint with training state"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_f1': best_val_f1,
        'vocab_size': vocab_size
    }
    os.makedirs(f"{MODEL_DIR}/bilstm", exist_ok=True)
    torch.save(checkpoint, f"{MODEL_DIR}/bilstm/checkpoint_epoch_{epoch}.pth")
    logging.info(f"Checkpoint saved for epoch {epoch}")

def evaluate_bilstm(model, val_loader, device):
    """Evaluate BiLSTM model with input validation"""
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets, masks) in enumerate(val_loader):
            inputs = torch.clamp(inputs, min=0, max=model.embedding.num_embeddings-1)
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            
            loss = model(inputs, masks, labels=targets)
            total_loss += loss.item()
            
            # Get predictions and handle them as tensors directly
            batch_preds = model(inputs, masks)
         
            # Apply masking consistently to both predictions and targets
            for pred, target, mask in zip(batch_preds, targets.cpu().numpy(), masks.cpu().numpy()):
                mask_indices = np.where(mask == 1)[0]  # Changed: ensure boolean mask is converted to indices
                pred_numpy = pred.cpu().numpy() if torch.is_tensor(pred) else np.array(pred)  # Changed: ensure pred is numpy array
                predictions.extend(pred_numpy[mask_indices])  # Changed: removed tolist() as it's redundant
                true_labels.extend(target[mask_indices])  # Changed: removed tolist() as it's redundant
    
    # Calculate metrics
    metrics = {
        'loss': total_loss / len(val_loader),
        'accuracy': accuracy_score(true_labels, predictions),
        'precision': precision_score(true_labels, predictions, average='weighted'),
        'recall': recall_score(true_labels, predictions, average='weighted'),
        'f1': f1_score(true_labels, predictions, average='weighted')
    }
    
    return metrics

def train_classic_bilstm():
    """Train a classic BiLSTM-CRF based NER model with best model saving and early stopping."""
    # Setup logging
    setup_logging()
    
    try:
        # Load label information
        label_list, label2id, id2label = get_label()
        num_labels = len(label_list)
        
        # Get vocab information
        vocab, word2id = get_vocab()
        vocab_size = len(vocab)
        logging.info(f"Vocabulary size: {vocab_size}")
        
        # Initialize datasets and dataloaders
        train_dataset = Dataset(type='train')
        val_dataset = Dataset(type='val')
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                                shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                              shuffle=False, collate_fn=collate_fn)
        
        # Initialize model
        model = ClassicBiLSTMNER(
            num_labels=num_labels, 
            vocab_size=vocab_size
        ).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        
        # Training loop with early stopping
        best_val_f1 = 0.0
        patience = 5
        patience_counter = 0
        
        for epoch in range(EPOCH):
            model.train()
            total_loss = 0
            
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCH}") as progress_bar:
                for batch_idx, (inputs, targets, masks) in enumerate(progress_bar):
                    # Clamp input values to valid vocab indices
                    inputs = torch.clamp(inputs, min=0, max=vocab_size-1)
                    
                    inputs = inputs.to(DEVICE)
                    targets = targets.to(DEVICE)
                    masks = masks.to(DEVICE)
                    
                    loss = model(inputs, masks, labels=targets)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    progress_bar.set_postfix(loss=loss.item())
                
                # Evaluate only every 10 batches and at the last batch
                if (batch_idx + 1) % 10 == 0 or batch_idx == len(train_loader) - 1:
                    avg_loss = total_loss / (batch_idx + 1)
                    logging.info(f"Epoch {epoch+1}/{EPOCH}, Batch {batch_idx+1}, Loss: {avg_loss:.4f}")
                    
                    # Evaluate on validation set
                    model.eval()
                    with torch.no_grad():
                        val_metrics = evaluate_bilstm(model, val_loader, DEVICE)
                    val_f1 = val_metrics['f1']
                    
                    logging.info(f"Validation metrics at epoch {epoch+1}, batch {batch_idx+1}:")
                    for metric, value in val_metrics.items():
                        logging.info(f"{metric}: {value:.4f}")
                    
                    # Save best model
                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        patience_counter = 0
                        # Only save checkpoint every 10 epochs
                        if (epoch + 1) % 10 == 0:
                            save_checkpoint(model, optimizer, epoch, best_val_f1, vocab_size)
                        # Always save the best model regardless of epoch
                        torch.save(model.state_dict(), f"{MODEL_DIR}/bilstm/best_model.pth")
                        logging.info(f"New best model saved with F1: {val_f1:.4f}")
                    else:
                        patience_counter += 1
                    
                    # Early stopping check
                    if patience_counter >= patience:
                        logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                        break
                    
                    model.train()  # Switch back to training mode
            
        
    except Exception as e:
        logging.error(f"An error occurred during training: {str(e)}")
        raise

if __name__ == '__main__':
    train_classic_bilstm()
