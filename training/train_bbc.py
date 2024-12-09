import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import logging
from typing import Dict, Any

from BBC.utils import Dataset, collate_fn, get_label, evaluate
from BBC.models.bbc_model import ImprovedBertBiLSTMCRF
from BBC.config import *

MODEL_DIR = './BBC/output/model/bbc/hyper/v2.0'
os.makedirs(MODEL_DIR, exist_ok=True)

def setup_logging():
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                   scheduler: Any, epoch: int, best_val_f1: float, config: Dict) -> None:
    """Save model checkpoint with all training state"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_f1': best_val_f1,
        'config': config
    }
    torch.save(checkpoint, f"{MODEL_DIR}/checkpoint_epoch_{epoch}.pth")
    logging.info(f"Checkpoint saved for epoch {epoch}")

def train_epoch(model: torch.nn.Module, train_loader: DataLoader, 
                optimizer: torch.optim.Optimizer, scheduler: Any, 
                epoch: int) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCH}") as progress_bar:
        for batch_idx, (input_ids, labels, attention_mask) in enumerate(progress_bar):
            # Move data to device
            input_ids = input_ids.to(DEVICE)
            labels = labels.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs['loss']

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            if scheduler:
                scheduler.step()

            # Update metrics
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}"
            )

    return total_loss / len(train_loader)

def main():
    # Setup logging
    setup_logging()
    
    try:
        # Initialize datasets
        train_dataset = Dataset(type='train')
        val_dataset = Dataset(type='val')
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
        )

        # Get label mappings
        label_list, label2id, id2label = get_label()
        num_labels = len(label_list)

        # Model configuration
        model_config = {
            'num_labels': num_labels,
            'bert_model_name': 'bert-base-uncased',
            'hidden_size': 256,
            'num_lstm_layers': 2,
            'dropout': 0.5,
            'freeze_bert': False
        }

        # Initialize model
        model = ImprovedBertBiLSTMCRF(**model_config).to(DEVICE)

        # Optimizer with different learning rates for BERT and other layers
        bert_params = list(model.bert.parameters())
        other_params = [p for n, p in model.named_parameters() if not n.startswith('bert.')]
        
        optimizer = AdamW([
            {'params': bert_params, 'lr': 1e-5},
            {'params': other_params, 'lr': 1e-3}
        ])

        # Learning rate scheduler with warmup
        num_training_steps = len(train_loader) * EPOCH
        num_warmup_steps = num_training_steps // 10
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        # Training loop
        best_val_f1 = 0.0
        patience = 5
        patience_counter = 0
        
        for epoch in range(EPOCH):
            avg_loss = train_epoch(model, train_loader, optimizer, scheduler, epoch)
            
            # Only evaluate and log every 10 epochs (and on the final epoch)
            if (epoch + 1) % 10 == 10 or epoch == EPOCH - 1:
                logging.info(f'Epoch [{epoch+1}/{EPOCH}] - Average Loss: {avg_loss:.4f}')
                
                # Validate
                val_metrics = evaluate(model, val_loader, DEVICE)
                val_f1 = val_metrics['f1']
                
                # Log all evaluation metrics
                logging.info(f"Evaluation metrics at epoch {epoch+1}:")
                for metric, value in val_metrics.items():
                    logging.info(f"{metric}: {value:.4f}")
                
                # Save checkpoint
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                    torch.save(model.state_dict(), f"{MODEL_DIR}/best_model.pth")
                    logging.info(f"New best model saved with F1: {val_f1:.4f}")
                    save_checkpoint(model, optimizer, scheduler, epoch, best_val_f1, model_config)
                else:
                    patience_counter += 1
            
            if patience_counter >= patience:
                logging.info(f'Early stopping triggered after {epoch + 1} epochs')
                break

    except Exception as e:
        logging.error(f"An error occurred during training: {str(e)}")
        raise

if __name__ == '__main__':
    main()
