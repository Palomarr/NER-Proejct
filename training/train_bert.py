import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from BBC.models.bert_model import ClassicBERTNER
from BBC.utils import Dataset, collate_fn, get_label
from BBC.config import *


def train_bert():
    """
    Train a BERT-based NER model using the dataset and parameters defined in config.
    """
    # Load label information
    label_list, label2id, id2label = get_label()
    num_labels = len(label_list)
    
    # Initialize dataset and dataloader for training
    train_dataset = Dataset(type='train')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    # Initialize model and optimizer
    model = ClassicBERTNER(num_labels=num_labels).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    # Training loop
    for epoch in range(EPOCH):
        model.train()
        total_loss = 0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCH}") as progress_bar:
            for batch_idx, (inputs, targets, masks) in enumerate(progress_bar):
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                masks = masks.to(DEVICE)

                loss = model(inputs, masks, labels=targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()

                if batch_idx % 10 == 0:
                    progress_bar.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCH}, Loss: {avg_loss:.4f}")
        
        # Save model checkpoint
        os.makedirs(f"{MODEL_DIR}/bert", exist_ok=True)
        torch.save(model.state_dict(), f"{MODEL_DIR}/bert/model_epoch_{epoch+1}.pth")


if __name__ == '__main__':
    train_bert()
