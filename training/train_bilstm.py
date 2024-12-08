import torch
from torch.utils.data import DataLoader
from models.bilstm_model import ClassicBiLSTMNER
from utils import Dataset, collate_fn, get_label
from config import *
import os

def train_classic_bilstm():
    # Load label information
    label_list, label2id, id2label = get_label()
    num_labels = len(label_list)
    
    # Initialize dataset and dataloader
    train_dataset = Dataset(type='train')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    # Initialize model
    vocab_size = len(get_vocab()[1])  # Assuming get_vocab() returns (id2word, word2id)
    model = ClassicBiLSTMNER(num_labels=num_labels, vocab_size=vocab_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # Training loop
    for epoch in range(EPOCH):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids, labels, attention_mask = [x.to(DEVICE) for x in batch]
            optimizer.zero_grad()
            loss = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCH}, Loss: {avg_loss:.4f}")
        
        # Save model checkpoint
        os.makedirs(f"{MODEL_DIR}/classic_bilstm", exist_ok=True)
        torch.save(model.state_dict(), f"{MODEL_DIR}/classic_bilstm/model_epoch_{epoch+1}.pth")

if __name__ == '__main__':
    train_classic_bilstm()
