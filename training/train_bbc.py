import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from BBC.utils import Dataset, collate_fn, get_label 
from BBC.models.bbc_model import Model 
from BBC.config import *

if __name__ == '__main__':
    # Initialize the dataset and data loader
    train_dataset = Dataset(type='train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Get label mappings
    label_list, label2id, id2label = get_label()
    num_labels = len(label_list)

    # Initialize the model and move it to the specified device
    model = Model(num_labels=num_labels).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Training loop
    for epoch in range(EPOCH):
        model.train()
        total_loss = 0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCH}") as progress_bar:
            for batch_idx, (inputs, targets, masks) in enumerate(progress_bar):
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                masks = masks.to(DEVICE)

                assert masks[:, 0].all(), "First timestep mask contains invalid values!"

                # Forward pass
                loss = model(inputs, masks, labels=targets)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update total loss
                total_loss += loss.item()

                # Update progress bar
                if batch_idx % 10 == 0:
                    progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{EPOCH}] - Average Loss: {avg_loss:.4f}')

        # Save the model checkpoint
        torch.save(model.state_dict(), f"{MODEL_DIR}/bbc/model_epoch_{epoch+1}.pth")
