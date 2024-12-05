import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

from utils import Dataset, collate_fn
from model import Model
from config import DEVICE, LR, EPOCH, MODEL_DIR, GRAD_CLIP



if __name__ == '__main__':
    # Initialize the dataset and data loader
    dataset = Dataset()
    loader = DataLoader(
        dataset,
        batch_size=100,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Initialize the model and move it to the specified device
    model = Model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Training loop
    for epoch in range(EPOCH):
        model.train()  # Set model to training mode
        total_loss = 0  # For tracking the loss
        with tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCH}") as progress_bar:
            for batch_idx, (inputs, targets, masks) in enumerate(progress_bar):
                # Move data to the specified device
                inputs = inputs.to(DEVICE)
                masks = masks.to(DEVICE)
                targets = targets.to(DEVICE)

                # Forward pass
                outputs = model(inputs, masks)
                loss = model.loss_fn(inputs, targets, masks)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update total loss
                total_loss += loss.item()

                # Update progress bar
                if batch_idx % 10 == 0:
                    progress_bar.set_postfix(loss=loss.item())

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(loader)
        print(f'Epoch [{epoch+1}/{EPOCH}] - Average Loss: {avg_loss:.4f}')

        # Save the model checkpoint
        torch.save(model.state_dict(), f"{MODEL_DIR}/model_epoch_{epoch+1}.pth")
