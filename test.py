import torch
from torch.utils.data import DataLoader
from seqeval.metrics import classification_report
from utils import Dataset, collate_fn, get_label, report
from model import Model
from config import *

if __name__ == '__main__':
    # Initialize the test dataset and data loader
    test_dataset = Dataset(type='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    label_list, label2id, id2label = get_label()
    num_labels = len(label_list)

    # Instantiate the model
    model = Model(num_labels=num_labels)
    
    # Load the saved state dictionary
    try:
        state_dict = torch.load(MODEL_DIR + 'model_epoch_50.pth', map_location=DEVICE, weights_only=True)
    except TypeError:
        # If weights_only is not supported, omit it
        state_dict = torch.load(MODEL_DIR + 'model_epoch_50.pth', map_location=DEVICE)
    
    model.load_state_dict(state_dict)
    
    # Move the model to the specified device and set to evaluation mode
    model.to(DEVICE)
    model.eval()

    y_true_list = []
    y_pred_list = []

    # Get label mappings
    _, _, id2label = get_label()

    with torch.no_grad():
        for b, (inputs, targets, masks) in enumerate(test_loader):
            # Move data to the specified device
            inputs = inputs.to(DEVICE)
            masks = masks.to(DEVICE)
            targets = targets.to(DEVICE)

            # Get predictions
            y_pred = model(inputs, masks)  # Returns list of predicted label indices

            # Compute the loss
            loss = model(inputs, masks, labels=targets)

            print(f'>> batch: {b}, loss: {loss.item()}')
        
            # Convert predicted and true labels to label names
            for lst in y_pred:
                y_pred_list.append([id2label[i] for i in lst])
            for y, m in zip(targets, masks):
                y_true_list.append([id2label[i.item()] for i in y[m == True]])

    # Generate and print the evaluation report
    print(report(y_true_list, y_pred_list))
