import torch
from torch.utils.data import DataLoader
from seqeval.metrics import classification_report, accuracy_score
from utils import Dataset, collate_fn, get_label
from model import Model
from config import *
from transformers import BertTokenizerFast

def main():
    # **Step 1: Load Label Information**
    label_list, label2id, id2label = get_label()
    num_labels = len(label_list)
    print(f"Number of Labels: {num_labels}")

    # **Step 2: Initialize the Model**
    model = Model(num_labels=num_labels)
    
    # **Step 3: Load Trained Weights**
    model_path = f"{MODEL_DIR}/model_epoch_50.pth"  # Adjust epoch number as needed
    try:
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print(f"Loaded model weights from {model_path}")
    except FileNotFoundError:
        print(f"Model file {model_path} not found.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.to(DEVICE)
    model.eval()

    # **Step 4: Initialize the Test Dataset and DataLoader**
    test_dataset = Dataset(type='test')  # Ensure this loads the test split
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )
    print(f"Number of Test Samples: {len(test_dataset)}")

    y_true_list = []
    y_pred_list = []

    # **Step 5: Run Predictions and Collect Labels**
    with torch.no_grad():
        for batch_idx, (inputs, targets, masks) in enumerate(test_loader):
            # Move data to the specified device
            inputs = inputs.to(DEVICE)
            masks = masks.to(DEVICE)
            targets = targets.to(DEVICE)

            # **Ensure Mask Validity**
            if not masks[:, 0].all():
                print(f"Warning: Not all first timestep masks are set to True in batch {batch_idx}")

            # **Get Predictions from the Model**
            y_pred = model(inputs, masks)  # Should return list of predicted label indices

            # **Convert Predictions and Targets to Label Names**
            for i in range(len(y_pred)):
                pred_labels = y_pred[i]
                true_labels = targets[i].tolist()
                mask = masks[i].tolist()

                # **Filter Out Padding Tokens Using Mask**
                pred_labels_filtered = [id2label[p] for p, m in zip(pred_labels, mask) if m]
                true_labels_filtered = [id2label[t] for t, m in zip(true_labels, mask) if m]

                y_pred_list.append(pred_labels_filtered)
                y_true_list.append(true_labels_filtered)

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1} batches.")

    # **Step 6: Compute Evaluation Metrics**
    accuracy = accuracy_score(y_true_list, y_pred_list)
    report = classification_report(y_true_list, y_pred_list, digits=4)

    print("\n=== Evaluation Metrics on Test Set ===\n")
    print(f"Accuracy: {accuracy:.4f}\n")
    print("Classification Report:")
    print(report)

    # **Optional: Save the Report to a File**
    with open(f"{MODEL_DIR}/evaluation_report_epoch_50.txt", "w") as f:
        f.write("=== Evaluation Metrics on Test Set ===\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"Saved evaluation report to {MODEL_DIR}/evaluation_report_epoch_50.txt")

if __name__ == '__main__':
    main()
