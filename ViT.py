import torch
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Pad, RandomAffine, RandomRotation, ColorJitter
from torch.utils.data import DataLoader
from transformers import (
    ViTConfig,
    ViTForImageClassification,
    ViTImageProcessor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import evaluate
import os
from transformers import TrainerCallback
import csv
from datetime import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import shutil
from tqdm import tqdm
from PIL import Image, ImageOps




def log_metrics_to_csv(metrics, filename="training_log.csv"):
    file_exists = os.path.isfile(filename)
    with open(filename, "a", newline='', encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["timestamp", "epoch", "eval_loss", "eval_accuracy"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "epoch": metrics.get("epoch"),
            "eval_loss": metrics.get("eval_loss"),
            "eval_accuracy": metrics.get("eval_accuracy")
        })


class HFStyleDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return {"pixel_values": image, "label": label}

    def __len__(self):
        return len(self.dataset)


# --- Disable wandb ---
os.environ["WANDB_DISABLED"] = "true"

# --- Paths ---
dataset_path = r"C:\Users\LIAD\Desktop\python project\vision transformer\dataset fore cnn hebrew letters.v1i.folder"

# --- Define Hebrew letter mapping ---
label_map = {
    0: "א", 1: "ב", 2: "ג", 3: "ד", 4: "ה", 5: "ו", 6: "ז", 7: "ח", 8: "ט",
    9: "י", 10: "כ", 11: "ך", 12: "ל", 13: "מ", 14: "ם", 15: "נ", 16: "ן",
    17: "ס", 18: "ע", 19: "פ", 20: "ף", 21: "צ", 22: "ץ", 23: "ק", 24: "ר",
    25: "ש", 26: "ת"
}

# --- THIS IS THE KEY PART: Create a custom dataset class that enforces our label map ---
class HebrewLetterDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transform=None):
        """
        A custom dataset that enforces the folder number to Hebrew letter mapping
        
        Args:
            root: Dataset root folder
            split: 'train', 'valid', or 'test'
            transform: Optional transforms
        """
        self.root = os.path.join(root, split)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.classes = []
        
        # Get all folder numbers and sort them numerically
        folder_names = [f for f in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, f))]
        folder_names = sorted(folder_names, key=lambda x: int(x))
        
        # Set up class mapping - crucial for correct labeling
        self.classes = folder_names
        self.class_to_idx = {folder_name: idx for idx, folder_name in enumerate(folder_names)}
        
        # Validate against our label map
        assert len(folder_names) <= len(label_map), f"Found {len(folder_names)} folders but label_map only has {len(label_map)} entries"
        
        # Store folder number to actual index mapping for verification
        self.folder_to_index = {int(folder): idx for idx, folder in enumerate(folder_names)}
        
        # Create samples list with (path, class_index) tuples
        for folder_name in folder_names:
            folder_path = os.path.join(self.root, folder_name)
            folder_idx = self.class_to_idx[folder_name]
            
            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(folder_path, img_name)
                    self.samples.append((img_path, folder_idx))
    
    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        img = plt.imread(img_path)

        # Handle grayscale images
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=2)

        # Convert RGBA to RGB
        if img.shape[2] == 4:
            img = img[:, :, :3]

        img = (img * 255).astype(np.uint8)  # Ensure correct dtype before PIL conversion
        img = Image.fromarray(img)  # ✅ Convert NumPy array to PIL Image here

        if self.transform:
            img = self.transform(img)

        return img, class_idx
    
    def __len__(self):
        return len(self.samples)
    
    def get_hebrew_letter(self, idx):
        """Get the Hebrew letter for a class index"""
        return label_map[idx]
    
    def print_mapping(self):
        """Print folder to Hebrew letter mapping for verification"""
        print("\n=== DATASET MAPPING VERIFICATION ===")
        for folder in self.classes:
            idx = self.class_to_idx[folder]
            hebrew = label_map[idx]
            print(f"Folder '{folder}' → Index {idx} → Letter '{hebrew}'")
        print("====================================\n")

# --- Load image processor ---
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

transform = Compose([
    Resize((224, 224)),  # Match ViT base expected size
    RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.95, 1.05)),
    RandomRotation(3),
    ColorJitter(brightness=0.1, contrast=0.1),
    ToTensor(),
    Normalize(mean=processor.image_mean, std=processor.image_std),
])
# --- Load datasets with our custom class ---
train_ds = HebrewLetterDataset(dataset_path, "train", transform=transform)
val_ds = HebrewLetterDataset(dataset_path, "valid", transform=transform)
test_ds = HebrewLetterDataset(dataset_path, "test", transform=transform)

# --- Print and verify mapping ---
train_ds.print_mapping()
val_ds.print_mapping()
test_ds.print_mapping()

# --- Create Hebrew letter labels list ---
num_classes = len(train_ds.classes)
labels = [label_map[i] for i in range(num_classes)]

# --- Configure ViT model ---
id2label = {i: label_map[i] for i in range(num_classes)}
label2id = {label_map[i]: i for i in range(num_classes)}

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=num_classes,
    id2label=id2label,
    label2id=label2id,
)

# --- Use GPU if available ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

# --- Metrics ---
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    
    # Add more detailed reporting
    accuracy_val = accuracy.compute(predictions=preds, references=labels)
    
    # Count correct predictions per class
    classes, counts = np.unique(labels, return_counts=True)
    correct_counts = np.zeros_like(counts)
    
    for i, class_idx in enumerate(classes):
        class_preds = preds[labels == class_idx]
        correct_counts[i] = np.sum(class_preds == class_idx)
    
    class_accuracies = correct_counts / counts
    
    # Print detailed class accuracies for monitoring
    print("\n=== Per-Class Accuracy ===")
    for i, class_idx in enumerate(classes):
        hebrew = label_map[class_idx]
        print(f"Class {class_idx} ({hebrew}): {class_accuracies[i]:.4f} ({correct_counts[i]}/{counts[i]})")
    
    return accuracy_val

# --- Training arguments ---
#training_args = TrainingArguments(
#    output_dir="./vit-hebrew",
#    per_device_train_batch_size=32,
#    per_device_eval_batch_size=32,
#    warmup_steps=500,
#    learning_rate=5e-5,  # try reducing from default 5e-5 to 3e-5 or 2e-5
#    num_train_epochs=100,
#    logging_dir="./logs",
#    logging_steps=100,
#    save_steps=1000,
#    save_total_limit=2,
#    fp16=torch.cuda.is_available()
#)
training_args = TrainingArguments(
    output_dir="./vit-hebrew",
    per_device_train_batch_size=16,       # ViT base needs more memory
    per_device_eval_batch_size=16,
    num_train_epochs=50,                  # Fine-tuning needs fewer epochs
    learning_rate=3e-5,
    warmup_steps=500,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    logging_dir="./logs",
    logging_steps=100,
    save_steps=1000,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
)
class ManualEvalCallback(TrainerCallback):
    def __init__(self, eval_steps=1000):
        self.eval_steps = eval_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps == 0 and state.global_step != 0:
            control.should_evaluate = True    

class LoggingCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            log_metrics_to_csv(metrics)
            print(f"🔍 Epoch {metrics.get('epoch', '?'):.2f} - Validation Accuracy: {metrics.get('eval_accuracy', 0):.4f}")

# --- Create Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=HFStyleDataset(train_ds),
    eval_dataset=HFStyleDataset(val_ds),
    compute_metrics=compute_metrics,
    tokenizer=processor,
    callbacks=[
        LoggingCallback(),
        ManualEvalCallback(eval_steps=2000)
    ]
)

# --- Print final verification before training ---
print("\n=== FINAL VERIFICATION BEFORE TRAINING ===")
print(f"Number of classes: {num_classes}")
print(f"Label mapping in model config:")
for idx, letter in id2label.items():
    print(f"  Index {idx} → Letter '{letter}'")
print("=======================================\n")

# --- Train the model ---
print("Starting training...")
trainer.train()

# --- Final evaluation ---
print("\n=== Test Evaluation ===")
results = trainer.evaluate(HFStyleDataset(test_ds))
print(results)

# --- Save the model ---
model.config.id2label = id2label
model.config.label2id = label2id
model.save_pretrained("./vit-hebrew-final", push_to_hub=False)
processor.save_pretrained("./vit-hebrew-final")

# --- Make predictions on test set ---
print("\n=== Generating predictions on test set ===")
preds_output = trainer.predict(HFStyleDataset(test_ds))
y_true = preds_output.label_ids
y_pred = np.argmax(preds_output.predictions, axis=1)

# --- Create confusion matrix ---
cm = confusion_matrix(y_true, y_pred)
hebrew_labels = [label_map[i] for i in range(num_classes)]

# --- Display confusion matrix ---
plt.figure(figsize=(12, 12))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=hebrew_labels)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix on Test Set with Hebrew Labels")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# --- Print classification report ---
print("\n=== Classification Report ===")
report = classification_report(y_true, y_pred, target_names=hebrew_labels, digits=3)
print(report)

# --- Analyze wrong predictions ---
wrong_indices = np.where(y_pred != y_true)[0]
print(f"\n=== Found {len(wrong_indices)} incorrect predictions ===")

# Use the test_ds.samples to get image paths
test_samples = test_ds.samples
image_paths = [s[0] for s in test_samples]

print("\n=== Sample of incorrect predictions ===")
for idx in wrong_indices[:min(20, len(wrong_indices))]:
    true_label_idx = y_true[idx]
    pred_label_idx = y_pred[idx]
    true_letter = label_map[true_label_idx]
    pred_letter = label_map[pred_label_idx]
    image_path = image_paths[idx]
    folder_name = os.path.basename(os.path.dirname(image_path))
    
    print(f"Image: {os.path.basename(image_path)} (Folder: {folder_name})")
    print(f"  True: {true_letter} (Index {true_label_idx})")
    print(f"  Pred: {pred_letter} (Index {pred_label_idx})")
    print("------")

# --- Save detailed wrong predictions to file ---
with open("wrong_predictions.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Image", "Folder", "True Index", "True Letter", "Pred Index", "Pred Letter"])
    
    for idx in wrong_indices:
        true_label_idx = y_true[idx]
        pred_label_idx = y_pred[idx]
        true_letter = label_map[true_label_idx]
        pred_letter = label_map[pred_label_idx]
        image_path = image_paths[idx]
        folder_name = os.path.basename(os.path.dirname(image_path))
        
        writer.writerow([
            os.path.basename(image_path),
            folder_name,
            true_label_idx,
            true_letter,
            pred_label_idx,
            pred_letter
        ])

print("\nAnalysis complete! Check 'wrong_predictions.csv' for detailed error analysis.")