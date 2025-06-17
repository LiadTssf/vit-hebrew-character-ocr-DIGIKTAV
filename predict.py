import torch
import numpy as np
from transformers import ViTForImageClassification, ViTImageProcessor, Trainer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import os
import shutil
import pandas as pd  # ✅ NEW
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

class NumericSortedImageFolder(ImageFolder):
    def find_classes(self, directory):
        classes = sorted([d.name for d in os.scandir(directory) if d.is_dir()], key=lambda x: int(x))
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
    
# --- Reload model and processor ---
model = ViTForImageClassification.from_pretrained("./vit-hebrew-final")
processor = ViTImageProcessor.from_pretrained("./vit-hebrew-final")

# --- Data transform ---
transform = Compose([
    Resize((72, 72)),
    ToTensor(),
    Normalize(mean=processor.image_mean, std=processor.image_std)
])

# --- Load test dataset ---
dataset_path = r"C:\Users\LIAD\Desktop\python project\vision transformer\dataset fore cnn hebrew letters.v1i.folder"

# --- Convert dataset ---
class HFStyleDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return {"pixel_values": image, "label": label}

    def __len__(self):
        return len(self.dataset)

# --- Run predictions ---
trainer = Trainer(model=model, tokenizer=processor)


label_map = {
    0: "א", 1: "ב", 2: "ג", 3: "ד", 4: "ה", 5: "ו", 6: "ז", 7: "ח", 8: "ט",
    9: "י", 10: "כ", 11: "ך", 12: "ל", 13: "מ", 14: "ם", 15: "נ", 16: "ן",
    17: "ס", 18: "ע", 19: "פ", 20: "ף", 21: "צ", 22: "ץ", 23: "ק", 24: "ר",
    25: "ש", 26: "ת"
}

# Corrected label order based on numeric folder names
folder_names = sorted(os.listdir(os.path.join(dataset_path, "test")), key=lambda x: int(x))
test_ds = NumericSortedImageFolder(os.path.join(dataset_path, "test"), transform=transform)
image_paths = [s[0] for s in test_ds.samples]
print("Class-to-Index Map:", test_ds.class_to_idx)
test_ds.classes = folder_names
test_ds.class_to_idx = {f: int(f) for f in folder_names}
labels = [label_map[int(f)] for f in folder_names]
print(model.config.id2label)
print(model.config.label2id)

preds_output = trainer.predict(HFStyleDataset(test_ds))
y_true = preds_output.label_ids
y_pred = np.argmax(preds_output.predictions, axis=1)

# --- Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
fig, ax = plt.subplots(figsize=(12, 12))
disp.plot(cmap="Blues", ax=ax, xticks_rotation=90)
plt.title("Confusion Matrix on Test Set with Hebrew Labels")
plt.tight_layout()
plt.show()

# --- Classification Report ---
print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=labels, digits=3))

# --- Save prediction results as Excel ---
os.makedirs("predictions_log/correct", exist_ok=True)
os.makedirs("predictions_log/wrong", exist_ok=True)

rows = []
for idx, (true, pred, logit, path) in enumerate(zip(y_true, y_pred, preds_output.predictions, image_paths)):
    true_label = labels[true]
    pred_label = labels[pred]
    confidence = torch.softmax(torch.tensor(logit), dim=0)[pred].item()
    is_correct = (true == pred)

    rows.append({
        "True Label": true_label,
        "Predicted Label": pred_label,
        "Confidence": round(confidence, 4),
        "Correct?": is_correct,
        "Image Path": path
    })

    dest_folder = "correct" if is_correct else "wrong"
    shutil.copy(path, f"predictions_log/{dest_folder}/{true_label}_as_{pred_label}_{os.path.basename(path)}")

df = pd.DataFrame(rows)
df.to_excel("predictions_log/prediction_report.xlsx", index=False, engine='openpyxl')  # ✅ Excel output
