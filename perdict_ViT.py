import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import ViTForImageClassification, ViTImageProcessor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import os

# --- Load model & processor ---
model_path = "./vit-hebrew-final"
model = ViTForImageClassification.from_pretrained(model_path)
processor = ViTImageProcessor.from_pretrained(model_path)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --- Hebrew letter label map (must match training!) ---
label_map = {
    0: "א", 1: "ב", 2: "ג", 3: "ד", 4: "ה", 5: "ו", 6: "ז", 7: "ח", 8: "ט",
    9: "י", 10: "כ", 11: "ך", 12: "ל", 13: "מ", 14: "ם", 15: "נ", 16: "ן",
    17: "ס", 18: "ע", 19: "פ", 20: "ף", 21: "צ", 22: "ץ", 23: "ק", 24: "ר",
    25: "ש", 26: "ת"
}

# --- Test dataset root ---
test_dir = r"C:\Users\LIAD\Desktop\python project\vision transformer\dataset fore cnn hebrew letters.v1i.folder\test"

# --- Define preprocessing ---
transform = Compose([
    Resize((224, 224)),  # Match ViT expected input size
    ToTensor(),
    Normalize(mean=processor.image_mean, std=processor.image_std)
])

# --- Collect test samples ---
y_true = []
y_pred = []
image_paths = []

for class_folder in sorted(os.listdir(test_dir), key=lambda x: int(x)):
    folder_path = os.path.join(test_dir, class_folder)
    class_idx = int(class_folder)

    for file in os.listdir(folder_path):
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(folder_path, file)
            img = Image.open(image_path).convert("RGB")
            input_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(pixel_values=input_tensor)
                logits = outputs.logits
                pred_class = logits.argmax(dim=1).item()

            y_true.append(class_idx)
            y_pred.append(pred_class)
            image_paths.append(image_path)

# --- Confusion Matrix ---
hebrew_labels = [label_map[i] for i in range(len(label_map))]
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=hebrew_labels)
plt.figure(figsize=(12, 12))
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix on Test Set with Hebrew Labels")
plt.tight_layout()
plt.savefig("vit_confusion_matrix_inference.png")
plt.show()

# --- Classification Report ---
print("\n=== Classification Report ===")
report = classification_report(y_true, y_pred, target_names=hebrew_labels, digits=3)
print(report)
