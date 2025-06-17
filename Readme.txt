This internal repository contains training scripts, model outputs, and evaluation artifacts for character-level Hebrew handwriting recognition using Vision Transformers (ViT). All models were developed as part of the Digi-Ktav OCR project.

📁 Repository Structure
dataset for cnn hebrew letters.v1i.folder/
Raw and augmented image dataset used for training all models.

vit-hebrew-final/
Directory containing final fine-tuned Vision Transformer weights and supporting files.

logs/
Training logs and metrics in CSV format.

predictions_log/
Per-character prediction results and analysis outputs.

ViT.py / ViT2.py
Vision Transformer model definitions and training pipeline. Both files contain similar code; one was used for final training, the other for experimentation.

predict.py / perdict_ViT.py
Scripts for running inference on new samples using trained ViT models.

training_log.csv
Main training performance log (accuracy/loss per epoch).

wrong_predictions.csv
List of incorrect predictions for targeted error analysis.

vit-hebrew-final.zip
Zipped archive of final trained model (used in production OCR backend). can be found in releases

confusion_matrix_pretrained_google.png
Visual confusion matrix comparing ground truth vs ViT pretrained model predictions.

vit_confusion_matrix_inference.png
Confusion matrix from inference-time results on validation set.

🧠 Model Summary
Three model variants were trained using the same dataset:

A baseline CNN (trained separately, results referenced in report)

A ViT trained from scratch on the Hebrew dataset

A fine-tuned ViT initialized with weights from HuggingFace’s google/vit-base-patch16-224

The best-performing model (fine-tuned ViT) reached:

Accuracy: 93.0%

Inference speed: ~40ms per character

🗂 Dataset
The training data was created by merging the following sources:

HHD handwritten dataset

Moranzargari Hebrew Letters collection

Sofia Naer’s open-source Hebrew characters set

Images were normalized to a consistent size (224×224) and labeled into 27 Hebrew letter classes. Augmentation was applied to increase robustness (rotation, noise, shift, blur). See ViT.py for preprocessing code.

⚙️ Notes
This repository is not set up for deployment. It serves as the internal training environment used to generate the models shipped in Digi-Ktav
