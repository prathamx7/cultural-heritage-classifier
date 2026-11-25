# Cultural Heritage Site Classifier

Image classification of 14 world cultural heritage sites using transfer learning and convolutional neural networks (CNNs), deployed as an interactive Streamlit web app.

## Overview
Upload a photo of a heritage site and the model predicts the most likely site, shows the top-5 class probabilities with a confidence level (high, moderate or low), and displays the site's civilization and location.

## Methodology
- **Transfer learning:** five ImageNet-pretrained CNN architectures were trained on an augmented heritage image dataset.
- **Model comparison:** each model was evaluated on a held-out test set using accuracy, precision, recall, F1-score and parameter count.
- **Model selection:** EfficientNetB0 was chosen as the deployed model for its best accuracy and its small size (4.38M parameters).
- **Inference:** architecture-specific preprocessing is applied before prediction, and the app returns softmax probabilities for the top 5 classes.

## Results

| Model | Test Accuracy | Precision | Recall | F1-Score | Params (M) | Train Time (s) |
|---|---|---|---|---|---|---|
| **EfficientNetB0** | **0.9627** | 0.9641 | 0.9627 | 0.9627 | 4.38 | 207.0 |
| MobileNetV2 | 0.9351 | 0.9364 | 0.9351 | 0.9346 | 2.59 | 194.4 |
| ResNet50V2 | 0.9315 | 0.9338 | 0.9315 | 0.9315 | 24.09 | 249.9 |
| DenseNet121 | 0.9195 | 0.9228 | 0.9195 | 0.9195 | 7.30 | 333.8 |
| Xception | 0.8594 | 0.8656 | 0.8594 | 0.8583 | 21.39 | 442.5 |

EfficientNetB0 gave the
