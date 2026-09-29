# Brain Tumor Classification & Benchmarking Pipeline

Complete Deep Learning and Machine Learning pipeline for brain tumor classification from MRI images (`.pt` tensor files), combining state-of-the-art architectures, Transfer Learning, attention mechanisms (CBAM), Vision Transformers, and explainability analysis (XAI).

---

## Project Objective

The main objective is to rigorously design, train, evaluate, and compare different families of models for binary/clinical detection of brain tumors (Healthy vs Tumor):

1. **Classical Machine Learning Models** on tabular features (with PCA and SMOTE).
2. **Custom CNNs** (SimpleCNN, DeepCNN, and CNNs with spatial and channel attention blocks **CBAM**).
3. **Transfer Learning** (DenseNet-121 and ResNet-50 with locally pre-trained weights).
4. **Vision Transformer (ViT-B/16)** for global attention.
5. **Explainable AI (XAI)** via **Grad-CAM** to validate the clinical relevance of targeted regions.

---

## Project Organization

```text
brain-tumor-project/
│
├── data/
│   └── processed/
│       └── augmented_images/
│           ├── yes/          # .pt tensors with tumor
│           └── no/           # .pt healthy tensors
│
├── models/                   # Local model weights (.pth)
│   ├── SimpleCNN_Trained.pth
│   ├── DeepCNN_Trained.pth
│   ├── Attention_CNN_Trained.pth
│   ├── resnet50_retrained.pth
│   └── vit_b_16-c867db91.pth
│
├── notebooks/                # Jupyter Notebooks by step
│   ├── 04_deep_learning_cnn.ipynb
│   ├── 06_transfer_learning.ipynb
│   ├── 07_vision_transformer.ipynb
│   ├── 08_global_results_comparison.ipynb
│   └── 09_explainability_xai.ipynb
│
├── src/                      # Reusable Python modules
│   ├── models.py             # Architecture definitions (CNN, CBAM, ViT, Transfer)
│   └── training.py           # Training, validation, and early stopping functions
│
├── .env                      # Environment variables
└── README.md                 # Project documentation
```

---

## Global Results Summary

The global benchmark evaluates all approaches on the test set:

| Model | Category | Accuracy | Precision | Recall | F1-Score |
|-------|----------|----------|-----------|--------|----------|
| Vision Transformer (ViT) | Transformer | 100.00% | 1.0000 | 1.0000 | 1.0000 |
| ResNet-50 (Transfer) | Transfer Learning | 98.68% | 0.9787 | 1.0000 | 0.9892 |
| DeepAttentionCNN (CBAM) | Deep Learning | 93.42% | 0.9767 | 0.9130 | 0.9438 |
| Extra Trees | Classical ML | 91.94% | 0.9062 | 0.9355 | 0.9206 |
| MLP Classifier | Classical ML | 89.52% | 0.8769 | 0.9194 | 0.8976 |
| DenseNet-121 (Transfer) | Transfer Learning | 89.47% | 0.8958 | 0.9348 | 0.9149 |
| DeepCNN | Deep Learning | 89.47% | 0.8654 | 0.9783 | 0.9184 |
| Random Forest | Classical ML | 88.71% | 0.8750 | 0.9032 | 0.8889 |
| SimpleCNN | Deep Learning | 76.32% | 0.7333 | 0.9565 | 0.8302 |

---

## Detailed Performance Analysis

### 1. Best Tabular Machine Learning Model: Extra Trees Classifier

- **Accuracy**: ~91.94% (highest overall success rate).
- **Precision**: ~90.62% (low false positive rate; when it predicts a tumor, it is correct in more than 90% of cases).
- **Recall**: ~93.55% (critical metric in medical imaging: detects more than 93% of actual tumors to minimize false negative risk).
- **F1-Score**: ~92.06% (excellent trade-off between precision and recall).

#### Reasons for These Performances:

- **Suitability of Tree Models**: Tabular features (color histograms, Laplace textures, perimeters) combined with PCA reduction (30 components) and SMOTE rebalancing form a non-linear space. Tree ensemble algorithms (Extra Trees, Random Forest) capture these interactions without linearity assumptions.
- **Why Extra Trees Outperforms Random Forest?** It introduces additional randomization when choosing split thresholds, which reduces overall variance, improves robustness to SMOTE noise, and limits overfitting.
- **MLP Strength**: The multi-layer perceptron exploits the 30 PCA dimensions to model complex non-linear relationships through its hidden layers.

### 2. The Top of the Pyramid: Deep Learning & Transformers

- **Vision Transformer (ViT) — 100%**: Its global self-attention mechanism allows simultaneous analysis of all image regions and captures tumor morphology without error.
- **ResNet-50 — 98.68% (Recall = 1.00)**: A 100% recall means zero false negatives (no missed tumors), ensuring maximum clinical safety.
- **DeepAttentionCNN (CBAM) — 93.42%**: Integration of spatial and channel attention blocks acts as a focusing mask to ignore background noise and precisely target the lesion.

---

## Critical Analysis, Limitations, and Explainability (Grad-CAM)

### Concordance and Variability

On many samples, the maximum activation zone (red/yellow) precisely overlaps the tumor lesion, confirming that the network relies on the correct Region of Interest (ROI). However, in other cases, the map appears more diffuse or slightly shifted.

### The Shortcut Learning Phenomenon

A model can correctly predict the presence of a tumor while sometimes relying on global contextual signals (such as overall brain structure asymmetry or background contrasts) rather than the strict contours of the lesion.

### Spatial Resolution Trade-off

The choice of target layer (for example, between intermediate and deep layers) involves a trade-off:

- Intermediate layers preserve finer spatial details but are noisier.
- Final layers offer strong semantic coherence at the expense of millimeter-level precision.

### Impact of Dataset Size and Diversity

Expanding the training database is a fundamental lever for improving explainability reliability. By increasing the volume of images with greater inter-patient and inter-scanner variability, the network learns to overcome artifacts and spurious correlations, enabling much more stable Grad-CAM activations strictly confined to pathological regions.

### Methodological Rigor

Highlighting these imperfections demonstrates an objective scientific approach (absence of cherry-picking), illustrating the inherent complexity of interpreting neural black boxes in medical imaging.
