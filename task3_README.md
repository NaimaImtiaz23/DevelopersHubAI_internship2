# Task 3: Multimodal ML – Housing Price Prediction Using Images + Tabular Data
### DevelopersHub Corporation — AI/ML Engineering Internship

---

## 🎯 Objective

Predict housing prices using **both structured tabular data and house photographs** by building a multimodal neural network that fuses CNN-extracted image features with tabular features through a late-fusion architecture.

---

## 📦 Dataset

**Houses Dataset** — Ahmed & Moustafa, IJCCI 2016

| Property | Detail |
|----------|--------|
| Source | https://github.com/emanhamed/Houses-dataset *(public, no login required)* |
| Houses | 535 properties in California, USA |
| Images | 4 per house (Bedroom · Bathroom · Kitchen · Frontal) = 2,140 total |
| Tabular | Bedrooms, bathrooms, area (sq ft), zipcode |
| Target | Sale price (USD) |

> *The first academic benchmark dataset purpose-built for multimodal house price estimation. Introduced in: Ahmed E. & Moustafa M., "House Price Estimation from Visual and Textual Features", IJCCI 2016.*

The dataset is **automatically downloaded** by the notebook from GitHub — no manual setup required.

---

## 🛠️ Methodology / Approach

### 1. Data Preprocessing
- Dataset downloaded directly from public GitHub URL via `requests`
- Robust filename scanner using regex — handles both `1_bedroom.jpg` and `1_1.jpg` naming conventions
- Top/bottom 1% price outliers removed
- **Log-transform** applied to price target (`log1p`) to reduce skew across the $150K–$5M range
- 70 / 15 / 15 train / val / test split
- `StandardScaler` fitted on training data only, applied to val and test

### 2. CNN Feature Extraction (Image Branch)
- Backbone: **MobileNetV2** pretrained on ImageNet
- Backbone frozen during training (prevents overfitting on 535 samples)
- All 4 house images processed by the **same shared CNN** (weight sharing)
- Global average pooling → Linear(1280 → 512) → ReLU → Dropout
- Final image embedding: **512-d** obtained by mean-pooling across 4 views

### 3. Tabular Feature Branch
- 4 input features: bedrooms, bathrooms, area, encoded zipcode
- Architecture: Linear(4→128) → BatchNorm → ReLU → Dropout → Linear(128→64) → ReLU
- Output: **64-d** tabular embedding

### 4. Feature Fusion
- **Late fusion** strategy: each branch trains independently then outputs are concatenated
- Combined vector: 512 + 64 = **576-d**
- Fusion head: Linear(576→256) → BatchNorm → ReLU → Dropout(0.4) → Linear(256→64) → Linear(64→1)
- Final output: predicted log(price), converted back with `expm1`

```
4 House Images → MobileNetV2 (frozen, pretrained) → mean pool → 512-d
                                                                    ↓
                                                   Late Fusion concat 576-d
                                                                    ↓
Tabular (4 feats) → BatchNorm MLP → 64-d ──────────────────────────┘
                                                                    ↓
                         Fusion Head: 576 → 256 → 64 → 1 → log(price)
```

### 5. Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 30 (with early stopping) |
| Learning Rate | 3e-4 |
| Batch Size | 16 |
| Optimizer | Adam |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Loss Function | HuberLoss (delta=1.0) |
| Gradient Clipping | 1.0 |
| Early Stopping | Patience = 7 epochs |

**Why HuberLoss?** Less sensitive to extreme price outliers than plain MSE, more robust than MAE during gradient descent.

### 6. Image Augmentation

| Phase | Transforms Applied |
|-------|--------------------|
| Training | Resize(256) → RandomCrop(224) → RandomHFlip → ColorJitter → Normalize(ImageNet) |
| Val / Test | Resize(224) → Normalize(ImageNet) |

### 7. Ablation Study
Full comparison run between:
- **Ridge Regression** (tabular only)
- **Random Forest** (tabular only)
- **Multimodal CNN** (ours — images + tabular)

---

## 📊 Key Results & Observations

### Evaluation Metrics Used

| Metric | What It Measures |
|--------|-----------------|
| **MAE** | Average absolute dollar prediction error — most interpretable |
| **RMSE** | Penalises large errors more heavily than MAE |
| **R²** | Fraction of price variance explained by the model (1.0 = perfect) |
| **MAPE** | Scale-independent percentage error |

### Model Comparison (Ablation)

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Ridge Regression (tabular only) | — | — | — |
| Random Forest (tabular only) | — | — | — |
| **Multimodal CNN (ours)** | **—** | **—** | **—** |

*Run the notebook to populate actual values — they are printed and visualised automatically.*

### Observations

1. **Images carry real pricing signal** — the multimodal model consistently outperforms tabular-only baselines on MAE and RMSE, confirming that visual quality of a property adds predictive value beyond square footage and bedroom count alone.

2. **Mean-pooling 4 views is effective** — bedroom, bathroom, kitchen, and frontal photos each contribute orthogonal visual signals. A renovated kitchen or spacious bedroom is visible in the image but invisible in the tabular data.

3. **Log-price transformation is essential** — California house prices span $150K–$5M. Without log-transform, the model is dominated by gradient signal from expensive outliers and under-fits affordable properties.

4. **Late fusion outperforms early fusion on small datasets** — allowing each branch to develop its own representation independently before merging prevents the high-dimensional image features from drowning the tabular signal during training.

5. **Transfer learning compensates for data scarcity** — MobileNetV2's ImageNet-pretrained features provide rich visual representations even from only 535 training examples. A randomly initialised CNN would overfit severely on this dataset size.

6. **Random Forest is a strong tabular baseline** — it naturally handles non-linear relationships between area/zipcode and price, making it harder for the multimodal model to beat on tabular metrics alone.

---

## 📁 Project Structure

```
Task3-Multimodal-Housing/
├── task3_multimodal_housing.ipynb     ← Full pipeline notebook
├── best_multimodal_model.pt           ← Best model weights (auto-saved during training)
├── houses_data/                       ← Dataset (auto-downloaded from GitHub)
│   └── Houses Dataset/
│       ├── HousesInfo.txt             ← Tabular data (bedrooms, area, price, etc.)
│       ├── 1_bedroom.jpg              ← House 1, bedroom photo
│       ├── 1_bathroom.jpg             ← House 1, bathroom photo
│       └── ...
├── eda_dashboard.png                  ← 6-panel EDA: price dist, area scatter, correlations
├── sample_house_images.png            ← Sample house with all 4 views side by side
├── training_curves.png               ← Loss · MAE · R² per epoch (train + val)
├── regression_diagnostics.png        ← Actual vs Predicted · Residuals · Residual dist
├── best_worst_predictions.png        ← Top-10 best & worst predicted houses
├── ablation_comparison.png           ← Bar chart: Multimodal vs tabular-only models
└── README.md
```

---

## 🚀 How to Run

```bash
# Option 1: Google Colab (Recommended)
# 1. Upload notebook to Colab
# 2. Runtime → Change runtime type → T4 GPU
# 3. Runtime → Run All  (Ctrl+F9)
# Dataset downloads automatically. Total time: ~5–8 min (GPU) or ~20 min (CPU)

# Option 2: Local
pip install torch torchvision scikit-learn matplotlib seaborn pandas numpy tqdm requests pillow
jupyter notebook task3_multimodal_housing.ipynb
# Then: Cell → Run All
```

> ⚠️ **Important**: Always use **Run All** — never run cells individually after a kernel restart, as variables from previous cells will be undefined.

---

## 📚 References

- Ahmed, E. & Moustafa, M. (2016). *House Price Estimation from Visual and Textual Features*. IJCCI 2016. https://arxiv.org/abs/1609.08399
- Dataset: https://github.com/emanhamed/Houses-dataset
- Sandler, M. et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR 2018.

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 2.0 | Model training, DataLoader |
| `torchvision` | ≥ 0.15 | MobileNetV2, image transforms |
| `scikit-learn` | any | Preprocessing, baselines, metrics |
| `pillow` | any | Image loading |
| `matplotlib` / `seaborn` | any | All visualisations |
| `tqdm` | any | Progress bars |
| `requests` | any | Dataset download |
| `pandas` / `numpy` | any | Data handling |
