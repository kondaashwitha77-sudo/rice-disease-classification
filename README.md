# 🌾 Rice Crop Disease Detection & Classification System
**A Working Machine Learning Prototype based on Nature Scientific Reports (Alsakar et al., 2024) and ML Foundations**

---

## 📌 Executive Summary

This project implements an enhanced, end-to-end Machine Learning prototype for the automatic detection and classification of rice plant leaf diseases. The system faithfully implements the methodology presented in **`Research.pdf`** (*"An enhanced classification system of various rice plant diseases based on multi-level handcrafted feature extraction technique"*, Nature Scientific Reports, 2024) and incorporates classical and ensemble machine learning algorithms from **`ML ALL.pdf`**.

The pipeline combines:
1. **Logarithmic Contrast Enhancement** ($c = 2.0$) to overcome field illumination variances.
2. **Multi-Level Multi-Channel Local Binary Patterns (ML-MCLBP)** to capture multi-scale color-texture features across Red, Green, and Blue channels.
3. **Spatial Color Correlogram (CC)** to capture the spatial correlation of quantized color distributions.
4. **Concatenation Feature Fusion** unifying chromatic and structural representations.
5. **Machine Learning Classifiers** (Linear SVM, RBF SVM, Polynomial SVM, Decision Tree, Random Forest, AdaBoost, KNN, MLP, and Softmax Regression).
6. **Agronomic Diagnosis & Advisory Engine** providing actionable fungicide/bactericide and cultural control measures for farmers.

---

## 📂 Project Architecture

The directory structure in `MLP2` is organized into clean, modular, and self-documenting components:

```text
MLP2/
├── Dataset/                             # Benchmark Dataset 2 (DS2: 5 classes, 2800 images)
│   ├── Full_Train_Data/                 # Training set (510 images per class)
│   └── Test_Data_SVM/                   # Test set (50 images per class)
├── Rice Leaf Disease Images/            # Benchmark Dataset 1 (DS1: 4 classes, 5932 images)
├── rice leaf diseases dataset/          # Benchmark Dataset 3 (DS3: 3 classes, 4684 images)
├── Research.pdf                         # Nature Scientific Reports reference paper
├── ML ALL.pdf                           # Machine Learning theory reference guide
│
├── src/                                 # Core Source Modules
│   ├── __init__.py                      # Package initializer
│   ├── config.py                        # Paths, dataset references & hyperparameters
│   ├── preprocessing.py                 # Resizing, Data Augmentation & Log Transformation
│   ├── feature_extraction.py            # Vectorized MCLBP, Color Correlogram & Feature Fusion
│   ├── dataset_loader.py                # Dataset ingestion and compressed feature caching (.npz)
│   ├── models.py                        # Model factory from ML ALL.pdf & Research.pdf
│   └── evaluation.py                    # Accuracy, Precision, Recall, F1, Specificity, MCC
│
├── train.py                             # Command-line training & multi-model benchmarking script
├── predict.py                           # Single-image diagnostic inference script with advisory
├── app.py                               # Interactive Streamlit Web Application
├── features_cache/                      # Cached extracted features for sub-second loading
├── saved_models/                        # Serialized best-performing ML models (.joblib)
├── reports/                             # CSV benchmark evaluation reports
└── README.md                            # Comprehensive technical documentation (this file)
```

---

## 🗄️ Dataset Coverage in MLP2

The system natively supports all three benchmark datasets present in the workspace:

| Dataset ID | Directory Name | Number of Images | Classes Present |
| :--- | :--- | :---: | :--- |
| **DS1** | `Rice Leaf Disease Images` | **5,932** | Bacterialblight (1584), Blast (1440), Brownspot (1600), Tungro (1308) |
| **DS2** | `Dataset` (`Full_Train_Data` + `Test_Data_SVM`) | **2,800** | Blight (560), Blast (560), Brownspot (560), Tungro (560), Sheath Blight (560) |
| **DS3** | `rice leaf diseases dataset` | **4,684** | Bacterialblight (1604), Brownspot (1620), Leafsmut (1460) |

---

## 🔬 Mathematical Formulations & Operations

### 1. Image Preprocessing (`src/preprocessing.py`)

#### A. Image Resizing (Equation 2)
All raw input images vary in dimension (e.g. $300 \times 300$, $479 \times 320$, $256 \times 256$). They are standardized into a uniform $224 \times 224$ dimension:
$$D_{resized}(x, y) = D_t\left(\frac{x}{s}, \frac{y}{t}\right)$$
where $s$ and $t$ are horizontal and vertical scaling factors.

#### B. Data Augmentation (Equations 3–6)
To alleviate dataset imbalance and prevent overfitting, data transformations are applied:
- **Translation (Eq. 4)**: $D_{translated}(x, y) = D_t(x - \Delta x, y - \Delta y)$, with $\Delta x, \Delta y \in [-5, +15]$ pixels.
- **Flipping (Eq. 5)**:
  - Horizontal: $D_{hf}(x, y) = D_t(W - x - 1, y)$
  - Vertical: $D_{vf}(x, y) = D_t(x, H - y - 1)$
- **Rotation (Eq. 6)**:
  $$D_{rotated}(x, y) = D_t\Big(x\cos\theta - y\sin\theta, \; x\sin\theta + y\cos\theta\Big), \quad \theta \in [-10^\circ, +10^\circ]$$

#### C. Logarithmic Contrast Enhancement (Equations 7–8)
Under varying solar angle, shading, and weather conditions, leaf lesions can suffer from low contrast. Logarithmic transformation expands the dynamic range of dark pixels:
1. **Normalization (Eq. 7)**:
   $$D_{tNorm} = \frac{D_t}{255}$$
2. **Logarithmic Mapping (Eq. 8)**:
   $$D_{tLog} = c \cdot \ln(1 + D_{tNorm})$$
   with scaling parameter $c = 2.0$ (as established in Section 3.2 of `Research.pdf`).

---

### 2. Handcrafted Feature Extraction (`src/feature_extraction.py`)

#### A. Multi-Channel Local Binary Patterns (MCLBP) (Equations 11–12)
Classical LBP operates solely on grayscale images, discarding crucial chromatic indicators of fungal/bacterial necrosis. The proposed system extends LBP across all three RGB channels independently:
$$LBP(x_c, y_c) = \sum_{p=0}^{N-1} 2^p \cdot g\Big(B_p - B(x_c, y_c)\Big)$$
where $N = 8$ neighboring pixels at radius $R = 1$, and:
$$g(z) = \begin{cases} 1, & z \ge 0 \\ 0, & z < 0 \end{cases}$$

For an RGB image:
$$LBP(RGB) = \Big[ LBP(Red), \; LBP(Green), \; LBP(Blue) \Big]$$
Each color channel generates a 256-bin normalized histogram, yielding $256 \times 3 = 768$ features per block.

#### B. Multi-Level (ML) Spatial Block Decomposition (Equation 10)
Rather than computing a single global histogram that loses spatial layout, the image is decomposed into an $L \times L$ grid:
$$N = L^2, \quad D_{tLog} = \bigcup_{n=1}^N B_n$$
- Level $L = 1$: $1 \times 1 = 1$ block ($768$ features)
- Level $L = 2$: $2 \times 2 = 4$ blocks ($3,072$ features)
- Level $L = 3$: $3 \times 3 = 9$ blocks ($6,912$ features)
- Level $L = 4$: $4 \times 4 = 16$ blocks ($12,288$ features)
- Level $L = 5$: $5 \times 5 = 25$ blocks ($19,200$ features)

#### C. Spatial Color Correlogram (Equation 9)
While standard color histograms only represent total color counts without spatial location, the Color Correlogram expresses the probability of finding a pixel of color $c_j$ at a specific spatial distance $D$ from a pixel of color $c_i$:
$$\gamma^{(D)}_{c_i, c_j} = \Pr_{p_1 \in I_{c_i}, \; p_2 \in I} \Big[ p_2 \in I_{c_j} \;\Big|\; |p_1 - p_2| = D \Big]$$
- RGB color space is quantized into $4 \times 4 \times 4 = 64$ discrete color indices.
- Spatial auto-correlograms ($c_i = c_j$) are computed across 8 directions for distances $D \in \{1, 3, 5, 7\}$.
- Feature length: $4 \times 64 = 256$ color features.

#### D. Feature Fusion (Equation 13)
The texture and color vectors are fused via direct vector concatenation:
$$FE = \Big[ \gamma^{(D)}_{c_i, c_j}, \; LBP(RGB) \Big]$$
For $L = 3$: $256 \text{ (color)} + 6,912 \text{ (texture)} = 7,168 \text{ features}$.

---

### 3. Machine Learning Classifiers (`src/models.py`)

Models derived from `ML ALL.pdf` and benchmarked against Table 11 of `Research.pdf`:

1. **Linear Support Vector Machine (One-Against-All / OAA)** *(Proposed Top Model)*:
   - Minimizes the soft margin objective function (Eq. 14):
     $$\min_{w_m, b_m, \xi} \frac{1}{2} \|w_m\|^2 + C \sum_{i=1}^l \xi_i^m$$
     subject to:
     $$(w_m)^T FE_i + b_m \ge 1 - \xi_i^m \quad \text{for } y_i = m$$
     $$(w_m)^T FE_i + b_m \le -1 + \xi_i^m \quad \text{for } y_i \ne m$$
   - Kernel function (Eq. 16): $K(x_i, x) = x_i^T x$.
2. **RBF Kernel SVM**: $K(x_1, x_2) = \exp(-\gamma \|x_1 - x_2\|^2)$ (`ML ALL.pdf` p. 43, 47).
3. **Polynomial Kernel SVM**: $K(x_1, x_2) = (\gamma x_1^T x_2 + r)^d$ (`ML ALL.pdf` p. 38).
4. **Decision Tree (CART)**: Gini index impurity splitting criteria (`ML ALL.pdf` p. 57–77).
5. **Random Forest**: Wisdom of crowds and bootstrap aggregation (bagging) (`ML ALL.pdf` p. 85–92).
6. **AdaBoost**: Adaptive boosting of weak learners with exponential reweighting (`ML ALL.pdf` p. 98).
7. **K-Nearest Neighbors (KNN)**: Euclidean distance proximity search (`ML ALL.pdf` p. 112–114).
8. **Multi-Layer Perceptron (MLP)**: Feedforward architecture with ReLU, backpropagation, and Adam optimizer (`ML ALL.pdf` p. 194–219).
9. **Softmax Regression**: Multinomial cross-entropy classification (`ML ALL.pdf` p. 22–25).

---

### 4. Performance Evaluation Metrics (`src/evaluation.py`)

Equations (17)–(22) from Section 4.2 of `Research.pdf`:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} \times 100\% \tag{Eq. 17}$$

$$\text{Precision} = \frac{TP}{TP + FP} \times 100\% \tag{Eq. 18}$$

$$\text{Sensitivity (Recall)} = \frac{TP}{TP + FN} \times 100\% \tag{Eq. 19}$$

$$\text{F1-Score} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN} \times 100\% \tag{Eq. 20}$$

$$\text{Specificity} = \frac{TN}{TN + FP} \times 100\% \tag{Eq. 21}$$

$$\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}} \times 100\% \tag{Eq. 22}$$

---

## 🚀 How to Run the Prototype

### 1. Training & Comparing Models (`train.py`)

Train models using stratified 3-way dataset partitioning (**70% Training, 10% Validation, 20% Testing**):

```bash
# 1. Benchmark comparison across all models on DS1 (70% train / 10% val / 20% test)
python train.py --dataset DS1 --model all --samples 150

# 2. Train on DS2 (Rice Diseases - 5 classes)
python train.py --dataset DS2 --model all --samples 150

# 3. Train on DS3 (Rice Leaf Diseases Dataset - 3 classes)
python train.py --dataset DS3 --model all --samples 150

# 4. Custom split ratio configuration (default: 0.70 / 0.10 / 0.20)
python train.py --dataset DS1 --train_ratio 0.70 --val_ratio 0.10 --test_ratio 0.20
```

The script automatically saves:
- The best trained model and class metadata to `saved_models/best_model_{dataset}.joblib`
- The comparative performance report (Train, Val, Test Acc) to `reports/model_comparison_{dataset}.csv`

---

### 2. Command-Line Disease Diagnosis (`predict.py`)

Pass any rice leaf image to receive a full diagnostic report:

```bash
# Diagnose a Blast sample
python3 predict.py --image "Rice Leaf Disease Images/Blast/BLAST4_114.jpg"

# Diagnose a Tungro sample from DS2
python3 predict.py --image "Dataset/Test_Data_SVM/tungro/tungro__0_9432.jpg"

# Diagnose a Bacterial Blight sample from DS3
python3 predict.py --image "rice leaf diseases dataset/Bacterialblight/BACTERAILBLIGHT4_132.JPG"
```

**Sample Output:**
```text
======================================================================
          RICE LEAF DISEASE DIAGNOSIS REPORT
======================================================================
Input Image:   Rice Leaf Disease Images/Blast/BLAST4_114.jpg
Model Used:    svm_linear (Trained on DS1)
Diagnosis:     Rice Blast (Magnaporthe oryzae / Pyricularia oryzae)
Confidence:    98.74%
----------------------------------------------------------------------
Class Probabilities:
  Blast              :  98.74%  ███████████████████
  Brownspot          :   0.82%  
  Bacterialblight    :   0.31%  
  Tungro             :   0.13%  
----------------------------------------------------------------------
Symptoms:
  Spindle-shaped or diamond-shaped lesions with gray/white centers and reddish-brown borders.

Causes:
  Fungal pathogen spreading rapidly during prolonged leaf wetness, cool nights, and high nitrogen.

Recommended Agronomic Treatment:
  1. Spray systemic fungicides: Tricyclazole 75 WP (0.6 g/L) or Isoprothiolane 40 EC (1.5 mL/L).
  2. Avoid dense planting to improve air circulation.
  3. Apply balanced NPK fertilizers (do not over-apply nitrogen).
======================================================================
```

---

### 3. Interactive Web Application (`app.py`)

Launch the modern graphical web interface in your browser:

```bash
streamlit run app.py
```

**Web Interface Features:**
- **Disease Diagnosis Tab**: Upload any leaf image or pick from benchmark samples; view confidence score, class probability chart, and agronomic management instructions.
- **Image Processing & Feature Map Inspector**: Inspect side-by-side original vs. Log-Enhanced image ($c=2.0$) and individual Red/Green LBP texture channels.
- **ML Model Benchmark Tab**: View comparison tables and accuracy bar charts for all 9 models across datasets.
- **Research Methodology Tab**: Interactive breakdown of all mathematical equations from `Research.pdf`.
- **Train / Retrain Tab**: 1-click model retraining with customizable spatial levels and sample sizes directly in the GUI.

---

## 📊 Summary of Functions & Operations

| File | Function / Class | Operation & Mathematical Purpose |
| :--- | :--- | :--- |
| `src/preprocessing.py` | `resize_image()` | Unifies image scale to $224 \times 224$ (Eq. 2) |
| `src/preprocessing.py` | `augment_image()` | Applies random translation, rotation, and flips to balance classes (Eq. 3–6) |
| `src/preprocessing.py` | `logarithmic_enhancement()` | Applies $D_{tLog} = c \cdot \ln(1 + D_t/255)$ with $c=2.0$ to normalize illumination (Eq. 7–8) |
| `src/feature_extraction.py`| `compute_lbp_channel()` | Fast vectorized 8-neighbor LBP calculation via bitwise bitshifts (Eq. 11) |
| `src/feature_extraction.py`| `compute_mclbp_block()` | Computes and normalizes 256-bin histograms across R, G, B channels (Eq. 12) |
| `src/feature_extraction.py`| `extract_ml_mclbp()` | Partitions image into $L \times L$ blocks and concatenates block histograms (Eq. 10) |
| `src/feature_extraction.py`| `quantize_colors()` | Quantizes 24-bit RGB space into 64 discrete color indices |
| `src/feature_extraction.py`| `extract_color_correlogram()` | Measures spatial color correlation probabilities at distances $D \in \{1, 3, 5, 7\}$ (Eq. 9) |
| `src/feature_extraction.py`| `extract_fused_features()` | Fuses Color Correlogram and ML-MCLBP into unified feature vector $FE$ (Eq. 13) |
| `src/dataset_loader.py` | `load_dataset()` | Loads DS1, DS2, or DS3, extracts features, and caches to `.npz` for sub-second reuse |
| `src/models.py` | `get_model()` | Builds pipeline with `StandardScaler` and classifier from `ML ALL.pdf` / `Research.pdf` |
| `src/evaluation.py` | `compute_class_metrics()` | Computes TP, FP, FN, TN, Specificity, and MCC per class (Eq. 17–22) |
| `src/evaluation.py` | `evaluate_predictions()` | Computes macro-averaged metrics and multi-class confusion matrix |
| `train.py` | `main()` | CLI training script for single or all models with automated best model export |
| `predict.py` | `predict_disease()` | Preprocesses, extracts fused features, predicts class, and provides agronomic advisory |
| `app.py` | `main()` | Streamlit Web GUI with visual diagnostic reports, feature maps, and model retraining |
