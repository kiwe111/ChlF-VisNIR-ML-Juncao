# ChlF-VisNIR-ML-Juncao

---

## 🧠 Code Modules

### `cluster_analysis.py`

> Automatically determines the optimal number of clusters for stress classification and visualizes results using t-SNE.

- 📥 **Input:** Physiological data (e.g., OJIP, SPAD, SOD)
- 🔧 **Methods:**
  - `t-SNE` for nonlinear dimensionality reduction
  - `KMeans` clustering
  - Cluster evaluation via **Silhouette Score** & **Calinski-Harabasz Index**
  - Cluster visualization with confidence ellipses
- 📤 **Output files:**
  - `tsne_cluster_data.csv`
  - `silhouette_data.csv`
  - `sse_ch_data.csv`
  - `confidence_ellipses.csv`

### `shap.py`

> Classifies clusters using XGBoost and interprets feature importance via SHAP values.

- 📥 **Input:** Cluster labels + selected physiological features
- 🔧 **Methods:**
  - `XGBoost` multi-class classifier
  - `SHAP` (Shapley Additive Explanations)
  - `Pearson` correlation analysis
- 📤 **Output:**
  - SHAP plots & summary
  - Confusion matrix and classification metrics
  - Feature importance CSVs/Excels:
    - `shap_contribution_sorted.xlsx`
    - `shap_contribution_centers.xlsx`
    - `pearson_correlation.csv`

---

## 🧪 Traditional ML & DL Models (MATLAB)

For full spectral modeling and Fv/Fm prediction, we provide MATLAB implementations of traditional machine learning and deep learning models.

### 🔎 Stress Severity Classification

Implemented in MATLAB R2022b:

- 📊 Models: `LDA`, `PLS-DA`, `RF`, `LS-SVM`, `BPNN`, `ELM`, `1D-CNN`
- 🔍 Feature selection: `CARS`, `Random Frog`, `Kennard-Stone`
- 📂 Files to run:
  - `ks.m`, `CARS_Feature selection.m`, `Random frog_Feature selection.m`
  - `*_Stress severity_Identifying.m`

### 📈 Fv/Fm Regression Models

- 📊 Models: `MLR`, `PLSR`, `RF`, `ELM`, `LS-SVR`, `BPNN`, `1D-CNN`
- 📂 Files to run:
  - `MLR_FvFm_Predicting.m`, `PLSR_FvFm_Predicting.m`, ...
  - All follow similar naming structure.

---

## 💾 Raw Data

- `Date-Stress severity-Identifying.xlsx`: Physiological feature matrix + labels
- `Date-FvFm-Predicting.xlsx`: Vis-NIR spectra + Fv/Fm values
- `Date-Chlorophyll fluorescence parameters.xlsx`: Raw OJIP and SPAD signals

---

## 🛠 Environment

- **Python Version:** 3.9+
- **Key Libraries:** `xgboost`, `shap`, `sklearn`, `matplotlib`, `seaborn`, `pandas`
- **MATLAB Version:** R2022b
- **System:** Windows 11, Intel Core i7-12650H (16 CPUs, 2.3 GHz)

---
