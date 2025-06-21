# ChlF-VisNIR-ML-Juncao

This repository contains the code for the paper:

**"A Rapid and Nondestructive Method for Assessing Low-Temperature Stress Severity and Predicting Fv/Fm in Juncao Seedlings Using Spectroscopic Analysis and Machine Learning"**

## 🌱 Project Overview

Low-temperature stress is a critical environmental factor limiting the growth of **Juncao (Cenchrus fungigraminus)**, a tropical C4 plant with poor cold tolerance. This project proposes a **rapid, nondestructive** method to classify stress severity and predict the chlorophyll fluorescence parameter **Fv/Fm** using:

- 🌿 **Chlorophyll a fluorescence (ChlF)**
- 🌈 **Visible-Near Infrared (Vis-NIR) spectroscopy**
- 🤖 **Machine learning algorithms**

---

## 📊 Code Modules

### `cluster_analysis.py`

> **Function:** Automatically determine the optimal number of clusters for stress severity classification, and visualize K-Means clustering using t-SNE.

- Input: Preprocessed physiological data (e.g., OJIP, SOD, SPAD, etc.)
- Techniques used:
  - **t-SNE** dimensionality reduction
  - **KMeans clustering**
  - **Silhouette score**, **Calinski-Harabasz Index**
  - Cluster visualization with confidence ellipses
- Output:
  - Cluster assignment (`tsne_cluster_data.csv`)
  - Silhouette metrics (`silhouette_data.csv`)
  - Optimal cluster metrics (`sse_ch_data.csv`)
  - Ellipse parameters for confidence intervals

### `shap_analysis.py`

> **Function:** Train an XGBoost model to classify clusters and apply SHAP to interpret which features contributed most to classification.

- Input: Cluster labels + physiological features
- Techniques used:
  - **XGBoost multi-class classification**
  - **SHAP (Shapley Additive Explanations)**
  - **Pearson correlation analysis**
- Output:
  - SHAP bar plots and summary plots
  - Confusion matrix and classification report
  - CSV/Excel of feature importance and SHAP contributions

---

## 📂 Example File Paths (for Linux)

```python
file_path = "/home/xqw/0326.xlsx"  # Replace with your data path
