# HYFORMER: A Vision Transformer AI Model for Identifying Tropical Tree Species Using Hyperspectral Images of Wood

This repository contains the official implementation of **HyFormer**, a lightweight and efficient **Vision Transformer (ViT)** model tailored for **spectral–spatial classification of tropical wood species** using hyperspectral images. HyFormer is designed to capture the fine-grained spectral patterns found in natural wood materials, enabling accurate species-level identification essential for forestry management, conservation, and anti-illegal logging efforts.

---

## 🌳 Overview

HyFormer introduces:

- A **spectral tokenizer** for compact band-to-token embedding  
- **Hybrid Spectral–Spatial Transformer encoder**  
- **Lightweight architecture** with only ~0.3M parameters  
- Top-tier performance on 9 tropical species from the HyperWood dataset  
- End-to-end pipeline: preprocessing → training → evaluation  

---

## 🌈 Spectral Signatures of the Tree Species

Average spectral signatures for all 9 tropical species used in this work.

<img width="2970" height="1770" alt="image" src="https://github.com/user-attachments/assets/abab4cf6-83ef-4ff4-b070-86c67d68b8cc" />

---

## 🧪 Hyperspectral Preprocessing Pipeline

Our preprocessing workflow includes:

<img width="975" height="450" alt="image" src="https://github.com/user-attachments/assets/897d252d-0914-4750-bacc-d03ee8a14ce7" />

---

## 🧠 HyFormer Architecture

HyFormer consists of:

<img width="599" height="280" alt="Picture1" src="https://github.com/user-attachments/assets/d2dacc90-9f1e-4316-ba51-0288be4bc8d9" />

---

# 📊 Experimental Results

### **Per-Class and Overall Classification Results**

| **Class** | 2DCNN | 3DCNN | SSFormer | SSLinFormer | **HyFormer** |
|----------|-------|--------|-----------|--------------|--------------|
| African Ipe | 95.89 | 96.76 | 96.56 | 72.05 | **95.57** |
| African Padauk | 98.06 | 98.29 | 97.44 | 72.57 | **96.24** |
| African Teak | 97.68 | 98.90 | 92.30 | 37.74 | **98.93** |
| Iroko | 98.10 | 99.28 | 99.24 | 79.26 | **99.37** |
| Obeche | 96.90 | 97.40 | 97.68 | 77.30 | **98.05** |
| Ovangkol | 95.95 | 98.10 | 96.82 | 72.73 | **96.72** |
| Merbau | 97.94 | 98.78 | 98.77 | 93.58 | **98.43** |
| Sapele | 97.38 | 98.56 | 97.28 | 86.32 | **98.61** |
| Teak | 96.30 | 96.80 | 96.18 | 88.56 | **97.14** |

### **Overall Metrics**

| Metric | 2DCNN | 3DCNN | SSFormer | SSLinFormer | **HyFormer** |
|--------|--------|--------|-----------|--------------|--------------|
| OA | 97.05 | 97.98 | 96.99 | 76.98 | **97.55** |
| κ (Kappa) | 96.67 | 97.73 | 96.61 | 74.02 | **97.23** |
| Train Time (s) | **183.81** | 219.83 | 344.40 | 367.65 | 369.23 |
| Test Time (s) | **26.37** | 30.34 | 48.30 | 49.29 | 55.79 |
| Parameters | 1,346,617 | 15,838,457 | **606,473** | **134,041** | **305,289** |

---

## 📦 Installation

HyFormer was developed and tested with:

- **Python 3.10.4**
- **TensorFlow 2.15.1**
- **NumPy 1.24.1**

---

## 🚀 Training the Model

### Step 1 — Prepare dataset structure

```
data/
│── African_Ipe/
│── African_Padauk/
│── ...
```

### Step 2 — Preprocessing

```bash
prepocessing.ipynb
```

---

## 🧪 Training & Evaluation

```bash
models.ipynb
```

---

## 📚 Citation

If you use HyFormer in your research, please cite:

```
@article{hyformer2025,
  title={HYFORMER: A Vision Transformer AI Model for Identifying Tropical Tree Species Using Hyperspectral Images of Wood},
  author={Butt, Muhammad Hassaan Farooq and others},
  journal={Submitted},
  year={2025}
}
```

---

## 📬 Contact

For questions or collaboration:

**Muhammad Hassaan Farooq Butt**  
Email: hassaanbutt67@gmail.com  

---




