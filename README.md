# Chest X-Ray Pneumonia Detection

Automatic detection of **pneumonia** from chest X-ray images using Deep Learning (CNN + Transfer Learning with ResNet).

## Medical Context

Pneumonia is an infection that inflames the air sacs in the lungs. It kills over 2 million children under 5 every year. Early and accurate diagnosis from chest X-rays is critical — this project explores how Deep Learning can assist radiologists.

## Dataset

**Chest X-Ray Images (Pneumonia)** — Kaggle / Guangzhou Women and Children's Medical Center  
- 5,863 chest X-ray images (JPEG)
- 2 classes: **NORMAL** vs **PNEUMONIA**
- Split: train / val / test already provided
- Download: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

## Project Structure

```
chest-xray-pneumonia/
├── data/
│   ├── raw/          # Original images (not tracked by git)
│   └── processed/    # Resized & normalized images
├── notebooks/
│   ├── 01_exploration.ipynb       # Understanding the data
│   ├── 02_preprocessing.ipynb     # Augmentation & normalization
│   ├── 03_baseline_cnn.ipynb      # Simple CNN from scratch
│   ├── 04_transfer_learning.ipynb # ResNet18 fine-tuning
│   └── 05_evaluation.ipynb        # Metrics, Grad-CAM, error analysis
├── src/
│   ├── data/         # Dataset & DataLoader classes
│   ├── models/       # CNN and ResNet architectures
│   └── utils/        # Metrics, visualization helpers
├── results/
│   ├── figures/      # Plots and Grad-CAM visualizations
│   └── models/       # Saved model weights
├── environment.yml   # Conda environment
└── README.md
```

## Pipeline

```
Raw X-ray images → Preprocessing & Augmentation → CNN / ResNet → Pneumonia classification
                                                              ↓
                                                       Grad-CAM visualization
                                                  (what does the model look at?)
```

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/ValentinDestouches/chest-xray-pneumonia.git
cd chest-xray-pneumonia

# 2. Create the environment
conda env create -f environment.yml
conda activate xray

# 3. Download the dataset on Kaggle and place it in data/raw/

# 4. Launch Jupyter
jupyter notebook
```

## Results

*(To be filled after training)*

| Model | Accuracy | Recall (Pneumonia) | F1-Score |
|-------|----------|--------------------|----------|
| Baseline CNN | - | - | - |
| ResNet18 (transfer learning) | - | - | - |

> **Why Recall?** In medical diagnosis, missing a sick patient (false negative) is more dangerous than a false alarm. We optimize Recall for the PNEUMONIA class.

## References

- Kermany et al. (2018) — *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning* (Cell)
- He et al. (2015) — *Deep Residual Learning for Image Recognition* (ResNet)
