# Casting Defect Detection System

Robustness analysis of an industrial casting product defect detection pipeline against Gaussian noise, spatial distortion, and denoising strategies. This project combines HOG + LBP feature extraction with Support Vector Machine (SVM) classification and a Convolutional Autoencoder for bounding box defect localization.

## Project Identity
- **Course**: Digital Image Processing (PCD)
- **Program**: S1 Data Science, Universitas Negeri Surabaya (Unesa)
- **Team**: 
  - Rizki Piji Fathoni (24031554029)
  - Daffa Ahmad Pangreksa (24031554159)

## Features & Architecture

### Stage 1: SVM Classifier
- **Features**: Histogram of Oriented Gradients (HOG) + Local Binary Patterns (LBP)
- **Kernel**: Radial Basis Function (RBF)
- **Robustness Tests**: Evaluated across 5 testing scenarios, measuring cross-domain resilience (Clean vs. Gaussian Noise vs. Denoised) and spatial robustness (Zoom & Crop augmentations).
- **Performance**: Achieved up to 99.16% accuracy in baseline conditions.

### Stage 2: Convolutional Autoencoder
- **Role**: Defect Localization (Bounding Box Generation)
- **Architecture**: 3-layer Encoder, 16x16 Latent Space Bottleneck, 3-layer Decoder.
- **Mechanism**: Trained strictly on "OK/Normal" samples to reconstruct images. Anomalies (defects) trigger high reconstruction error (MSE), captured via a dynamic 93rd percentile threshold to generate accurate bounding boxes.
- **Performance**: 100% localization success rate across 200 defect sample images.

## Repository Structure
- `029_159_PCD.ipynb`: The main Google Colab notebook containing all training, evaluation, and pipeline extraction scripts.
- `029_159_Final Report PCD.pdf`: Comprehensive final project report detailing methodologies and findings.
- `Dashboard/app.py`: Interactive Streamlit dashboard demonstrating dataset statistics, model architecture, robustness findings, and live inference.
- `dashboard_assets/`: Static image exports, metric data, and visualization assets utilized by the dashboard.
- `PROPOSAL PROJECT PCD_029_159.pdf`: Original project proposal.

## Running the Dashboard

1. Install dependencies:
   ```bash
   pip install -r Dashboard/requirements.txt
   ```
2. Run the application:
   ```bash
   cd Dashboard
   streamlit run app.py
   ```
*(Note: The Live Inference page will automatically fetch and lazy-load the required SVM `.pkl` models directly from GitHub Releases to optimize memory and bypass the 100MB repository file size limit).*
