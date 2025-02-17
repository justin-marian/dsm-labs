# 🦾 ML Notebooks

This repository contains **Jupyter notebooks** covering various deep learning and computer vision topics. Each notebook provides hands-on experience with different techniques and architectures.

---

## 📚 Notebooks

### 0️⃣ Intro Labs [![▶️ Open in Google Colab: Intro Lab 1](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cosmaadrian/ml-environment/blob/master/EP_Plotting_I.ipynb) [![▶️ Open in Google Colab: Intro Lab 2](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cosmaadrian/ml-environment/blob/master/EP_Plotting_II.ipynb) [![▶️ Open in Google Colab: Intro Lab 3](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AndyCatruna/DSM/blob/main/Lab_01_Neural_Networks_in_PyTorch.ipynb) 🏁

- 🔰 Introduction to Machine Learning concepts:
  - `Python`
  - `NumPy`
  - `Pandas`
  - Data visualization
  - Basic `supervised` learning
  - Basic `semi-supervised` learning

### 1️⃣ Convolutional Neural Networks (CNNs) [![▶️ Open in Google Colab: CNNs](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AndyCatruna/DSM/blob/main/Lab_02_Convolutional_Neural_Networks.ipynb) 🖼️

- 📌 Introduction to `Convolutional Neural Networks (CNNs)` for image classification.  
- 🛠️ Practical implementation using deep learning frameworks.

### 2️⃣ CNN Architectures [![▶️ Open in Google Colab: CNN Architectures](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AndyCatruna/DSM/blob/main/Lab_03_CNN_Architectures.ipynb) 🏗️

- 🔍 Exploration of various `CNN architectures`.  
- 🎛️ Understanding `hyperparameter tuning` and architectural choices.  
- 📦 Utilizing `pre-trained models` for improved performance.

### 3️⃣ Encoder-Decoder Transformers [![▶️ Open in Google Colab: Encoder-Decoder Transformers](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AndyCatruna/DSM/blob/main/Lab_04_Transformer_Encoder.ipynb) 🤖

- 📖 Implementation of the `Transformer model` for deep learning tasks.  
- 🧠 Understanding `self-attention mechanisms`.

### 4️⃣ Object Detection [![▶️ Open in Google Colab: Object Detection](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AndyCatruna/DSM/blob/main/Lab_05b_Detection.ipynb) 🎯

- 🔍 `Training and evaluating object detection models`.  
- 🏷️ Understanding `bounding boxes` and `class predictions`.  
- 🛠️ Recommended external resources: **[Ultralytics](https://github.com/ultralytics/ultralytics)** for efficient detection models.

### 5️⃣ Semantic Segmentation [![▶️ Open in Google Colab: Semantic Segmentation](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AndyCatruna/DSM/blob/main/Lab_05a_Segmentation.ipynb) 🏞️

- 🎨 Assigning class labels to each pixel in an image.  
- 🚗 Applications in medical imaging, autonomous driving, and more.

### 6️⃣ Semi-Supervised Learning with FixMatch [![▶️ Open in Google Colab: Semi-Supervised Learning (FixMatch](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cosmaadrian/ml-environment/blob/master/DSM_Lab_FixMatch.ipynb) 🏆

- 🏗️ Exploring `FixMatch`, a method for `semi-supervised learning`.  
- 🖼️ Training on a fraction of `CIFAR-10` images.

### 7️⃣ Self-Supervised Learning with SimCLR [![▶️ Open in Google Colab: Self-Supervised Learning (SimCLR)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cosmaadrian/ml-environment/blob/master/DSM_SimCLR.ipynb) 🧩

- 🔄 Implementing `SimCLR` for contrastive learning.  
- 📸 Pretraining on `CIFAR-10` and fine-tuning on `SVHN`.

---

## ⭐ Getting Started

To run these notebooks locally, install the necessary dependencies:

```bash
pip install torch torchvision timm jupyter matplotlib seaborn tqdm
```

Alternatively, use Google Colab links provided in each section to run them directly in the cloud. ☁️

---

## 📄 Research Papers

Some of the research papers referenced in the notebooks are available in the `pdfs` folder. These papers provide theoretical background and additional insights into the concepts explored in the notebooks. These papers complement the practical implementations in this repository and provide additional theoretical insights into the methods used.

### 📚 Available Papers

- **[FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence](https://arxiv.org/abs/2001.07685)**  
  - 📄 Found in [`pdfs/Semi-Supervised Learning FixMatch.pdf`](pdfs/Semi-Supervised%20Learning%20FixMatch.pdf)

- **[SimCLR: A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)**  
  - 📄 Found in [`pdfs/CLR.pdf`](pdfs/CLR.pdf)

- **[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)**  
  - 📄 Found in [`pdfs/BatchNormalization.pdf`](pdfs/BatchNormalization.pdf)

- **[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)**  
  - 📄 Found in [`pdfs/MobileNet.pdf`](pdfs/MobileNet.pdf)

- **[MobileNetV3: Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)**  
  - 📄 Found in [`pdfs/MobileNetV3.pdf`](pdfs/MobileNetV3.pdf)

- **[Deep Residual Learning for Image Recognition (ResNet)](https://arxiv.org/abs/1512.03385)**  
  - 📄 Found in [`pdfs/ResNet.pdf`](pdfs/ResNet.pdf)

- **[Identifying Mislabeled Data using the Area Under the Margin Ranking](https://arxiv.org/abs/2001.10528)**  
  - 📄 Found in [`pdfs/Mislabeled Data AUC.pdf`](pdfs/Mislabeled%20Data%20AUC.pdf)

- **[Mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)**  
  - 📄 Found in [`pdfs/RiskMinimization.pdf`](pdfs/RiskMinimization.pdf)

- **[MarginMatch: Improving Semi-Supervised Learning with Pseudo-Margins](https://arxiv.org/abs/2308.09037)**  
  - 📄 Found in [`pdfs/Semi-Supervised Learning Pseudo-Margins.pdf`](pdfs/Semi-Supervised%20Learning%20Pseudo-Margins.pdf)

- **[SqueezeNet: AlexNet-Level Accuracy with 50x Fewer Parameters](https://arxiv.org/abs/1602.07360)**  
  - 📄 Found in [`pdfs/SqueezeNet.pdf`](pdfs/SqueezeNet.pdf)

- **[SSD: Single Shot MultiBox Detector for Object Detection](https://arxiv.org/abs/1512.02325)**  
  - 📄 Found in [`pdfs/SSD.pdf`](pdfs/SSD.pdf)
