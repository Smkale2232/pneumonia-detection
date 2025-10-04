# Pneumonia Detection with TensorFlow

A deep learning pipeline for detecting pneumonia from chest X-ray images using TensorFlow and Keras. The project covers model training, evaluation, and prediction with visualizations such as training curves and confusion matrices.

---

## 📌 Version Information

* **Python**: 3.8+
* **TensorFlow**: 2.11+
* **Keras**: Included in TensorFlow
* **NumPy**: 1.21+
* **Pandas**: 1.3+
* **Matplotlib**: 3.5+
* **Scikit-learn**: 1.0+
* **OpenCV**: 4.5+

(Refer to `requirements.txt` or `requirements_project.txt` for the exact versions used.)

---

## 📁 Project Structure

```
.
├── config/  
│   └── (configuration files and parameters)  
├── data/  
│   └── chest_xray/  
│       ├── train/  
│       ├── val/  
│       └── test/  
├── results/  
│   ├── Confusion Matrix.png  
│   └── Training & Validation Metrics.png  
├── src/  
│   ├── train_model.py  
│   ├── evaluate_model.py  
│   └── main.py  
├── environment.yml  
├── requirements.txt  
├── requirements_project.txt  
├── run_project.bat  
├── .gitattributes  
├── .gitignore  
└── README.md
```

---

## ⚙️ Installation Guide

## 📥 Kaggle Dataset

You can use the **Chest X-Ray Images (Pneumonia)** dataset from Kaggle, which is a commonly used benchmark.  
Kaggle dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

The dataset is organized into three splits (train, test, val), with subfolders for `NORMAL` and `PNEUMONIA` classes.


### 1. Clone the Repository

```bash
git clone https://github.com/Smkale2232/pneumonia-detection.git
cd pneumonia-detection
```

### 2. Set Up Environment

#### Option A: Using Conda

```bash
conda env create -f environment.yml
conda activate pneumonia-detection
```

#### Option B: Using Pip

```bash
pip install -r requirements.txt
```

### 3. Dataset Preparation

Ensure the dataset is placed under `data/chest_xray/` with the following structure:

```
data/chest_xray/
    train/
    val/
    test/
```

Each folder should contain `NORMAL/` and `PNEUMONIA/` subdirectories with images.

---

## 🚀 Usage

### Training

```bash
python src/train_model.py
```

or:

```bash
python src/main.py --train
```

### Evaluation

```bash
python src/evaluate_model.py
```

or:

```bash
python src/main.py --eval
```

---

## 📊 Results

* **Training & Validation Metrics.png** — Accuracy and loss curves

<img src="results/Training & Validation Metrics.png" alt="Confusion Matrix" width="500"/>

* **Confusion Matrix.png** — Classification performance on test set

<img src="results/Confusion Matrix.png" alt="Confusion Matrix" width="500"/>

* **Model Checkpoints** — Saved weights for reuse and inference

---

## 🔧 Configuration

Parameters such as batch size, learning rate, number of epochs, and augmentation options can be adjusted in `config/` or inside the training script.

---

## 🧪 Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

## ✅ Workflow

1. Install dependencies
2. Prepare dataset in the correct structure
3. Train the model
4. Evaluate the trained model
5. Review results in `results/`
6. Reuse the saved model for predictions

---

## 🛠️ Best Practices

* Apply normalization to input images
* Use data augmentation to improve robustness
* Use early stopping and model checkpoints
* Consider transfer learning with pretrained models for better performance

---

## 🤝 Contributing

1. Fork the repository
2. Create a new branch
3. Commit and push your changes
4. Open a pull request

Follow PEP8 coding style and document your changes.

---

## 📝 License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute it with proper attribution.

---

## 📂 Additional Notes

* `run_project.bat` allows quick execution on Windows
* `.gitignore` excludes unnecessary files such as datasets and models
* `.gitattributes` ensures consistent file handling across platforms

---
