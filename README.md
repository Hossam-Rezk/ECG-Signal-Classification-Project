# ECG-Signal-Classification-Project

## Overview
This project involves the classification of ECG signals to detect Normal and LBBB (Left Bundle Branch Block) heart conditions using machine learning models, specifically **K-Nearest Neighbors (KNN)** and **Support Vector Machines (SVM)**. The project includes data preprocessing, wavelet feature extraction, model training, evaluation, and a user-friendly GUI for ECG signal classification.

## Features
1. **Data Loading & Preprocessing**:
   - Load ECG signals from text files.
   - Pad signals to ensure uniform length.
   - Preprocess signals by removing the DC component, applying a bandpass filter, and normalizing.

2. **Feature Extraction**:
   - Use **Daubechies wavelets (`db6`)** with level 6 decomposition to extract features.
   - Extract statistical features (mean, standard deviation, energy) from wavelet coefficients.

3. **Classification Models**:
   - **KNN** with Euclidean distance and various `n_neighbors` values.
   - **SVM** with Linear and RBF kernels.

4. **Performance Metrics**:
   - Accuracy on training and test sets.
   - Cross-validation scores.
   - Confusion matrix and classification reports.

5. **GUI Application**:
   - Load ECG signals from files or enter them manually.
   - Predict Normal/LBBB using KNN or SVM models.
   - Visualize ECG signals with annotated P, QRS, and T waves.

## Project Structure
```plaintext
├── data
│   ├── Normal_Train.txt
│   ├── LBBB_Train.txt
│   ├── Normal_Test.txt
│   ├── LBBB_Test.txt
├── models
│   ├── knn_model.pkl
│   ├── svm_model.pkl
├── ecg_classification.py
├── requirements.txt
├── README.md
└── logs
    └── ecg_classification.log
```

## Requirements
- Python 3.8+
- Libraries:
  - NumPy
  - Matplotlib
  - Scikit-learn
  - PyWavelets
  - Joblib
  - SciPy
  - Tkinter (built-in with Python)

Install dependencies using:
```bash
pip install -r requirements.txt
```

## How to Run
### 1. Train Models
To train the KNN and SVM models and evaluate their performance:
```bash
python ecg_classification.py --train
```
This will:
- Preprocess the data.
- Train KNN and SVM models.
- Save the trained models in the `models/` directory.

### 2. Launch the GUI
To launch the GUI for signal classification:
```bash
python ecg_classification.py --gui
```

### 3. Dataset Preparation
Ensure the dataset files are located in the `data/` directory with the following format:
- Each line represents a single ECG signal.
- Samples in a line are separated by the `|` delimiter.

## Performance
### **KNN**:
| n_neighbors | Accuracy (%) |
|-------------|--------------|
| 3           | 89.06        |
| 5           | 93.27        |
| 7           | 95.29        |
| 9           | **96.80**    |

### **SVM**:
| Kernel  | Accuracy (%) |
|---------|--------------|
| RBF     | 89.06        |
| Linear  | **98.65**    |

### Summary:
- **Best KNN Configuration:** `n_neighbors = 9` with 96.80% accuracy.
- **Best SVM Configuration:** Linear kernel with 98.65% accuracy.

## GUI Features
- **File Upload:** Load ECG signals from text files.
- **Manual Entry:** Enter ECG samples manually.
- **Prediction:** Predict Normal or LBBB using KNN or SVM.
- **Visualization:** Plot ECG signals with labeled wave components (P, QRS, T).

## Logs
All logs, including training and evaluation metrics, are saved in `logs/ecg_classification.log`.

## Future Improvements
- Test on additional datasets to validate generalization.
- Implement real-time ECG signal processing.
- Add support for additional classification models.
- Enhance GUI with real-time signal visualization.

## License
This project is licensed under the MIT License.
