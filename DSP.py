import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import pywt
import joblib
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from scipy.signal import butter, lfilter
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import sys
import os
import random

# --------------------------- #
#        Setup Logging         #
# --------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ecg_classification.log", mode='w')
    ]
)

# --------------------------- #
#    Set Random Seeds          #
# --------------------------- #

random.seed(42)
np.random.seed(42)

# --------------------------- #
#      Constants/Paths        #
# --------------------------- #

NORMAL_TRAIN_FILE = "Normal_Train.txt"
LBBB_TRAIN_FILE = "LBBB_Train.txt"
NORMAL_TEST_FILE = "Normal_Test.txt"
LBBB_TEST_FILE = "LBBB_Test.txt"

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
KNN_MODEL_PATH = os.path.join(MODEL_DIR, "knn_model.pkl")
SVM_MODEL_PATH = os.path.join(MODEL_DIR, "svm_model.pkl")

# --------------------------- #
#     Data Loading Module     #
# --------------------------- #

def load_ecg_file(file_path, delimiter='|'):
    """
    Loads ECG data from a text file.
    Each line represents a separate ECG signal with samples separated by the specified delimiter.
    """
    data = []
    if not os.path.isfile(file_path):
        logging.error(f"File not found: {file_path}")
        return np.array(data, dtype=object)
    
    with open(file_path, 'r') as file:
        for line_num, line in enumerate(file, start=1):
            values = line.strip().split(delimiter)
            try:
                row = [float(value) for value in values if value.strip()]
                data.append(row)
            except ValueError:
                logging.warning(f"Skipping invalid line {line_num} in {file_path}: {line.strip()}")
    logging.info(f"Loaded {len(data)} signals from {file_path}")
    return np.array(data, dtype=object)

def pad_data(data):
    """
    Pads all ECG signals to the length of the longest signal in the dataset.
    Pads with zeros at the end of shorter signals.
    """
    if len(data) == 0:
        logging.error("Empty dataset provided to pad_data.")
        return np.array([])
    max_len = max(len(row) for row in data)
    padded = np.array([np.pad(row, (0, max_len - len(row)), constant_values=0) for row in data])
    logging.info(f"Padded data to shape: {padded.shape}")
    return padded

# --------------------------- #
#    Preprocessing Module     #
# --------------------------- #

def remove_dc_time_domain(signal):
    """
    Remove DC component by subtracting the mean from each sample.
    """
    if signal is None or len(signal) == 0:
        raise ValueError("Signal is empty or not provided.")
    mean_val = np.mean(signal)
    return signal - mean_val

def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Design a Butterworth bandpass filter.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    if low >= high or low <= 0 or high >= 1:
        raise ValueError("Invalid bandpass filter parameters.")
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=0.5, highcut=40, fs=360, order=5):
    """
    Apply a bandpass filter to a 1D signal.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered_data = lfilter(b, a, data)
    return filtered_data

def normalize_signal(signal):
    """
    Normalize a 1D signal to the range [0, 1].
    """
    signal_min = np.min(signal)
    signal_max = np.max(signal)
    if signal_max - signal_min == 0:
        return np.zeros_like(signal)
    return (signal - signal_min) / (signal_max - signal_min)

def preprocess_signal(signal, fs=360, lowcut=0.5, highcut=40):
    """
    Complete preprocessing pipeline on a single signal:
    1) Remove DC
    2) Bandpass Filter
    3) Normalize
    """
    dc_removed = remove_dc_time_domain(signal)
    filtered = bandpass_filter(dc_removed, lowcut=lowcut, highcut=highcut, fs=fs)
    normalized = normalize_signal(filtered)
    return normalized

def preprocess_dataset(data):
    """
    Apply the preprocess_signal() to all rows in a 2D array.
    """
    if len(data) == 0:
        logging.error("Empty dataset provided to preprocess_dataset.")
        return np.array([])
    processed = np.array([preprocess_signal(row) for row in data])
    logging.info(f"Preprocessed dataset shape: {processed.shape}")
    return processed

# --------------------------- #
#   Wave Labeling & Visual    #
# --------------------------- #

def label_ecg_waves(ecg_signal, fs=360):
    """
    Naive approach to detect R-peaks and approximate Q,S,P,T.
    Returns dict with keys: R_peaks, Q_points, S_points, P_waves, T_waves
    """
    if len(ecg_signal) == 0:
        return {
            'R_peaks': [],
            'Q_points': [],
            'S_points': [],
            'P_waves': [],
            'T_waves': []
        }
    squared = ecg_signal ** 2
    threshold = 0.4 * np.max(squared)
    R_peaks = []
    Q_points = []
    S_points = []
    P_waves = []
    T_waves = []

    i = 0
    refractory = int(0.2 * fs)
    while i < len(squared):
        if squared[i] > threshold:
            peak_region_end = min(i + refractory, len(squared))
            peak_idx = i + np.argmax(squared[i:peak_region_end])
            R_peaks.append(peak_idx)

            # approximate Q,S ~ 5 samples around R
            q_idx = max(0, peak_idx - 5)
            s_idx = min(len(ecg_signal) - 1, peak_idx + 5)
            Q_points.append(q_idx)
            S_points.append(s_idx)

            # P wave ~200 ms -> 200 ms before R
            p_start = peak_idx - int(0.2 * fs)
            p_end   = peak_idx - int(0.05 * fs)
            p_start = max(0, p_start)
            p_end = max(0, p_end)
            P_waves.append((p_start, p_end))

            # T wave ~300 ms after R
            t_start = peak_idx + int(0.05 * fs)
            t_end   = peak_idx + int(0.3 * fs)
            t_start = min(t_start, len(ecg_signal)-1)
            t_end = min(t_end, len(ecg_signal)-1)
            T_waves.append((t_start, t_end))

            i = peak_region_end
        else:
            i += 1

    return {
        'R_peaks': R_peaks,
        'Q_points': Q_points,
        'S_points': S_points,
        'P_waves': P_waves,
        'T_waves': T_waves
    }

def visualize_ecg_waves(ecg_signal, wave_labels, fs=360):
    """
    Plot ECG with vertical lines for Q,R,S and shaded spans for P,T.
    """
    if len(ecg_signal) == 0:
        logging.warning("Empty ECG signal provided for visualization.")
        return

    plt.figure(figsize=(12, 4))
    plt.plot(ecg_signal, label='ECG', color='blue')

    for r_idx in wave_labels['R_peaks']:
        plt.axvline(r_idx, color='red', linestyle='--', alpha=0.7)
    for q_idx in wave_labels['Q_points']:
        plt.axvline(q_idx, color='green', linestyle=':', alpha=0.7)
    for s_idx in wave_labels['S_points']:
        plt.axvline(s_idx, color='green', linestyle=':', alpha=0.7)

    for (p_start, p_end) in wave_labels['P_waves']:
        plt.axvspan(p_start, p_end, color='yellow', alpha=0.2)
    for (t_start, t_end) in wave_labels['T_waves']:
        plt.axvspan(t_start, t_end, color='orange', alpha=0.2)

    plt.title("Labeled ECG Waves (Q,R,S,P,T)")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend(["ECG Signal", "R-peaks", "Q/S", "P wave region", "T wave region"])
    plt.tight_layout()
    plt.show()

# --------------------------- #
#   Wavelet Feature Module    #
# --------------------------- #

def wavelet_decompose(signal, wavelet='db6', level=6):
    """
    Decompose a signal using wavelet transform.
    """
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
    return coeffs

def extract_features_from_coeffs(coeffs):
    """
    Extract simple statistical features from wavelet coefficients:
    For each level of coefficients: mean, std, and energy (sum of squares).
    """
    features = []
    for coef in coeffs:
        mean_val = np.mean(coef)
        std_val  = np.std(coef)
        energy   = np.sum(np.square(coef))
        features.extend([mean_val, std_val, energy])
    return np.array(features)

def extract_features_dataset(data):
    """
    Apply wavelet decomposition + feature extraction to each signal in a 2D dataset.
    """
    if len(data) == 0:
        logging.error("Empty dataset provided to extract_features_dataset.")
        return np.array([])
    features_list = []
    for idx, signal in enumerate(data):
        try:
            coeffs = wavelet_decompose(signal)
            feat = extract_features_from_coeffs(coeffs)
            features_list.append(feat)
        except Exception as e:
            logging.warning(f"Feature extraction failed for signal {idx}: {e}")
    features_array = np.array(features_list)
    logging.info(f"Extracted features shape: {features_array.shape}")
    return features_array

# --------------------------- #
#    Model Training Module    #
# --------------------------- #

def plot_model_accuracies(knn_acc, svm_acc):
    """
    Plots a comparison of KNN and SVM model accuracies on the test set.
    """
    models = ['KNN', 'SVM']
    accuracies = [knn_acc, svm_acc]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(models, accuracies, color=['skyblue', 'lightgreen'])
    plt.title("Model Accuracy Comparison (Test Set)")
    plt.ylabel("Accuracy")
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01,
                 f"{acc*100:.2f}%",
                 ha='center', va='bottom', fontsize=10)
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.show()

def train_and_evaluate():
    """
    Trains KNN and SVM classifiers on the ECG dataset and evaluates their performance.
    Saves the trained models for later use in the GUI.
    """
    try:
        # 1) Load data
        normal_train_raw = load_ecg_file(NORMAL_TRAIN_FILE)
        lbbb_train_raw   = load_ecg_file(LBBB_TRAIN_FILE)
        normal_test_raw  = load_ecg_file(NORMAL_TEST_FILE)
        lbbb_test_raw    = load_ecg_file(LBBB_TEST_FILE)

        # 2) Pad
        padded_normal_train = pad_data(normal_train_raw)
        padded_lbbb_train   = pad_data(lbbb_train_raw)
        padded_normal_test  = pad_data(normal_test_raw)
        padded_lbbb_test    = pad_data(lbbb_test_raw)

        # Check if data is loaded correctly
        if (len(padded_normal_train) == 0 or len(padded_lbbb_train) == 0 or
            len(padded_normal_test) == 0 or len(padded_lbbb_test) == 0):
            raise ValueError("One or more datasets are empty after padding.")

        logging.info(f"Padded Normal Train Shape: {padded_normal_train.shape}")
        logging.info(f"Padded LBBB Train Shape: {padded_lbbb_train.shape}")
        logging.info(f"Padded Normal Test Shape: {padded_normal_test.shape}")
        logging.info(f"Padded LBBB Test Shape: {padded_lbbb_test.shape}")

        # 3) Visualize raw signals
        visualize_raw_signals(padded_normal_train, padded_lbbb_train)

        # 4) Preprocess
        processed_normal_train = preprocess_dataset(padded_normal_train)
        processed_lbbb_train   = preprocess_dataset(padded_lbbb_train)
        processed_normal_test  = preprocess_dataset(padded_normal_test)
        processed_lbbb_test    = preprocess_dataset(padded_lbbb_test)

        # 5) Visualize processed
        visualize_processed_signals(processed_normal_train, processed_lbbb_train)

        # 6) Extract wavelet features
        features_normal_train = extract_features_dataset(processed_normal_train)
        features_lbbb_train   = extract_features_dataset(processed_lbbb_train)
        features_normal_test  = extract_features_dataset(processed_normal_test)
        features_lbbb_test    = extract_features_dataset(processed_lbbb_test)  # Corrected line

        # Combine
        X_train = np.vstack((features_normal_train, features_lbbb_train))
        y_train = np.hstack((np.zeros(len(features_normal_train)), np.ones(len(features_lbbb_train))))
        X_test = np.vstack((features_normal_test, features_lbbb_test))
        y_test = np.hstack((np.zeros(len(features_normal_test)), np.ones(len(features_lbbb_test))))

        logging.info(f"Training Feature Matrix Shape: {X_train.shape}")
        logging.info(f"Training Labels Shape: {y_train.shape}")
        logging.info(f"Testing Feature Matrix Shape: {X_test.shape}")
        logging.info(f"Testing Labels Shape: {y_test.shape}")

        # 7) Train/val split
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
        )
        logging.info(f"X_train_split: {X_train_split.shape}, y_train_split: {y_train_split.shape}")
        logging.info(f"X_val_split: {X_val_split.shape}, y_val_split: {y_val_split.shape}")

        # --- KNN ---
        knn = KNeighborsClassifier(n_neighbors=9)
        knn.fit(X_train_split, y_train_split)
        y_val_pred_knn = knn.predict(X_val_split)
        knn_val_acc = accuracy_score(y_val_split, y_val_pred_knn)
        logging.info(f"KNN Validation Accuracy: {knn_val_acc * 100:.2f}%")
        logging.info("KNN Classification Report:")
        logging.info("\n" + classification_report(y_val_split, y_val_pred_knn))

        # --- SVM ---
        svm_model = SVC(kernel='linear', probability=True, random_state=42)
        svm_model.fit(X_train_split, y_train_split)
        y_val_pred_svm = svm_model.predict(X_val_split)
        svm_val_acc = accuracy_score(y_val_split, y_val_pred_svm)
        logging.info(f"SVM Validation Accuracy: {svm_val_acc * 100:.2f}%")
        logging.info("SVM Classification Report:")
        logging.info("\n" + classification_report(y_val_split, y_val_pred_svm))

        # Evaluate on test
        # Evaluate KNN on training data
        y_train_pred_knn = knn.predict(X_train)
        knn_train_acc = accuracy_score(y_train, y_train_pred_knn)
        logging.info(f"KNN Training Accuracy: {knn_train_acc * 100:.2f}%")

        # Evaluate KNN on test data
        y_test_pred_knn = knn.predict(X_test)
        knn_test_acc = accuracy_score(y_test, y_test_pred_knn)
        logging.info(f"KNN Test Accuracy: {knn_test_acc * 100:.2f}%")

        # Evaluate SVM on training data
        y_train_pred_svm = svm_model.predict(X_train)
        svm_train_acc = accuracy_score(y_train, y_train_pred_svm)
        logging.info(f"SVM Training Accuracy: {svm_train_acc * 100:.2f}%")

        # Evaluate SVM on test data
        y_test_pred_svm = svm_model.predict(X_test)
        svm_test_acc = accuracy_score(y_test, y_test_pred_svm)
        logging.info(f"SVM Test Accuracy: {svm_test_acc * 100:.2f}%")

        # Confusion matrix for SVM
        logging.info("SVM Test Classification Report:")
        logging.info("\n" + classification_report(y_test, y_test_pred_svm, target_names=["Normal", "LBBB"]))
        cm = confusion_matrix(y_test, y_test_pred_svm)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "LBBB"])
        disp.plot()
        plt.title("SVM Confusion Matrix on Test Set")
        plt.tight_layout()
        plt.show()

        # Compare model accuracies
        plot_model_accuracies(knn_test_acc, svm_test_acc)

        # Learning curve (SVM)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_sizes, train_scores, val_scores = learning_curve(
            svm_model, X_train, y_train, cv=skf, scoring='accuracy', n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 5)
        )
        train_mean = np.mean(train_scores, axis=1)
        train_std  = np.std(train_scores, axis=1)
        val_mean   = np.mean(val_scores, axis=1)
        val_std    = np.std(val_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Accuracy')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
        plt.plot(train_sizes, val_mean, 'o-', color='green', label='Validation Accuracy')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color='green')
        plt.title("Learning Curve (SVM)")
        plt.xlabel("Training Set Size")
        plt.ylabel("Accuracy")
        plt.legend(loc="best")
        plt.grid()
        plt.tight_layout()
        plt.show()

        # Save models
        joblib.dump(knn, KNN_MODEL_PATH)
        logging.info(f"KNN model saved as '{KNN_MODEL_PATH}'")
        joblib.dump(svm_model, SVM_MODEL_PATH)
        logging.info(f"SVM model saved as '{SVM_MODEL_PATH}'")

        # PCA Visualization
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_train)
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, ticks=[0, 1], label='Class (0=Normal, 1=LBBB)')
        plt.title("PCA Visualization of Training Data")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.tight_layout()
        plt.show()

        logging.info("Training & Evaluation complete!")

        # ------------------- LABEL & VISUALIZE WAVES EXAMPLE -------------------- #
        # For demonstration, let's label & visualize waves on the first Normal Test ECG
        if len(processed_normal_test) > 0:
            example_ecg = processed_normal_test[0]  # 1D array
            wave_labels = label_ecg_waves(example_ecg, fs=360)
            visualize_ecg_waves(example_ecg, wave_labels, fs=360)
            # This will show the labeled waves in a plot

    except Exception as e:
        logging.error(f"An error occurred during training and evaluation: {e}")

def visualize_raw_signals(normal_data, lbbb_data):
    """
    Visualizes the first raw (padded) Normal and LBBB ECG signals before preprocessing.
    """
    if len(normal_data) == 0 or len(lbbb_data) == 0:
        logging.warning("Empty data provided to visualize_raw_signals.")
        return

    # Normal
    plt.figure(figsize=(10, 4))
    plt.plot(normal_data[0], label='Raw Normal Signal')
    plt.title("Raw (Padded) Normal Signal - Before Preprocessing")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # LBBB
    plt.figure(figsize=(10, 4))
    plt.plot(lbbb_data[0], label='Raw LBBB Signal', color='orange')
    plt.title("Raw (Padded) LBBB Signal - Before Preprocessing")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()

def visualize_processed_signals(normal_data, lbbb_data):
    """
    Visualizes the first processed (preprocessed) Normal and LBBB ECG signals after preprocessing.
    """
    if len(normal_data) == 0 or len(lbbb_data) == 0:
        logging.warning("Empty data provided to visualize_processed_signals.")
        return

    # Normal
    plt.figure(figsize=(10, 4))
    plt.plot(normal_data[0], label='Processed Normal Signal')
    plt.title("Processed Normal Signal - After Preprocessing")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude (Normalized)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # LBBB
    plt.figure(figsize=(10, 4))
    plt.plot(lbbb_data[0], label='Processed LBBB Signal', color='orange')
    plt.title("Processed LBBB Signal - After Preprocessing")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude (Normalized)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# --------------------------- #
#      Tkinter GUI Class      #
# --------------------------- #

class ECGApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # ---- UI Colors / Style ----
        self.bg_color = "#FFFFFF"          # White background
        self.btn_color = "#4CAF50"         # Green buttons
        self.btn_hover_color = "#45A049"   # Darker green on hover
        self.text_color = "#000000"        # Black text
        self.title_color = "#FF0000"       # Red title

        # Configure the main window
        self.title("Digital Signal Processing - ECG Classification")
        self.geometry("1000x800")           # Increased size for better layout
        self.configure(bg=self.bg_color)

        # Title label with static red color
        self.title_label = tk.Label(self, text="DIGITAL SIGNAL PROCESSING",
                                    font=("Helvetica", 24, "bold"),
                                    fg=self.title_color,
                                    bg=self.bg_color)
        self.title_label.pack(pady=20)

        # Attempt to load trained models
        self.knn_model = None
        self.svm_model = None
        self.load_models()

        # Frame for file upload and manual entry (arranged vertically)
        frame_input = tk.Frame(self, bg=self.bg_color)
        frame_input.pack(pady=20)

        # Load ECG File Button
        load_btn = tk.Button(frame_input, text="Load ECG File", command=self.load_file,
                             bg=self.btn_color, fg=self.text_color, font=("Helvetica", 14, "bold"),
                             activebackground=self.btn_hover_color, activeforeground=self.text_color,
                             width=25, height=2, borderwidth=0, cursor="hand2")
        load_btn.pack(pady=10)
        self.add_hover_effect(load_btn)

        # Enter Signal Manually Button
        manual_btn = tk.Button(frame_input, text="Enter Signal Manually", command=self.enter_signal_manually,
                               bg=self.btn_color, fg=self.text_color, font=("Helvetica", 14, "bold"),
                               activebackground=self.btn_hover_color, activeforeground=self.text_color,
                               width=25, height=2, borderwidth=0, cursor="hand2")
        manual_btn.pack(pady=10)
        self.add_hover_effect(manual_btn)

        # Frame for classification buttons (arranged vertically)
        frame_classify = tk.Frame(self, bg=self.bg_color)
        frame_classify.pack(pady=30)

        # Predict with KNN Button
        knn_btn = tk.Button(frame_classify, text="Predict with KNN", command=self.predict_knn,
                            bg=self.btn_color, fg=self.text_color, font=("Helvetica", 14, "bold"),
                            activebackground=self.btn_hover_color, activeforeground=self.text_color,
                            width=25, height=2, borderwidth=0, cursor="hand2")
        knn_btn.pack(pady=10)
        self.add_hover_effect(knn_btn)

        # Predict with SVM Button
        svm_btn = tk.Button(frame_classify, text="Predict with SVM", command=self.predict_svm,
                            bg=self.btn_color, fg=self.text_color, font=("Helvetica", 14, "bold"),
                            activebackground=self.btn_hover_color, activeforeground=self.text_color,
                            width=25, height=2, borderwidth=0, cursor="hand2")
        svm_btn.pack(pady=10)
        self.add_hover_effect(svm_btn)

        # Result Display Label
        self.result_label = tk.Label(self, text="", font=("Helvetica", 16, "bold"),
                                     bg=self.bg_color, fg="#FF5733")
        self.result_label.pack(pady=20)

        # Status Display Label
        self.label_file = tk.Label(self, text="No file or manual data loaded yet.", font=("Helvetica", 12),
                                   bg=self.bg_color, fg=self.text_color)
        self.label_file.pack()

        # Visualization Frame
        self.frame_visual = tk.Frame(self, bg=self.bg_color)
        self.frame_visual.pack(pady=20)

        # Button to visualize the ECG signal
        visualize_btn = tk.Button(self.frame_visual, text="Visualize ECG Signal", command=self.visualize_signal,
                                  bg="#FFC300", fg=self.text_color, font=("Helvetica", 14, "bold"),
                                  activebackground="#FF5733", activeforeground=self.text_color,
                                  width=25, height=2, borderwidth=0, cursor="hand2")
        visualize_btn.pack(pady=10)
        self.add_hover_effect(visualize_btn)

        # Placeholder for matplotlib canvas
        self.canvas = None

        self.ecg_data = None

    def load_models(self):
        """
        Loads the trained KNN and SVM models from disk.
        """
        try:
            if os.path.isfile(KNN_MODEL_PATH):
                self.knn_model = joblib.load(KNN_MODEL_PATH)
                logging.info(f"KNN model loaded from '{KNN_MODEL_PATH}'")
            else:
                logging.warning(f"KNN model file not found: '{KNN_MODEL_PATH}'")
            
            if os.path.isfile(SVM_MODEL_PATH):
                self.svm_model = joblib.load(SVM_MODEL_PATH)
                logging.info(f"SVM model loaded from '{SVM_MODEL_PATH}'")
            else:
                logging.warning(f"SVM model file not found: '{SVM_MODEL_PATH}'")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load models: {e}")
            logging.error(f"Error loading models: {e}")

    def add_hover_effect(self, button):
        """
        Adds a hover effect to a button.
        Changes background color on enter and revert on leave.
        """
        def on_enter(e):
            button['background'] = self.btn_hover_color

        def on_leave(e):
            button['background'] = self.btn_color

        button.bind("<Enter>", on_enter)
        button.bind("<Leave>", on_leave)

    def load_file(self):
        """
        Allows the user to load an ECG file from their system.
        Parses the first line of the file as the ECG signal.
        """
        filepath = filedialog.askopenfilename(
            title="Select ECG File",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if filepath:
            self.label_file.config(text=f"Loaded File: {filepath}")
            try:
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                # For simplicity, parse only the first line
                row = lines[0].strip().split('|')
                samples = [float(x) for x in row if x.strip()]
                self.ecg_data = np.array(samples, dtype=float)
                messagebox.showinfo("File Loaded", f"{len(self.ecg_data)} samples loaded.")
                logging.info(f"ECG data loaded from file '{filepath}' with {len(self.ecg_data)} samples.")
            except Exception as e:
                messagebox.showerror("Error", f"Cannot parse ECG data: {e}")
                logging.error(f"Error parsing ECG data from file '{filepath}': {e}")
                self.ecg_data = None

    def enter_signal_manually(self):
        """
        Opens a Toplevel window for the user to manually enter ECG samples.
        """
        manual_window = tk.Toplevel(self)
        manual_window.title("Manual ECG Entry")
        manual_window.geometry("600x400")
        manual_window.configure(bg=self.bg_color)

        # Instruction Label
        label_inst = tk.Label(manual_window, text="Enter ECG samples (space, comma, or '|' delimited):",
                              bg=self.bg_color, fg=self.text_color, font=("Helvetica", 12, "bold"))
        label_inst.pack(pady=10)

        # Text Entry Box
        text_box = tk.Text(manual_window, height=15, width=60, font=("Helvetica", 12))
        text_box.pack(pady=10)

        def confirm_manual():
            """
            Parses the manually entered ECG samples and updates the main ECG data.
            """
            user_input = text_box.get("1.0", "end").strip()
            # Attempt to parse
            replaced = user_input.replace(',', ' ').replace('|', ' ')
            splitted = replaced.split()

            try:
                samples = [float(x) for x in splitted]
                if len(samples) == 0:
                    raise ValueError("No valid samples entered.")
                self.ecg_data = np.array(samples, dtype=float)
                self.label_file.config(text="Manual Signal Entered.")
                messagebox.showinfo("Manual Entry", f"{len(samples)} samples parsed.")
                logging.info(f"Manual ECG data entered with {len(samples)} samples.")
                manual_window.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to parse manually entered data: {e}")
                logging.error(f"Error parsing manual ECG data: {e}")

        # Confirm Button
        confirm_btn = tk.Button(manual_window, text="Confirm", command=confirm_manual,
                                bg=self.btn_color, fg=self.text_color, font=("Helvetica", 12, "bold"),
                                activebackground=self.btn_hover_color, activeforeground=self.text_color,
                                width=15, height=2, borderwidth=0, cursor="hand2")
        confirm_btn.pack(pady=20)
        self.add_hover_effect(confirm_btn)

    def predict_knn(self):
        """
        Predicts the class of the loaded ECG signal using the KNN model.
        """
        if self.ecg_data is None:
            messagebox.showwarning("Warning", "No ECG data loaded (file or manual).")
            logging.warning("Prediction attempted without ECG data.")
            return
        if not self.knn_model:
            messagebox.showerror("Error", "KNN model not loaded or unavailable.")
            logging.error("KNN model not loaded.")
            return

        try:
            processed = preprocess_signal(self.ecg_data)
            coeffs = wavelet_decompose(processed, 'db6', level=6)
            features = extract_features_from_coeffs(coeffs).reshape(1, -1)

            pred = self.knn_model.predict(features)[0]
            label = "Normal" if pred == 0 else "LBBB"
            self.result_label.config(text=f"KNN Prediction: {label}")
            logging.info(f"KNN Prediction: {label}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during KNN prediction: {e}")
            logging.error(f"Error during KNN prediction: {e}")

    def predict_svm(self):
        """
        Predicts the class of the loaded ECG signal using the SVM model.
        """
        if self.ecg_data is None:
            messagebox.showwarning("Warning", "No ECG data loaded (file or manual).")
            logging.warning("Prediction attempted without ECG data.")
            return
        if not self.svm_model:
            messagebox.showerror("Error", "SVM model not loaded or unavailable.")
            logging.error("SVM model not loaded.")
            return

        try:
            processed = preprocess_signal(self.ecg_data)
            coeffs = wavelet_decompose(processed, 'db6', level=6)
            features = extract_features_from_coeffs(coeffs).reshape(1, -1)

            pred = self.svm_model.predict(features)[0]
            label = "Normal" if pred == 0 else "LBBB"
            self.result_label.config(text=f"SVM Prediction: {label}")
            logging.info(f"SVM Prediction: {label}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during SVM prediction: {e}")
            logging.error(f"Error during SVM prediction: {e}")

    def visualize_signal(self):
        """
        Visualizes the loaded ECG signal with labeled P, T, QRS parts.
        Embeds the plot within the Tkinter GUI.
        """
        if self.ecg_data is None:
            messagebox.showwarning("Warning", "No ECG data loaded to visualize.")
            logging.warning("Visualization attempted without ECG data.")
            return

        try:
            # Clear previous canvas if exists
            if self.canvas:
                self.canvas.get_tk_widget().destroy()

            # Preprocess the signal
            processed = preprocess_signal(self.ecg_data)
            wave_labels = label_ecg_waves(processed, fs=360)

            # Create a matplotlib figure
            fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
            ax.plot(processed, label='ECG Signal', color='black')

            # Mark R-peaks
            for idx, r in enumerate(wave_labels['R_peaks']):
                ax.axvline(r, color='red', linestyle='--', label='R-peak' if idx == 0 else "")
            
            # Mark Q and S points
            for idx, q in enumerate(wave_labels['Q_points']):
                ax.axvline(q, color='green', linestyle=':', label='Q/S Point' if idx == 0 else "")
            for idx, s in enumerate(wave_labels['S_points']):
                ax.axvline(s, color='green', linestyle=':', label='Q/S Point' if idx == 0 else "")

            # Highlight P and T waves
            for idx, (p_start, p_end) in enumerate(wave_labels['P_waves']):
                ax.axvspan(p_start, p_end, color='yellow', alpha=0.3, label='P Wave' if idx == 0 else "")
            
            for idx, (t_start, t_end) in enumerate(wave_labels['T_waves']):
                ax.axvspan(t_start, t_end, color='orange', alpha=0.3, label='T Wave' if idx == 0 else "")

            ax.set_title("ECG Signal with Marked P, T, QRS Parts")
            ax.set_xlabel("Samples")
            ax.set_ylabel("Amplitude (Normalized)")
            ax.legend(loc='upper right')
            plt.tight_layout()

            # Embed the plot in Tkinter
            self.canvas = FigureCanvasTkAgg(fig, master=self.frame_visual)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack()
            logging.info("ECG signal visualized in GUI.")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during visualization: {e}")
            logging.error(f"Error during ECG visualization: {e}")

# --------------------------- #
#        Main Function        #
# --------------------------- #

def main():
    """
    Main function to handle command-line arguments and launch appropriate functionalities.
    """
    parser = argparse.ArgumentParser(description="ECG Classification - Train or GUI")
    parser.add_argument("--train", action="store_true", help="Train models and visualize results.")
    parser.add_argument("--gui", action="store_true", help="Launch the Tkinter GUI.")
    args = parser.parse_args()

    if args.train:
        train_and_evaluate()

    if args.gui:
        app = ECGApp()
        app.mainloop()

    if not args.train and not args.gui:
        parser.print_help()

if __name__ == "__main__":
    main()
