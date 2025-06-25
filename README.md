# Speech Emotion Recognition (SER) — DNN + Random Forest Ensemble

## Project Description

This project is a robust Speech Emotion Recognition (SER) system that predicts emotions from both **song** and **speech** audio files. It uses a hybrid ensemble of a Deep Neural Network (DNN) and a Random Forest (RF), trained on a comprehensive set of audio features. The pipeline is designed for high accuracy, strong generalization, and strict data hygiene, making it suitable for research and competition use.

---

## Pre-processing Methodology

### 1. Audio Loading and Resampling
- **Purpose:** All `.wav` files are loaded at a fixed sample rate (16,000 Hz).
- **Reasoning:**  
  Standardizing the sample rate ensures that all extracted features are comparable across files, regardless of their original recording settings. This is essential for consistent feature extraction and model performance.

### 2. Segmentation
- **Purpose:** Each audio file is split into overlapping segments (2 seconds, 50% overlap).
- **Reasoning:**  
  Emotions can change within an utterance. Segmenting helps capture local emotional cues and increases the number of training samples, improving model robustness. Overlap ensures emotional transitions are not missed at segment boundaries.

### 3. Feature Extraction
- **Purpose:** For each segment, a rich set of features is extracted:
  - MFCCs (mean, std, delta, delta2)
  - Chroma
  - Mel Spectrogram (dB)
  - Spectral Contrast
  - Spectral Bandwidth, Rolloff, Flatness
  - Zero Crossing Rate, RMSE
- **Reasoning:**  
  These features capture both the spectral and temporal characteristics of speech and song, which are essential for distinguishing emotional states. Aggregating with mean, std, min, and max across segments ensures both central tendency and variability are captured, making features robust to local fluctuations and noise.

### 4. Data Augmentation
- **Purpose:** Each training file is augmented 3 times with a random transformation (pitch shift, time stretch, noise, volume, or shift).
- **Reasoning:**  
  Augmentation increases the diversity of the training set, helping the model generalize to new speakers, microphones, and environments. Each augmentation type simulates a different kind of real-world variability. Augmentation is only applied to the training set to avoid data leakage.

### 5. Label Encoding
- **Purpose:** Emotions are mapped to integers using `LabelEncoder`.
- **Reasoning:**  
  Machine learning models require numeric labels for classification.

### 6. Feature Scaling
- **Purpose:** Features are standardized using `StandardScaler` (fit on train set only).
- **Reasoning:**  
  Scaling ensures all features contribute equally to model training and improves convergence during training.

### 7. Dimensionality Reduction (PCA)
- **Purpose:** PCA is applied to retain 98% variance, fit on training set only.
- **Reasoning:**  
  Reduces feature redundancy and noise, speeds up training, and prevents overfitting by focusing on the most informative components.

### 8. Class Weighting
- **Purpose:** Class weights are computed from the training set and used in DNN and RF.
- **Reasoning:**  
  Addresses class imbalance so the model does not bias toward majority classes and ensures fair learning across all emotions.

---

## Model Pipeline

1. **Data Split:** Stratified train/test split (80/20) to preserve class balance.
2. **Data Augmentation:** Only applied to training data, as described above.
3. **Feature Extraction, Scaling, PCA:** As described in preprocessing.
4. **Model Training:**
    - **DNN:** 512 → 256 → 128 dense layers (ReLU, Dropout), softmax output, focal loss, Adam optimizer, class weights, early stopping.
    - **Random Forest:** 500 trees, class-balanced, trained on PCA features.
5. **Ensemble Prediction:** DNN softmax and RF probabilities are averaged for each test sample; final class is the argmax.
6. **Evaluation & Reporting:** Prints classification report, confusion matrix, per-class accuracy, overall accuracy, and macro F1 score. Confusion matrix is saved as `confusion_matrix.png`.
7. **Saving Artifacts:** DNN (`final_ser_model.keras`), RF (`rf_model.pkl`), scaler (`scaler.pkl`), PCA (`pca.pkl`), and label encoder (`label_encoder.pkl`) are all saved.

---

## Accuracy Metrics (Sample Results)

- **Overall Accuracy:** 78.0%
- **Macro F1 Score:** 0.775
- **Per-Class Accuracy:**
    - angry:     89.3%
    - calm:      94.7%
    - disgust:   61.5%
    - fearful:   70.7%
    - happy:     72.0%
    - neutral:   92.1%
    - sad:       58.7%
    - surprised: 89.7%

---

## Why Are Some Classes Below 75% Accuracy?

Despite strong overall performance, some classes (notably **sad**, **disgust**, **happy**, and **fearful**) may have per-class accuracy below 75%. This is a common challenge in SER and can be attributed to:

- **Acoustic Similarity:** Emotions like *sad*, *fearful*, and *disgust* often share low-energy, low-pitch, or monotone characteristics, making them harder to distinguish, especially in noisy or song contexts.
- **Class Imbalance:** Even with augmentation, some emotions are underrepresented, so the model may not see enough diverse examples to generalize well.
- **Overlap in Expressive Cues:** Some emotions (e.g., *happy* and *surprised*, or *sad* and *fearful*) have overlapping prosodic and spectral features, leading to confusion in the model.
- **Song vs. Speech Differences:** Emotional cues in song can be masked by melody, harmony, or musical effects, making it harder for the model to pick up on subtle emotional differences compared to speech.
- **Augmentation Limitations:** While augmentation helps, it cannot fully create the diversity of real emotional expression, especially for minority classes.

---

## Required Libraries

Install all dependencies with:
!pip install numpy librosa soundfile matplotlib seaborn scikit-learn tensorflow

## Usage

1. **Place your audio files** in the specified folders.
2. **Run the script**. All preprocessing, training, evaluation, and saving are handled automatically.
3. **Check the printed metrics and `confusion_matrix.png`** for model performance.
