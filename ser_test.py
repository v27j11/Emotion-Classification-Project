import os
import numpy as np
import librosa
import pickle
from tensorflow.keras.models import load_model

# ==== CONFIG ====
SAMPLE_RATE = 16000
SEGMENT_DURATION = 2.0
SEGMENT_OVERLAP = 0.5

# ==== Feature Extraction (same as training) ====
def extract_segment_features(y, sr):
    segment_len = int(SEGMENT_DURATION * sr)
    hop_len = int(segment_len * (1 - SEGMENT_OVERLAP))
    features = []
    for start in range(0, len(y) - segment_len + 1, hop_len):
        segment = y[start:start + segment_len]
        feats = []
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=20)
        mfcc_d1 = librosa.feature.delta(mfcc)
        mfcc_d2 = librosa.feature.delta(mfcc, order=2)
        feats += list(np.mean(mfcc, axis=1)) + list(np.std(mfcc, axis=1))
        feats += list(np.mean(mfcc_d1, axis=1)) + list(np.mean(mfcc_d2, axis=1))
        feats += list(np.mean(librosa.feature.chroma_stft(y=segment, sr=sr), axis=1))
        feats += list(np.mean(librosa.power_to_db(librosa.feature.melspectrogram(y=segment, sr=sr)), axis=1))
        feats += list(np.mean(librosa.feature.spectral_contrast(y=segment, sr=sr), axis=1))
        feats.append(np.mean(librosa.feature.spectral_bandwidth(y=segment, sr=sr)))
        feats.append(np.mean(librosa.feature.spectral_rolloff(y=segment, sr=sr)))
        feats.append(np.mean(librosa.feature.spectral_flatness(y=segment)))
        feats.append(np.mean(librosa.feature.zero_crossing_rate(segment)))
        feats.append(np.mean(librosa.feature.rms(y=segment)))
        features.append(feats)
    features = np.array(features)
    stats = []
    for stat in [np.mean, np.std, np.min, np.max]:
        stats += list(stat(features, axis=0))
    return np.array(stats)

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def predict_audio(filename, dnn, rf, scaler, pca, le):
    y_raw, sr = librosa.load(filename, sr=SAMPLE_RATE)
    feat = extract_segment_features(y_raw, sr)
    feat_scaled = scaler.transform([feat])
    feat_pca = pca.transform(feat_scaled)
    pred_dnn = dnn.predict(feat_pca)
    pred_rf = rf.predict_proba(feat_pca)
    pred = (pred_dnn + pred_rf) / 2
    pred_label = np.argmax(pred, axis=1)[0]
    emotion = le.inverse_transform([pred_label])[0]
    return emotion

if __name__ == "__main__":
    # Load models and preprocessors
    dnn = load_model("final_ser_model.keras")
    with open("rf_model.pkl", "rb") as f: rf = pickle.load(f)
    with open("scaler.pkl", "rb") as f: scaler = pickle.load(f)
    with open("pca.pkl", "rb") as f: pca = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f: le = pickle.load(f)

    # Folder with test wav files
    TEST_FOLDER = "./test_wavs"
    for wavfile in os.listdir(TEST_FOLDER):
        if not wavfile.endswith(".wav"): continue
        path = os.path.join(TEST_FOLDER, wavfile)
        emotion = predict_audio(path, dnn, rf, scaler, pca, le)
        print(f"{wavfile}: {emotion}")
