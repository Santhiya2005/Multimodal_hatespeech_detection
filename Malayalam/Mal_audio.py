import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau
import pandas as pd

# Paths
train_audio_dir = r"C:\Users\nishd\OneDrive\Documents\shared task\multimodal-hatespeech\multimodal-hatespeech\train\malayalam\audio"
test_audio_dir = r"C:\Users\nishd\OneDrive\Documents\shared task\multimodal-hatespeech\multimodal-hatespeech\test (1)\test\malayalam\audio"
results_dir = r"C:\Users\nishd\OneDrive\Documents\shared task\results"

# Ensure results directory exists
os.makedirs(results_dir, exist_ok=True)

# Extract class labels from filenames
def get_label(file_name):
    label_parts = file_name.split('_')
    if label_parts[0] == 'H':  # Hate
        return label_parts[3]  # G, P, R, C
    elif label_parts[0] == 'NH':  # Non-Hate
        return 'NH'

# Noise reduction
def noise_reduction(audio, sr):
    noise_profile = np.mean(audio[:int(0.5 * sr)])  # Estimate noise from the first 0.5 seconds
    return audio - noise_profile

# Feature extraction
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        audio = noise_reduction(audio, sr)  # Apply noise reduction
        mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr).T, axis=0)
        return np.concatenate((mfcc, chroma, spectral_contrast))
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Preprocess training data
data = []
labels = []

for file_name in os.listdir(train_audio_dir):
    if file_name.endswith('.wav'):
        label = get_label(file_name)
        file_path = os.path.join(train_audio_dir, file_name)
        features = extract_features(file_path)
        if features is not None:
            data.append(features)
            labels.append(label)

# Convert to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)

# Reshape for CNN input
X_train_cnn = X_train[..., np.newaxis]
X_val_cnn = X_val[..., np.newaxis]

# Build CNN model
model = Sequential([
    Conv1D(128, kernel_size=5, activation='relu', input_shape=(data.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.5),
    Conv1D(256, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
history = model.fit(X_train_cnn, y_train, epochs=20, batch_size=32, validation_data=(X_val_cnn, y_val), callbacks=[lr_scheduler])

# Evaluate the model on validation data
val_loss, val_accuracy = model.evaluate(X_val_cnn, y_val)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Test data preprocessing
test_data = []
test_filenames = []

for file_name in os.listdir(test_audio_dir):
    if file_name.endswith('.wav'):
        file_path = os.path.join(test_audio_dir, file_name)
        features = extract_features(file_path)
        if features is not None:
            test_data.append(features)
            test_filenames.append(file_name)

# Convert test data to numpy array and reshape
test_data = np.array(test_data)
test_data_cnn = test_data[..., np.newaxis]

# Predictions on test data
test_predictions = np.argmax(model.predict(test_data_cnn), axis=1)
test_labels = label_encoder.inverse_transform(test_predictions)

# Save predictions to TSV file
results_df = pd.DataFrame({
    'Filename': test_filenames,
    'Predicted Label': test_labels
})
output_path = os.path.join(results_dir, "malayalam_predictions.tsv")
results_df.to_csv(output_path, sep='\t', index=False)

print(f"Predictions saved to '{output_path}'")
# Generate classification report
y_val_pred = np.argmax(model.predict(X_val_cnn), axis=1)
print("Classification Report:\n", classification_report(y_val, y_val_pred, target_names=label_encoder.classes_))

