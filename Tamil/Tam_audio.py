import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout

# Path to the audio directory
AUDIO_DIR = r"C:\Users\nishd\OneDrive\Documents\shared task\multimodal-hatespeech\multimodal-hatespeech\train\tamil\audio"

# Helper function: Extract label from filename
def extract_labels(filename):
    parts = filename.split('_')
    hate_status = parts[0]  # H or NH
    class_label = parts[3]  # G, P, R, C, or N
    return hate_status, class_label

# Feature extraction: MFCC
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        return np.mean(mfcc.T, axis=0)  # Average across time
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Preprocessing and saving features
data = []
labels = []

for root, dirs, files in os.walk(AUDIO_DIR):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            hate_status, class_label = extract_labels(file)
            
            # Combine Hate/Non-Hate and Class labels
            combined_label = f"{hate_status}_{class_label}"
            feature = extract_features(file_path)
            if feature is not None:
                data.append(feature)
                labels.append(combined_label)

# Convert to NumPy arrays
data = np.array(data)
labels = np.array(labels)

# Encode labels
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data, encoded_labels, test_size=0.2, random_state=42)

# CNN Model for Classification
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(data.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    Conv1D(128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(label_encoder.classes_), activation='softmax')  # Number of classes
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Reshape for CNN input
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Train the model
history = model.fit(X_train_cnn, y_train, epochs=20, batch_size=32, validation_data=(X_test_cnn, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test_cnn, y_test, verbose=0)  # Evaluate on the test set
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Evaluate the model
y_pred = np.argmax(model.predict(X_test_cnn), axis=1)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


import os
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Path to the test audio directory
TEST_AUDIO_DIR = r"C:\Users\nishd\OneDrive\Documents\shared task\multimodal-hatespeech\multimodal-hatespeech\test (1)\test\tamil\audio"

# Function to extract features from audio
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        return np.mean(mfcc.T, axis=0)  # Average across time
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Load test data and extract features
test_data = []
test_filenames = []

for root, dirs, files in os.walk(TEST_AUDIO_DIR):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            feature = extract_features(file_path)
            if feature is not None:
                test_data.append(feature)
                test_filenames.append(file)

# Convert test data to NumPy array
test_data = np.array(test_data)

# Make predictions on the test data
y_test_pred = rf_model.predict(test_data)

# Decode predictions to class labels
y_test_pred_labels = label_encoder.inverse_transform(y_test_pred)

# Create a DataFrame for the results
results_df = pd.DataFrame({
    'Filename': test_filenames,
    'Predicted Label': y_test_pred_labels
})

# Save the results to a TSV file
output_path = r"C:\Users\nishd\OneDrive\Documents\shared task\results\tamil_randomforest_results_11.tsv"
results_df.to_csv(output_path, sep='\t', index=False)

print(f"Predictions saved to '{output_path}'")