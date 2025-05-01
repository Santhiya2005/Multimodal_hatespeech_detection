# Random forest final
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import re

# Load training dataset
df_train = pd.read_excel(r'C:\Users\nishd\OneDrive\Documents\shared task\multimodal-hatespeech\multimodal-hatespeech\train\tamil\text\TA-AT-train.xlsx')

# Preprocess Tamil text
def preprocess_tamil_text(text):
    # Remove non-alphanumeric characters
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

# Apply preprocessing to training data
df_train['Transcript'] = df_train['Transcript'].apply(preprocess_tamil_text)

# Save preprocessed training data to TSV
preprocessed_file_path = r'C:\Users\nishd\OneDrive\Documents\shared task\preprocessedfiles\randompreprocess.tsv'
df_train.to_csv(preprocessed_file_path, sep='\t', index=False)

# Vectorization: TF-IDF for text data
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(df_train['Transcript'])

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
y = df_train['Class Label Short']

# Label encode the target variable to numeric values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Apply SMOTE to balance classes
X_resampled, y_resampled = smote.fit_resample(X_tfidf, y_encoded)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Output results
print(f'Accuracy: {accuracy:.4f}')

# Load test dataset (without labels)
df_test = pd.read_excel(r'C:\Users\nishd\OneDrive\Documents\shared task\multimodal-hatespeech\multimodal-hatespeech\test (1)\test\tamil\text\TA-AT-test.xlsx')

# Preprocess the test data using the same function as the training data
df_test['Transcript'] = df_test['Transcript'].apply(preprocess_tamil_text)

# Vectorize the test data using the same TF-IDF vectorizer
X_test_tfidf = tfidf_vectorizer.transform(df_test['Transcript'])

# Make predictions on the test data
y_test_pred = rf_model.predict(X_test_tfidf)

# Decode the predicted labels back to the original classes
y_test_pred_labels = label_encoder.inverse_transform(y_test_pred)

# Store the predictions in the test DataFrame
df_test['Predicted Label'] = y_test_pred_labels

# Save the predictions to a new TSV file
prediction_file_path = r'C:\Users\nishd\OneDrive\Documents\shared task\results\Brightred_Tamiltext.tsv'
df_test.to_csv(prediction_file_path, sep='\t', index=False)

print(f"Preprocessed data saved to '{preprocessed_file_path}'")
print(f"Predictions saved to '{prediction_file_path}'")