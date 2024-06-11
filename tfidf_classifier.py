import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# Load JSON data
def load_json_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data


# Convert JSON data to DataFrame
def json_to_dataframe(json_data):
    df = pd.DataFrame(json_data)
    return df


# Preprocess text data
def preprocess_text(df):
    # Add any additional preprocessing steps if needed
    return df


# Load data
file_path = 'data/DPR_pairs/DPR_pairs_train.jsonl'
json_data = load_json_data(file_path)

# Convert to DataFrame
df = json_to_dataframe(json_data)

# Preprocess data
df = preprocess_text(df)

# Split data into features and labels
X = df['x']
y = df['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train an SVM classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_tfidf, y_train)

# Predict on the test data
y_pred = svm_classifier.predict(X_test_tfidf)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
