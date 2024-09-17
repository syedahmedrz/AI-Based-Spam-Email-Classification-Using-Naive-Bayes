import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import json

def read_file(file_path):
    """Reads a JSON file and returns the data."""
    with open(file_path) as file:
        data = json.load(file)
    return data

# Load data
data = read_file('data.json')
test_data = read_file('test_data.json')

# Create DataFrame
df = pd.DataFrame(data)

# Prepare the data
X = df['email']
Y = df['label']

# Convert text data to numerical data
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, Y, test_size=0.25, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

def classify_new_email(new_email):
    """Classifies a new email as 'spam' or 'not spam'."""
    new_email_vectorized = vectorizer.transform([new_email])
    prediction = model.predict(new_email_vectorized)
    return prediction[0]

# Test with a new email subject
new_email = "Important information about your account"
result = classify_new_email(new_email)
print(f'The new email is classified as: {result}')

# Test a bunch of subjects
for subject in test_data:
    test_subject = classify_new_email(subject)
    print(f'The new email is classified as: {test_subject}')
