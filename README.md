### Import Libraries
```python 
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import json
```

### Libraries and Their Uses

- **numpy and pandas**: Used for handling and manipulating data.
- **sklearn**: Contains tools for machine learning tasks:
  - **train_test_split**: To split the data into training and testing sets.
  - **CountVectorizer**: Converts text data into numerical data.
  - **MultinomialNB**: A type of Naive Bayes classifier suitable for text data.
  - **accuracy_score**: Evaluates the accuracy of the model.
- **json**: For reading JSON files.

### Read Data from JSON Files

```python
def read_file(file_path):
    """Reads a JSON file and returns the data."""
    with open(file_path) as file:
        data = json.load(file)
    return data

# load data
data = read_file('data.json')
test_data = read_file('test_data.json')
```

### Prepare the Data
```python
# Create DataFrame
df = pd.DataFrame(data)
X = df['email']
Y = df['label']
```
**df:** Converts the loaded data into a pandas DataFrame.
**X**: Contains the email texts.
**Y:** Contains the labels (e.g., "spam" or "not spam").

### Text Data to Numerical Data
```python
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)
```
**CountVectorizer:** Converts text into a matrix of token counts (i.e., numerical data).
**X_vectorized:** Contains the numerical representation of the email texts.

### Split Data into Training and Testing Sets
```python
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, Y, test_size=0.25, random_state=42)
```
**train_test_split:** Splits the data into training (75%) and testing (25%) sets.
**X_train and y_train:** Used to train the model.
**X_test and y_test:** Used to evaluate the model's performance.

### Train the Model
```python
model = MultinomialNB()
model.fit(X_train, y_train)
```
**MultinomialNB:** Creates a Naive Bayes classifier model.
**model.fit:** Trains the model using the training data.

### Make Predictions and Evaluate
```python
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
```
**model.predict:** Makes predictions on the test set.
**accuracy_score:** Calculates the accuracy of the predictions.
**print:** Displays the accuracy of the model.

### Classify New Email Function
```python
def classify_new_email(new_email):
    """Classifies a new email as 'spam' or 'not spam'."""
    new_email_vectorized = vectorizer.transform([new_email])
    prediction = model.predict(new_email_vectorized)
    return prediction[0]
```

```python
# Test with a new email subject
new_email = "Important information about your account"
result = classify_new_email(new_email)
print(f'The new email is classified as: {result}')
```
**classify_new_email:** Function that takes a new email, converts it to numerical data, and predicts its label.
**new_email:** An example email to test the classification.
**result:** Shows whether the email is classified as "spam" or "not spam".

### Classify Multiple Emails
```python
# Test a bunch of subjects
for subject in test_data:
    test_subject = classify_new_email(subject)
    print(f'The new email is classified as: {test_subject}')
```
**for loop:** Iterates through the test_data and classifies each email using the classify_new_email function.
**print:** Displays the classification result for each email.

### Email Subjects Dataset Example
| Email                                | Label    |
| ------------------------------------ | -------- |
| "Win money now!"                     | Spam     |
| "Important information about your account" | Not Spam |
| "Cheap loans available"              | Spam     |
| "Meeting on Friday"                  | Not Spam |
| "Get rich quick"                     | Spam     |
| "Your invoice is ready"              | Not Spam |
| "Earn money while you sleep"         | Spam     |
| "Team lunch tomorrow"                | Not Spam |

