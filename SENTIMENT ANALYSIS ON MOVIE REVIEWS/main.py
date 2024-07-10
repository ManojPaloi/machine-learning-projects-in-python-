import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Sample dataset (for demonstration purposes)
reviews = [
    ("I loved the movie, it was fantastic!", 1),
    ("The movie was terrible, I hated it.", 0),
    ("It was an okay movie, not great but not bad either.", 1),
    ("Absolutely wonderful movie, will watch again.", 1),
    ("Worst movie ever, waste of time.", 0),
    ("The plot was predictable and boring.", 0),
    ("A masterpiece, beautifully executed.", 1),
    ("I wouldn't recommend this movie to anyone.", 0),
    ("A decent movie, could have been better.", 1),
    ("Loved the characters and the storyline.", 1),
    ("Terrible acting and awful direction.", 0),
    ("An outstanding film, a must-watch.", 1),
    ("The movie was dull and uninspiring.", 0),
    ("An enjoyable experience, really liked it.", 1),
    ("Not worth watching, very disappointing.", 0)
]

# Convert to DataFrame
data = pd.DataFrame(reviews, columns=['Review', 'Sentiment'])

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    tokens = text.split()
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords.words('english')]
    return ' '.join(tokens)

# Apply preprocessing
data['Processed_Review'] = data['Review'].apply(preprocess_text)

# Splitting the dataset into features and target variable
X = data['Processed_Review']
y = data['Sentiment']

# Feature extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dictionary to store models and their names
models = {
    'Naive Bayes': MultinomialNB(),
    'Random Forest Classifier': RandomForestClassifier(n_estimators=100, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'XGBoost Classifier': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    'Logistic Regression': LogisticRegression(),
    'Decision Tree Classifier': DecisionTreeClassifier(random_state=42)
}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name} Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))
    print()

# Confusion Matrix for Logistic Regression as an example
conf_matrix = confusion_matrix(y_test, models['Logistic Regression'].predict(X_test))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()
