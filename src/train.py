import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from preprocess import clean_text

print("🚀 Training started...")

# Load datasets
fake = pd.read_csv('data/Fake.csv')
real = pd.read_csv('data/True.csv')

# Add labels
fake['label'] = 0
real['label'] = 1

# Combine datasets
data = pd.concat([fake, real], ignore_index=True)

# Shuffle data (fix randomness)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Combine title + text (IMPORTANT FIX)
data['content'] = data['title'] + " " + data['text']

# Keep only required columns
data = data[['content', 'label']]

# Clean text
data['content'] = data['content'].apply(clean_text)

# Debug check
print("\n📊 Label Distribution:\n", data['label'].value_counts())

# Split data
X = data['content']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF (strong)
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.7,
    ngram_range=(1,2),
    min_df=2
)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Model (stable + balanced)
model = LogisticRegression(
    class_weight='balanced',
    max_iter=300,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Report:\n", classification_report(y_test, y_pred))

# Save model
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("\n🎉 Model trained & saved successfully!")
