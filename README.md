A Machine Learning-based NLP project that classifies news articles as Fake or Real using text classification techniques. The system processes news headlines and article content, converts textual data into numerical features using TF-IDF Vectorization, and trains a Logistic Regression model to detect misinformation.

Features Preprocesses news headlines and article text Removes punctuation, URLs, numbers, and unwanted text Converts text into numerical vectors using TF-IDF Trains a binary classification model Predicts whether a news article is fake or real Allows real-time custom text testing through user input

Tech Stack Python Pandas NumPy Scikit-learn Natural Language Processing (NLP) Machine Learning Workflow Data Collection using Fake and Real News datasets Data Cleaning and Text Preprocessing Feature Extraction using TF-IDF Model Training using Logistic Regression Model Evaluation using Accuracy and Classification Report Real-time Prediction on custom news input

Dataset: The project uses the Fake and Real News Dataset from Kaggle containing thousands of labeled news articles for training and testing.

Run Process Type in terminal:python Brain.py

Expected Output Fake News Samples: (23481, 4) True News Samples: (21417, 4) Combined Dataset Shape: (44898, 5) Accuracy: 0.98

Enter news headline/article: NASA confirms aliens landed in New York yesterday

Prediction: Fake News.
