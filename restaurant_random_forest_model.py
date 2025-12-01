import pandas as pd
import re
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle
import time


def naturalize(data_to_naturalize: str):
    if not isinstance(data_to_naturalize, str):
        data_to_naturalize = ""
    cleaned_data = data_to_naturalize.lower()
    cleaned_data = re.sub(r"\n", " ", cleaned_data)
    cleaned_data = re.sub(r"\s+", " ", cleaned_data)
    cleaned_data = cleaned_data.strip()

    tokens = word_tokenize(cleaned_data)
    filtered_tokens_alpha = [word for word in tokens if word.isalpha()]

    return " ".join(filtered_tokens_alpha)


def begin_cleaner():
    data = pd.read_csv("Restaurant reviews.csv")

    data["Review"] = data["Review"].apply(naturalize)

    data["Rating"] = pd.to_numeric(data["Rating"], errors='coerce')

    data = data.dropna(subset=["Rating"])

    X = data["Review"].values

    y = data["Rating"].values

    y = (y >= 3).astype(int)

    return X, y


def train_model():
    X, y = begin_cleaner()

    print("=== Class Distribution ===")
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        label_name = "Positive" if label == 1 else "Negative"
        print(f"{label_name}: {count} ({count / len(y) * 100:.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n=== Vectorizing Text ===")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"Feature matrix shape: {X_train_tfidf.shape}")

    print("\n=== Training Random Forest ===")
    start_timer = time.time()
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=70,
        min_samples_split=3,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    model.fit(X_train_tfidf, y_train)

    end_timer = time.time()
    print(f"\nTraining completed in {end_timer - start_timer:.2f} seconds")

    print("\n=== Generating Predictions ===")
    y_pred = model.predict(X_test_tfidf)

    print("\n=== Prediction Distribution ===")
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    for label, count in zip(unique_pred, counts_pred):
        label_name = "Positive" if label == 1 else "Negative"
        print(f"Predicted {label_name}: {count} ({count / len(y_pred) * 100:.1f}%)")

    print("\n=== Model Evaluation ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"\nTrue Negatives: {cm[0, 0]}, False Positives: {cm[0, 1]}")
    print(f"False Negatives: {cm[1, 0]}, True Positives: {cm[1, 1]}")

    print("\n=== Top 20 Most Important Features ===")
    feature_names = vectorizer.get_feature_names_out()
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]

    for i, idx in enumerate(indices, 1):
        print(f"{i}. {feature_names[idx]}: {importances[idx]:.4f}")

    print("\n=== Saving Model ===")
    with open('random_forest_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print("Model saved as 'random_forest_model.pkl'")
    print("Vectorizer saved as 'tfidf_vectorizer.pkl'")

    return model, vectorizer


def predict_sentiment(review_text, model, vectorizer):
    cleaned = naturalize(review_text)

    review_tfidf = vectorizer.transform([cleaned])

    pred = model.predict(review_tfidf)[0]
    pred_proba = model.predict_proba(review_tfidf)[0]

    sentiment = "Positive" if pred == 1 else "Negative"
    confidence = pred_proba[pred]

    return sentiment, confidence


def load_model():
    with open('random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


if __name__ == "__main__":
    model, vectorizer = train_model()

    test_reviews = [
        "The food was absolutely amazing and the service was great!",
        "Terrible experience, cold food and rude staff.",
        "Average meal, nothing special but not bad either.",
        "Best restaurant in town! Will definitely come back.",
        "Overpriced and underwhelming. Not worth the money."
    ]

    print("\n" + "=" * 60)
    print("=== Test Predictions ===")
    print("=" * 60)
    for review in test_reviews:
        sentiment, confidence = predict_sentiment(review, model, vectorizer)
        print(f"\nReview: {review}")
        print(f"Sentiment: {sentiment} (confidence: {confidence:.2%})")