import pandas as pd
import re
import nltk
import spacy
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import time


def begin_cleaner():
    data = pd.read_csv("Restaurant reviews.csv")
    data = data.drop(columns=['7514'])
    data = data.drop(columns=['Pictures'])
    data = data.drop(columns=['Metadata'])
    data = data.drop(columns=['Time'])

    data["Review"] = data["Review"].apply(naturalize)
    data["Rating"] = pd.to_numeric(data["Rating"], errors='coerce')

    data = data.dropna(subset=["Rating"])

    X = data["Review"].values
    y = data["Rating"].values


    y = (y >= 3).astype(int)

    return X, y

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


def build_lstm_model(vectorize_layer, embedding_dim, max_length):
    model = Sequential([
        vectorize_layer,
        Embedding(input_dim=5000 + 1, output_dim=embedding_dim, mask_zero=True),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.5),
        Bidirectional(LSTM(32)),
        Dropout(0.5),
        Dense(24, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model


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

    max_tokens = 5000
    max_length = 100
    embedding_dim = 256

    vectorize_layer = TextVectorization(
        max_tokens=max_tokens,
        output_mode='int',
        output_sequence_length=max_length
    )

    vectorize_layer.adapt(X_train)

    model = build_lstm_model(vectorize_layer, embedding_dim, max_length)
    print("\n" + "=" * 50)
    print(model.summary())
    print("=" * 50 + "\n")

    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    start_timer = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    end_timer = time.time()
    print(f"\nTraining completed in {end_timer - start_timer:.2f} seconds")

    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

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
    print(confusion_matrix(y_test, y_pred))

    try:
        model.save('restaurant_lstm_model.keras')
        print("\nModel saved successfully as 'restaurant_lstm_model.keras'")
    except UnicodeEncodeError:
        print("\nUnicode error with .keras format, saving as .h5 instead")
        model.save('restaurant_lstm_model.h5')
        print("Model saved successfully as 'restaurant_lstm_model.h5'")

    return model, vectorize_layer, history


def predict_sentiment(review_text, model):
    cleaned = naturalize(review_text)

    pred = model.predict(tf.constant([cleaned]))[0][0]

    sentiment = "Positive" if pred > 0.5 else "Negative"
    confidence = pred if pred > 0.5 else 1 - pred

    return sentiment, confidence


if __name__ == "__main__":
    model, vectorize_layer, history = train_model()

    test_reviews = [
        "The food was absolutely amazing and the service was great!",
        "Terrible experience, cold food and rude staff.",
        "Average meal, nothing special but not bad either.",
        "Everything was extremely humbling. I was overwhelmed by the variety of food!",
        "bad",
        "hehe this is a fake review"
    ]

    print("\n=== Test Predictions ===")
    for review in test_reviews:
        sentiment, confidence = predict_sentiment(review, model)
        print(f"\nReview: {review}")
        print(f"Sentiment: {sentiment} (confidence: {confidence:.2%})")