import os
import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split
)

from sklearn.feature_extraction.text import (
    TfidfVectorizer
)

from sklearn.linear_model import (
    LogisticRegression
)


df = pd.read_csv(
    "data/processed/cleaned_reviews.csv"
)


# Remove missing values
df = df.dropna(
    subset=["clean_review", "sentiment"]
)


# Check available sentiment classes
print("Sentiment Distribution:")
print(df["sentiment"].value_counts())


X = df["clean_review"]

y = df["sentiment"]


# Stop if there is only one sentiment class
if y.nunique() < 2:
    raise ValueError(
        "Training cannot continue because the dataset contains only one "
        "sentiment class. Please check the processed dataset."
    )


vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2)
)


X = vectorizer.fit_transform(X)


X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
)


model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)


model.fit(
    X_train,
    y_train
)


os.makedirs(
    "models",
    exist_ok=True
)


joblib.dump(
    model,
    "models/sentiment_model.pkl"
)


joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)


print("Training Completed")