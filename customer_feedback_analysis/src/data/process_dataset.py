import os
import pandas as pd

from src.data.preprocess import clean_batch


def process_dataset():
    print("Script Started")

    input_path = "data/raw/amazon_books_Data.csv"
    output_path = "data/processed/cleaned_reviews.csv"

    df = pd.read_csv(input_path)

    print(f"Dataset Loaded: {df.shape}")

    # Select up to 50,000 rows
    sample_size = min(50000, len(df))
    df = df.sample(n=sample_size, random_state=42)

    print(f"Sample Selected: {len(df)} rows")

    # Use existing sentiment labels
    df["sentiment"] = df["Sentiment_books"].astype(str).str.lower().str.strip()

    print("Sentiment Created")
    print(df["sentiment"].value_counts())

    # Clean review text
    texts = df["review_body"].fillna("").astype(str).str.lower()

    texts = texts.str.replace(r"http\S+", "", regex=True)
    texts = texts.str.replace(r"[^a-zA-Z ]", "", regex=True)
    texts = texts.str.replace(r"\s+", " ", regex=True).str.strip()

    # Tokenization and lemmatization
    df["clean_review"] = clean_batch(texts.tolist())

    print("Cleaning Completed")

    os.makedirs("data/processed", exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"File Saved: {output_path}")
    print(f"Final Dataset Shape: {df.shape}")


if __name__ == "__main__":
    process_dataset()