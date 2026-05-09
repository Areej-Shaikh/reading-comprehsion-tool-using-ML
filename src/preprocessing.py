import pandas as pd
import string
import os
from sklearn.model_selection import train_test_split

#remoce punctuation from text
def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


def clean_text(text):
    text = str(text)
    text = text.lower()
    text = remove_punctuation(text)
    text = text.strip()
    return text

#combine all text columns into a combined column
def prepare_text_columns(df):
    df["combined_text"] = (
        df["article"].apply(clean_text) + " " +
        df["question"].apply(clean_text) + " " +
        df["A"].apply(clean_text) + " " +
        df["B"].apply(clean_text) + " " +
        df["C"].apply(clean_text) + " " +
        df["D"].apply(clean_text)
    )
    return df


def create_processed_splits():
    df = pd.read_csv("data/raw/train.csv")  

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    train_df, temp_df = train_test_split(
    df,
    test_size=0.20,    #80% train, 10% dev, 10% test
    random_state=42,
    stratify=df["answer"]
)

    dev_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["answer"]
    )

    train_df = prepare_text_columns(train_df)
    dev_df = prepare_text_columns(dev_df)
    test_df = prepare_text_columns(test_df)

    os.makedirs("data/processed", exist_ok=True)
#save the processed splits
    train_df.to_csv("data/processed/train.csv", index=False)
    dev_df.to_csv("data/processed/dev.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)

    print("Processed splits created")
    print("Train:", train_df.shape)
    print("Dev:  ", dev_df.shape)
    print("Test: ", test_df.shape)

#load to use in training
def load_processed_data():
    train = pd.read_csv("data/processed/train.csv")
    dev   = pd.read_csv("data/processed/dev.csv")
    test  = pd.read_csv("data/processed/test.csv")
    return train, dev, test


if __name__ == "__main__":
    create_processed_splits()
    train, dev, test = load_processed_data()
    print("\nColumns:", list(train.columns))
    print("\nTrain sample:")
    print(train[["combined_text", "answer"]].head())