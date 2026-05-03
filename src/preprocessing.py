import pandas as pd

train = pd.read_csv("data/raw/train.csv")
dev = pd.read_csv("data/raw/dev.csv")
test = pd.read_csv("data/raw/test.csv")

print("Train:", train.shape)
print("Dev:", dev.shape)
print("Test:", test.shape)

print("\nColumns:")
print(train.columns)

import pandas as pd
from sklearn.model_selection import train_test_split

def clean_text(text):
    text=str(text)
    text=text.lower()
    text=text.strip()
    return text

def create_processed_splits():   # Function to create processed splits from the raw train.csv
    df = pd.read_csv("data/raw/train.csv")

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    train_df, temp_df = train_test_split(  #split between train and temp 
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["answer"]
    )

    dev_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["answer"]  #equal split of answers between all files
    )

    train_df.to_csv("data/processed/train.csv", index=False)
    dev_df.to_csv("data/processed/dev.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)

    print("Processed splits created")
    print("Train:", train_df.shape)
    print("Dev:", dev_df.shape)
    print("Test:", test_df.shape)

def load_processed_data():   #loads processed data
    train = pd.read_csv("data/processed/train.csv")
    dev = pd.read_csv("data/processed/dev.csv")
    test = pd.read_csv("data/processed/test.csv")

    return train, dev, test

def prepare_text_columns(df):
    df["combined_text"]= (
        df["article"].apply(clean_text) + " " +
        df["question"].apply(clean_text) + " " +
        df["A"].apply(clean_text) + " " +
        df["B"].apply(clean_text) + " " +  
        df["C"].apply(clean_text) + " " +
        df["D"].apply(clean_text)
    )

    return df
if __name__ == "__main__":
    create_processed_splits()
    train, dev, test = load_processed_data()
    train = prepare_text_columns(train)
    print("Train data after preprocessing:", train.shape)
    print(train[["combined_text", "answer"]].head())
