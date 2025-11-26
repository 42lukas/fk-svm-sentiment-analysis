# src/format_dataset/clean_dataset.py

# this file is to split the Sentiment140 dataset
# into test training and validation sets

import pandas as pd

df = pd.read_csv(
    "/Users/lukaskarsten/Desktop/Informatik/Repos/fk-svm-sentiment-analysis/data/sentiment140-subset.csv",
    encoding='latin-1',
    header=0,                      
    names=['polarity', 'text']    
)
print(df.head()) # print the first 5 rows to see the structure
print(df.polarity.value_counts()) # check the distribution of classes

# split the dataset into training (80%), validation (10%) and test (10%)
train_df = df.sample(frac=0.8, random_state=42) # random state for reproducibility => meaning if i would run this again, i would get the same split
temp_df = df.drop(train_df.index) # remaining 20%
val_df = temp_df.sample(frac=0.5, random_state=42) # getting half of the remaining 20% => 10%
test_df = temp_df.drop(val_df.index) # remaining 10%
print(f"Training set size: {len(train_df)} | {train_df.polarity.value_counts()}")
print(f"Validation set size: {len(val_df)} | {val_df.polarity.value_counts()}")
print(f"Test set size: {len(test_df)} | {test_df.polarity.value_counts()}")

# save the datasets
train_df.to_csv("data/train/sentiment140-train.csv", index=False)
val_df.to_csv("data/valid/sentiment140-val.csv", index=False)
test_df.to_csv("data/test/sentiment140-test.csv", index=False)