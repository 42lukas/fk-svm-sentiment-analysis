# src/format_dataset/clean_dataset.py

# this file is to clean the Sentiment140 dataset after downloading
# it removes unnecessary columns

import pandas as pd

df = pd.read_csv("/Users/lukaskarsten/Desktop/Informatik/Repos/fk-svm-sentiment-analysis/data/training.1600000.processed.noemoticon.csv",
                names=['polarity', 'id', 'date', 'query', 'user', 'text'],
                encoding='latin-1')
print(df.head()) # print the first 5 rows to see the structure
print(df.polarity.value_counts()) # check the distribution of classes

# keep only the necessary columns: polarity and text
df = df.drop(columns=['id', 'date', 'query', 'user'])
print(df.head())

# switch polarity values from 0,4 to 0,1
df.polarity = df.polarity.replace({0: 0, 4: 1})
print(df.polarity.value_counts())

# save the cleaned dataset
df.to_csv("data/sentiment140-subset.csv", index=False)