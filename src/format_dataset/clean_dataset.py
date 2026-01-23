# src/format_dataset/clean_dataset.py

# this file is to clean the Sentiment140 dataset after downloading
# it removes unnecessary columns and cleans the text data

import pandas as pd
import re
import html

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # HTML-Entities decodieren (z.B. &lt; -> <)
    text = html.unescape(text)

    # URLs entfernen
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # @usernames entfernen
    text = re.sub(r"@\w+", " ", text)

    # Nicht-ASCII-Zeichen entfernen (Â, ã…, kyrillisch, usw.)
    text = text.encode("ascii", "ignore").decode("ascii")

    # Lowercase
    text = text.lower()

    # Mehrfache Whitespaces zu einem machen
    text = re.sub(r"\s+", " ", text).strip()

    return text


df = pd.read_csv("/Users/lukaskarsten/Desktop/Informatik/Repos/fk-svm-sentiment-analysis/data/training.1600000.processed.noemoticon.csv",
                names=['polarity', 'id', 'date', 'query', 'user', 'text'],
                encoding='latin-1')
print(df.head()) # print the first 5 rows to see the structure of the dataset
print(df.polarity.value_counts()) # check the distribution of classes

# keep only the necessary columns: polarity and text
df = df.drop(columns=['id', 'date', 'query', 'user'])
print(df.head())

# clean the text column
df["text"] = df["text"].apply(clean_text)
print("Text cleaning done!")
print(df.head())

# switch polarity values from 0,4 to 0,1
df.polarity = df.polarity.replace({0: 0, 4: 1})
print(df.polarity.value_counts())

# save the cleaned dataset to a new csv file
df.to_csv("data/sentiment140-subset.csv", index=False)