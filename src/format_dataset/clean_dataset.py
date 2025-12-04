# src/format_dataset/clean_dataset.py

# this file is to clean the Sentiment140 dataset after downloading
# it removes unnecessary columns

import pandas as pd
import re
import html

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # HTML-Entities decodieren (z.B. &lt; -> <)
    text = html.unescape(text)

    # remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # remove @usernames
    text = re.sub(r"@\w+", " ", text)

    # remove non ASCII characters ((Â, ã…, kyrillisch, ...)
    text = text.encode("ascii", "ignore").decode("ascii")

    # to lower case
    text = text.lower()

    # transfom multiple spaces to single space and strip leading/trailing spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


df = pd.read_csv("/Users/lukaskarsten/Desktop/Informatik/Repos/fk-svm-sentiment-analysis/data/training.1600000.processed.noemoticon.csv",
                names=['polarity', 'id', 'date', 'query', 'user', 'text'],
                encoding='latin-1')
print(df.head()) # print the first 5 rows to see the structure
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

# save the cleaned dataset
df.to_csv("data/sentiment140-subset.csv", index=False)