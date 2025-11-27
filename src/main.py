# src/main.py
import pandas as pd
from text_prep import Text_prep

def main():
    '''
    main Pipeline to start text sentiment analysis.
    '''

    print("Starting the application...")
    text_list = load_file("data/train/sentiment140-train.csv")
    test_list = load_file("data/test/sentiment140-test.csv")
    val_list = load_file("data/val/sentiment140-val.csv")

    if not text_list or not test_list or not val_list:
        print(f"Failed to load one or more files. Exiting the application.")
        return

    print("File content loaded successfully...")
    text_processor = Text_prep(text_list)
    test_processor = Text_prep(test_list)
    val_processor = Text_prep(val_list)
    token_list = text_processor.preprocess_list()
    test_tokens = test_processor.preprocess_list()
    val_tokens = val_processor.preprocess_list()
    print("tokens were created successfully...")
    dict = text_processor.count_tokens_frequency(token_list)
    print("Token frequency counted successfully...")
    sorted_tokens_list = text_processor.sort_tokens(dict)
    print("Dictionary is sorted descending...")
    index2word = text_processor.index_to_word(sorted_tokens_list)
    word2index = text_processor.word_to_index(sorted_tokens_list)
    print("word/index lists created successfully...")
    print(sorted_tokens_list[:10])



def load_file(file_path):
    '''
    loads and prints the content of a file given its path.
    '''
    try:
        df = pd.read_csv(file_path,
                    names=['polarity', 'text'],
                    encoding='latin-1')

        content = df["text"].fillna("").astype(str).tolist()
        return content
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return []




if __name__ == "__main__":
    main()