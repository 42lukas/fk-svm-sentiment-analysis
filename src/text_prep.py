# src/text_prep.py
import string

class Text_prep():
    def __init__(self, text):
        self.text = text

    def to_lowercase(self, text):
        return text.lower()
    
    def remove_urls(self, text):
        return ' '.join(word for word in text.split() if not word.startswith('http') and not word.startswith('www') and not word.startswith('https') and not word.endswith('.com'))

    def remove_punctuation(self, text):
        return text.translate(str.maketrans('', '', string.punctuation))

    def tokenize(self, text):
        return text.split()

    def preprocess(self, text):
        text = self.to_lowercase(text)
        text = self.remove_urls(text)
        text = self.remove_punctuation(text)
        return self.tokenize(text)
    
    def preprocess_list(self):
        all_tokens = []
        for tweet in self.text:
            tokens = self.preprocess(tweet)
            all_tokens.append(tokens)
        return all_tokens
    
    def count_tokens_frequency(self, token_list):
        frequency = {}
        for tweet in token_list:
            for token in tweet:
                frequency[token] = frequency.get(token, 0) + 1
        return frequency
    
    def sort_tokens(self, dict):
        V = 10000

        sorted_tokens = sorted(dict.items(), key=lambda item: item[1], reverse=True)
        sorted_tokens = sorted_tokens[:V] # keep only the top V tokens
        return sorted_tokens

