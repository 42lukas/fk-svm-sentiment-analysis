# src/main.py
import pandas as pd
from text_prep import Text_prep

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def main():
    prepare_data()
    
    # ----------------------------------------------------
    # 1. Beispiel-Daten (hier nur Dummy – du ersetzt sie)
    # ----------------------------------------------------
    texts = [
        "I love this movie", "This was terrible",
        "Absolutely fantastic!", "Worst thing ever",
        "I enjoyed it", "I hate it"
    ]
    labels = [1, 0, 1, 0, 1, 0]   # 1=positiv, 0=negativ

    # ----------------------------------------------------
    # 2. Preprocessing + Bag-of-Words
    # ----------------------------------------------------
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    X = X.toarray()

    vocab_size = X.shape[1]
    num_docs = X.shape[0]

    # ----------------------------------------------------
    # 3. Mini-PLSA Implementierung
    # ----------------------------------------------------
    def plsa(X, num_topics=2, iterations=30):
        num_docs, num_words = X.shape
        
        # Zufällige Initialisierung
        P_z = np.random.rand(num_topics)
        P_w_z = np.random.rand(num_topics, num_words)
        P_z_d = np.random.rand(num_docs, num_topics)

        # Normalisieren
        P_z /= P_z.sum()
        P_w_z /= P_w_z.sum(axis=1, keepdims=True)
        P_z_d /= P_z_d.sum(axis=1, keepdims=True)

        # EM Algorithmus
        for _ in range(iterations):
            # E-Step
            P_z_dw = np.zeros((num_docs, num_words, num_topics))
            for d in range(num_docs):
                for w in range(num_words):
                    denom = np.sum(P_z_d[d] * P_w_z[:, w])
                    if denom > 0:
                        P_z_dw[d, w] = (P_z_d[d] * P_w_z[:, w]) / denom

            # M-Step
            # Update P(w|z)
            for z in range(num_topics):
                for w in range(num_words):
                    P_w_z[z, w] = np.sum(X[:, w] * P_z_dw[:, w, z])
                P_w_z[z] /= np.sum(P_w_z[z])

            # Update P(z|d)
            for d in range(num_docs):
                for z in range(num_topics):
                    P_z_d[d, z] = np.sum(X[d] * P_z_dw[d, :, z])
                P_z_d[d] /= np.sum(P_z_d[d])

        return P_z_d  # Dokument-Topic-Vektoren

    # ----------------------------------------------------
    # 4. Topic-Features (entspricht Z-Vektor im Paper)
    # ----------------------------------------------------
    topic_features = plsa(X, num_topics=2)

    # ----------------------------------------------------
    # 5. SVM Training mit den PLSA-Features
    # ----------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(topic_features, labels, test_size=0.3)

    svm = SVC(kernel="linear")
    svm.fit(X_train, y_train)

    preds = svm.predict(X_test)

    print(classification_report(y_test, preds))
    
    

def prepare_data():
    '''
    main Pipeline to start text sentiment analysis.py
    '''

    print("Starting the application...")
    text_list = load_file("data/train/sentiment140-train.csv")
    print("File content loaded successfully...")
    text_processor = Text_prep(text_list)
    token_list = text_processor.preprocess_list()
    print("tokens were created successfully...")
    dict = text_processor.count_tokens_frequency(token_list)
    print("Token frequency counted successfully...")
    sorted_tokens_list = text_processor.sort_tokens(dict)
    print("Dictionary is sorted descending...")
    index2word = text_processor.index_to_word(sorted_tokens_list)
    word2index = text_processor.word_to_index(sorted_tokens_list)
    print("word/index lists created successfully...")
    print(sorted_tokens_list[:100])


def load_file(file_path):
    '''
    loads and prints the content of a file given its path.
    '''

    df = pd.read_csv(file_path,
                names=['polarity', 'text'],
                encoding='latin-1')

    content = df["text"].fillna("").astype(str).tolist()
    return content




if __name__ == "__main__":
    main()