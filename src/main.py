# src/main.py
import pandas as pd
from joblib import dump

from text_prep import Text_prep
from bow import BoW
from plsa import PLSA
from fisher import Fisher_Vectorizer
from svm import SVM_Classifier

from scipy.sparse import csr_matrix


def main():
    print("Starting the application...")

    # choice for PLSA or not
    choice = input(
        "Train with PLSA (smaller dataset, slower) or without PLSA (full dataset, faster)? [plsa/none]: "
    ).strip().lower()

    use_plsa = choice.startswith("p")

    if use_plsa:
        print(
            "\n[MODE] PLSA mode selected.\n"
            "The training size will be reduced to a smaller subset "
            "(approx. 10k/2k/2k), so that PLSA can run on this machine.\n"
        )
        max_train = 10000
        max_val   = 2000
        max_test  = 2000
        model_path = "models/fisher_svm_sentiment-plsa.joblib"
    else:
        print(
            "\n[MODE] Non-PLSA mode selected.\n"
            "The full training, validation and test datasets will be used.\n"
        )
        max_train = None
        max_val   = None
        max_test  = None
        model_path = "models/fisher_svm_sentiment.joblib"

    train_path = "data/train/sentiment140-train.csv"
    val_path   = "data/val/sentiment140-val.csv"
    test_path  = "data/test/sentiment140-test.csv"

    # -------- load data --------
    train_texts, y_train = load_file(train_path)
    val_texts,   y_val   = load_file(val_path)
    test_texts,  y_test  = load_file(test_path)

    if not train_texts or not val_texts or not test_texts:
        print("Failed to load one or more files. Exiting the application.")
        return

    print("File content loaded successfully...")

    # -------- limit datasets --------
    if use_plsa:
        train_texts = train_texts[:max_train]
        y_train     = y_train[:max_train]

        val_texts = val_texts[:max_val]
        y_val     = y_val[:max_val]

        test_texts = test_texts[:max_test]
        y_test     = y_test[:max_test]

        print(f"Datasets truncated for PLSA:")
        print(f"  Train: {len(train_texts)}")
        print(f"  Val:   {len(val_texts)}")
        print(f"  Test:  {len(test_texts)}")
    else:
        print(f"Using full datasets:")
        print(f"  Train: {len(train_texts)}")
        print(f"  Val:   {len(val_texts)}")
        print(f"  Test:  {len(test_texts)}")

    # -------- Preprocessing --------
    train_proc = Text_prep(train_texts)
    val_proc   = Text_prep(val_texts)
    test_proc  = Text_prep(test_texts)

    token_list  = train_proc.preprocess_list()
    val_tokens  = val_proc.preprocess_list()
    test_tokens = test_proc.preprocess_list()
    print("Tokens were created successfully...")

    # -------- create vocabular --------
    freq_dict = train_proc.count_tokens_frequency(token_list)
    print("Token frequency counted successfully...")

    sorted_tokens_list = train_proc.sort_tokens(freq_dict)
    print("Dictionary is sorted descending...")
    print(sorted_tokens_list[:10])

    bow = BoW()
    word2idx, idx2word = bow.build_vocabulary(sorted_tokens_list)
    print("Vocabulary built successfully...")

    # -------- BoW-Matrices --------
    X_train = bow.vectorize_list(token_list)
    X_val   = bow.vectorize_list(val_tokens)
    X_test  = bow.vectorize_list(test_tokens)
    print("BoW-Matrices created successfully...")
    print("X_train shape:", X_train.shape)
    print("X_val shape:  ", X_val.shape)
    print("X_test shape: ", X_test.shape)

    # -------- Fisher ((PLSA) + SVM) --------
    if use_plsa:
        # ----- PLSA -----
        plsa = PLSA(num_topics=20, iterations=15)
        topic_train = plsa.fit(X_train)
        topic_val   = plsa.transform(X_val)
        topic_test  = plsa.transform(X_test)
        print("PLSA topic features created successfully...")
        print("topic_train shape:", topic_train.shape)

        # transform to csr_matrices for Fisher_Vectorizer
        T_train = csr_matrix(topic_train.astype("float32"))
        T_val   = csr_matrix(topic_val.astype("float32"))
        T_test  = csr_matrix(topic_test.astype("float32"))

        # -------- Fisher --------
        fisher_vectorizer = Fisher_Vectorizer(alpha=1.0)
        Phi_train = fisher_vectorizer.fit_transform(T_train)
        Phi_val   = fisher_vectorizer.transform(T_val)
        Phi_test  = fisher_vectorizer.transform(T_test)
        print("Fisher-Features from PLSA created successfully...")

        # -------- SVM --------
        svm_classifier = SVM_Classifier(C=1.0, max_iter=2000)
        svm_classifier.train(Phi_train, y_train)
        print("SVM Classifier trained successfully...")

        model_bundle = {
            "bow": bow,
            "plsa": plsa,
            "fisher": fisher_vectorizer,
            "svm": svm_classifier.model,
            "idx2word": idx2word,
        }

    else:
        # -------- no plsa, using fisher from BoW --------
        fisher_vectorizer = Fisher_Vectorizer(alpha=1.0)
        Phi_train = fisher_vectorizer.fit_transform(X_train)
        Phi_val   = fisher_vectorizer.transform(X_val)
        Phi_test  = fisher_vectorizer.transform(X_test)
        print("Fisher-Features from BoW created successfully...")

        # -------- SVM --------
        svm_classifier = SVM_Classifier(C=1.0, max_iter=2000)
        svm_classifier.train(Phi_train, y_train)
        print("SVM Classifier trained successfully...")

        model_bundle = {
            "bow": bow,
            "fisher": fisher_vectorizer,
            "svm": svm_classifier.model,
            "idx2word": idx2word,
        }

    print("Train labels:", len(y_train),
          "Val labels:", len(y_val),
          "Test labels:", len(y_test))

    val_acc  = svm_classifier.evaluate(Phi_val,  y_val)
    test_acc = svm_classifier.evaluate(Phi_test, y_test)
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy:       {test_acc:.4f}")

    # -------- save model --------
    dump(model_bundle, model_path)
    print(f"Model saved to {model_path}")


def load_file(file_path):
    """
    loads csv file with columns 'polarity' and 'text'.

    returns: (texts, labels)
    """
    try:
        df = pd.read_csv(file_path, encoding='latin-1')

        if "polarity" not in df.columns or "text" not in df.columns:
            raise ValueError(f"'polarity' or 'text' column is missing in {file_path}")

        df["text"] = df["text"].fillna("").astype(str)
        df["polarity"] = df["polarity"].astype(int)

        texts  = df["text"].tolist()
        labels = df["polarity"].tolist()
        return texts, labels

    except Exception as e:
        print(f"An error occurred while loading the file '{file_path}': {e}")
        return [], []


if __name__ == "__main__":
    main()




# next Step for wear-app:
# 0. remove plsa completely in new branch wear-app-model
# 1. fetch new data from jigsaw dataset with 0=toxic/bad and 1=neutral/ok comments (reddit/twitter/insta, usw. geht nicht einfach Jigsaw Toxic Comments nutzen)
# 2. retrain model with new categories
# 3. test model in wear-app and report toxic/bad categorized comments