# src/test_model.py
import sys
from joblib import load
from text_prep import Text_prep
from scipy.sparse import csr_matrix


def load_model(model_path):
    """
    Loads a saved model bundle.

    The bundle may contain:
      - 'bow': BoW instance with vocabulary (word2idx, idx2word)
      - 'fisher': Fisher_Vectorizer with theta and I_diag
      - 'svm': trained SVM model
      - optionally 'plsa': trained PLSA-like model

    Returns:
      bow, fisher, svm_model, plsa_or_none
    """
    try:
        bundle = load(model_path)
    except Exception as e:
        print(f"Error loading model from '{model_path}': {e}")
        sys.exit(1)

    if "bow" not in bundle or "fisher" not in bundle or "svm" not in bundle:
        print("Loaded bundle does not contain required keys ('bow', 'fisher', 'svm').")
        sys.exit(1)

    bow = bundle["bow"]
    fisher = bundle["fisher"]
    svm_model = bundle["svm"]
    plsa = bundle.get("plsa", None)  # may be None for BoW-only model

    return bow, fisher, svm_model, plsa


def predict_text(bow, fisher, svm_model, text: str, plsa=None) -> int:
    """
    Takes a raw text, applies the same preprocessing pipeline as in training,
    and returns the predicted label (0/1).

    If 'plsa' is not None, the pipeline is:
      Text -> Preprocessing/Tokenization
           -> BoW vector
           -> PLSA: P(z|d) for this text
           -> Fisher features in topic space
           -> SVM prediction

    If 'plsa' is None, the pipeline is:
      Text -> Preprocessing/Tokenization
           -> BoW vector
           -> Fisher features from BoW
           -> SVM prediction
    """

    # 1) Preprocessing + Tokenization
    processor = Text_prep([text])
    token_list = processor.preprocess_list()  # list with exactly one entry
    tokens = token_list[0]

    # 2) BoW vector for this single text
    X_user = bow.vectorize_list([tokens])  # csr_matrix with shape (1, vocab_size)

    # 3) Either go through PLSA or directly to Fisher
    if plsa is not None:
        # PLSA: doc-topic distribution P(z|d)
        topic_user = plsa.transform(X_user)            # np.array with shape (1, num_topics)
        T_user = csr_matrix(topic_user.astype("float32"))  # (1, num_topics)
        Phi_user = fisher.transform(T_user)           # csr_matrix (1, num_topics)
    else:
        # No PLSA: Fisher directly on BoW
        Phi_user = fisher.transform(X_user)           # csr_matrix (1, vocab_size)

    # 4) SVM prediction
    pred = svm_model.predict(Phi_user)[0]            # single label (0 or 1)

    return pred


def interactive_loop():
    """
    Loads a saved model and enters an interactive loop
    to predict sentiment on user-input texts.
    """
    choice = input("Which model would you like to load? (plsa/bow): ").strip().lower()
    if choice not in ("plsa", "bow"):
        print("Please enter 'plsa' or 'bow', nothing else!")
        return

    if choice == "plsa":
        model_path = "models/fisher_svm_sentiment-plsa.joblib"
    else:
        model_path = "models/fisher_svm_sentiment.joblib"

    print(f"Load model from '{model_path}' ...")
    bow, fisher, svm_model, plsa = load_model(model_path=model_path)
    print("Model was loaded successfully.\n")

    if plsa is not None:
        print("Loaded PLSA-based FK-SVM model.")
    else:
        print("Loaded BoW + Fisher + SVM model (no PLSA).")

    print("\nInteractive sentiment test.")
    print("Enter a text and press 'Enter' to proceed.")
    print("Enter 'quit' or 'exit' to end the test.\n")

    while True:
        try:
            user_text = input("Text: ")
        except (EOFError, KeyboardInterrupt):
            print("\nEnd.")
            break

        if user_text.strip().lower() in ("quit", "exit"):
            print("End.")
            break

        if not user_text.strip():
            print("Please don't enter an empty text.\n")
            continue

        pred = predict_text(bow, fisher, svm_model, user_text, plsa=plsa)
        label = "NEGATIV" if pred == 0 else "POSITIV"

        print(f"-> Prediction: {label}\n")


if __name__ == "__main__":
    interactive_loop()