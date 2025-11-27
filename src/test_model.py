# src/test_model.py
import sys
from joblib import load
from text_prep import Text_prep

MODEL_PATH = "models/fisher_svm_sentiment.joblib"


def load_model(model_path=MODEL_PATH):
    """
    Lädt das gespeicherte Modell-Bundle:
    - bow: BoW-Instanz mit Vokabular (word2idx, idx2word)
    - fisher: Fisher_Vectorizer mit theta und I_diag
    - svm: trainiertes LinearSVC-Modell
    """
    try:
        bundle = load(model_path)
    except Exception as e:
        print(f"Error loading model from '{model_path}': {e}")
        sys.exit(1)

    bow = bundle["bow"]
    fisher = bundle["fisher"]
    svm_model = bundle["svm"]

    return bow, fisher, svm_model


def predict_text(bow, fisher, svm_model, text: str) -> int:
    """
    Nimmt einen Roh-Text, wendet die gleiche Preprocessing-Pipeline an
    wie beim Training, und gibt das vorhergesagte Label (0/1) zurück.
    """

    # 1) Preprocessing + Tokenizing (Text_prep wie im Training benutzen)
    processor = Text_prep([text])
    token_list = processor.preprocess_list()  # Liste mit genau einem Eintrag
    tokens = token_list[0]

    # 2) BoW-Vektor für diesen einen Text
    X_user = bow.vectorize_list([tokens])  # csr_matrix mit Shape (1, vocab_size)

    # 3) Fisher-Features
    Phi_user = fisher.transform(X_user)    # csr_matrix (1, vocab_size)

    # 4) SVM-Prediction
    pred = svm_model.predict(Phi_user)[0]  # einzelnes Label (0 oder 1)

    return pred


def interactive_loop():
    """
    Lädt das Modell und startet eine interaktive Eingabe-Schleife,
    in der der User Texte eingibt und Sentiment-Vorhersagen bekommt.
    """
    print(f"Lade Modell aus '{MODEL_PATH}' ...")
    bow, fisher, svm_model = load_model()
    print("Modell erfolgreich geladen.\n")

    print("Interaktiver Sentiment-Test.")
    print("Gib einen Text ein und drücke Enter.")
    print("Tippe 'quit' oder 'exit', um zu beenden.\n")

    while True:
        try:
            user_text = input("Text: ")
        except (EOFError, KeyboardInterrupt):
            print("\nBeende.")
            break

        if user_text.strip().lower() in ("quit", "exit"):
            print("Beende.")
            break

        if not user_text.strip():
            print("Bitte keinen leeren Text eingeben.\n")
            continue

        pred = predict_text(bow, fisher, svm_model, user_text)
        label = "NEGATIV" if pred == 0 else "POSITIV"

        print(f"-> Prediction: {label}\n")


if __name__ == "__main__":
    interactive_loop()