# src/fisher.py

import numpy as np
from scipy.sparse import csr_matrix

class Fisher_Vectorizer:
    def __init__(self, alpha=1.0):
        """
        alpha: Parameter für theta-Schätzung
        """
        self.alpha = alpha      # Parameter um division durch Null zu vermeiden
        self.theta = None       # Sprachmodell-Parameter (Vektor Länge V)
        self.I_diag = None      # Diagonale der Fisher-Informationsmatrix (Vektor Länge V)

    def fit_theta(self, X_train: csr_matrix):
        """
        Schätzt theta aus der Trainings-Bag-of-Words-Matrix.
        X_train: csr_matrix mit Shape (num_docs, vocab_size)
        """
        # 1) Globale Counts pro Wort aufaddieren (über alle Dokumente)
        # X_train.sum(axis=0) gibt für jede Spalte i die Summe über alle Zeilen (Tweets)
        c = np.array(X_train.sum(axis=0)).ravel()  # c[i] = wie oft Wort i im gesamten Trainingsset vorkommt

        # 2) Gesamte Anzahl aller Wörter im Trainingsset
        C = c.sum()

        # 3) Vokabulargröße
        V = c.shape[0]
        
        # 4) Theta mit alpha um division durch Null zu vermeiden
        self.theta = (c + self.alpha) / (C + self.alpha * V)

    def fisher_scores(self, X: csr_matrix) -> csr_matrix:
        # Wenn ein Wort in einem Tweet häufig vorkommt, aber im Modell selten ist (kleine θ (theta)), bekommt man einen hohen Score.
        """
        Berechnet Fisher-Scores U_x = n_i / theta_i für alle Dokumente in X.
        X: csr_matrix (num_docs x vocab_size)
        Rückgabe: csr_matrix gleicher Shape
        """
        if self.theta is None:
            raise ValueError("theta not fitted. Call fit_theta() first.")

        # 1) 1 / theta berechnen (Spalten-Skalierung)
        inv_theta = 1.0 / self.theta  # shape (V,)

        # 2) Jede Spalte j von X mit inv_theta[j] multiplizieren
        # X.multiply(...) skaliert spaltenweise, wenn inv_theta ein 1D-Array ist.
        U = X.multiply(inv_theta)

        return U # Rückgabe der Fisher-Scores
    
    def fit_I_diag(self, U_train: csr_matrix):
        # Erklärung:
        # •	Für jedes Wort (jede Dimension i) wird geschaut:
        #   Wie groß sind im Schnitt die quadratischen Fisher-Scores über alle Trainingstexte?
	    # •	Wenn ein Wort sehr stark schwankt, ist I_{ii} größer → es trägt mehr Information.
	    # •	Bei der Normalisierung teilen wir durch \sqrt{I_{ii}}, damit das Ganze „gleich gewichtet“ wird.
        """
        Schätzt die diagonale Fisher-Informationsmatrix I_diag aus den
        Fisher-Scores der Trainingsdaten.
        U_train: csr_matrix (num_docs x vocab_size)
        """
        # 1) Quadrate der Fisher-Scores
        U2 = U_train.multiply(U_train)  # elementweise Quadrat

        # 2) Summe über alle Dokumente -> für jede Spalte ein Wert
        s = np.array(U2.sum(axis=0)).ravel()  # shape (V,)

        # 3) Mittelwert pro Dimension (Dokumentenanzahl M)
        M = U_train.shape[0]
        I_diag = s / M

        # 4) Numerisch stabil machen: keine 0-Werte erlauben
        I_diag[I_diag == 0] = 1e-8

        self.I_diag = I_diag

    def transform(self, X: csr_matrix) -> csr_matrix:
        """
        Wandelt eine BoW-Matrix X in Fisher-Features Phi(X) um.
        X: csr_matrix (num_docs x vocab_size)
        Rückgabe: csr_matrix (num_docs x vocab_size) – Fisher-Features
        """
        if self.theta is None or self.I_diag is None:
            raise ValueError("theta or I_diag not fitted. Call fit_theta() and fit_I_diag() first.")

        # 1) Fisher-Scores für X
        U = self.fisher_scores(X)

        # 2) 1 / sqrt(I_diag) berechnen
        inv_sqrt_I = 1.0 / np.sqrt(self.I_diag)  # shape (V,)

        # 3) Jede Spalte j in U mit inv_sqrt_I[j] skalieren -> Phi
        Phi = U.multiply(inv_sqrt_I)

        return Phi
    

    def fit(self, X_train: csr_matrix):
        """
        Lernt theta und I_diag aus der Trainingsmatrix.
        """
        self.fit_theta(X_train)
        U_train = self.fisher_scores(X_train)
        self.fit_I_diag(U_train)

    def fit_transform(self, X_train: csr_matrix) -> csr_matrix:
        """
        Kombination aus fit() und transform() für das Training.
        """
        self.fit(X_train)
        return self.transform(X_train)