# src/fisher.py

import numpy as np
from scipy.sparse import csr_matrix

class Fisher_Vectorizer:
    def __init__(self, alpha=1.0):
        """
        alpha: parameter for Laplace smoothing
        """
        self.alpha = alpha 
        self.theta = None
        self.I_diag = None

    def fit_theta(self, X_train: csr_matrix):
        """
        guesses theta from the training BoW matrix.
        X_train: csr_matrix (num_docs x vocab_size)
        ändert self.theta
        """

        c = np.array(X_train.sum(axis=0)).ravel()  # c[i] = how many times word i appears in training set
        C = c.sum()
        V = c.shape[0] # vocabulary size
        self.theta = ((c + self.alpha) / (C + self.alpha * V)).astype(np.float32)

    def fisher_scores(self, X: csr_matrix) -> csr_matrix:
        """
        calculates the Fisher scores U for a BoW matrix X.
        X: csr_matrix (num_docs x vocab_size)
        returns: csr_matrix (num_docs x vocab_size) - Fisher scores U
        """
        if self.theta is None:
            raise ValueError("theta not fitted. Call fit_theta() first.")

        inv_theta = 1.0 / self.theta
        U = X.multiply(inv_theta)

        return U
    
    def fit_I_diag(self, U_train: csr_matrix):
        # Erklärung:
        # •	Für jedes Wort (jede Dimension i) wird geschaut:
        #   Wie groß sind im Schnitt die quadratischen Fisher-Scores über alle Trainingstexte?
	    # •	Wenn ein Wort sehr stark schwankt, ist I_{ii} größer → es trägt mehr Information.
	    # •	Bei der Normalisierung teilen wir durch \sqrt{I_{ii}}, damit das Ganze „gleich gewichtet“ wird.
        """
        guesses the diagonal of the Fisher Information Matrix I from the Fisher scores U_train.
        U_train: csr_matrix (num_docs x vocab_size)
        changes self.I_diag
        """
        U2 = U_train.multiply(U_train)
        s = np.array(U2.sum(axis=0)).ravel()

        M = U_train.shape[0]
        I_diag = s / M

        I_diag[I_diag == 0] = 1e-8
        self.I_diag = I_diag.astype(np.float32)

    def transform(self, X: csr_matrix) -> csr_matrix:
        """
        transforms a BoW matrix X into Fisher features Phi.
        X: csr_matrix (num_docs x vocab_size)
        returns: csr_matrix (num_docs x vocab_size) - Fisher features Phi
        """
        if self.theta is None or self.I_diag is None:
            raise ValueError("theta or I_diag not fitted. Call fit_theta() and fit_I_diag() first.")

        U = self.fisher_scores(X)
        inv_sqrt_I = 1.0 / np.sqrt(self.I_diag)
        Phi = U.multiply(inv_sqrt_I)

        return Phi
    

    def fit(self, X_train: csr_matrix):
        """
        fits the Fisher_Vectorizer to the training BoW matrix X_train.
        X_train: csr_matrix (num_docs x vocab_size)
        ändert self.theta und self.I_diag
        """
        self.fit_theta(X_train)
        U_train = self.fisher_scores(X_train)
        self.fit_I_diag(U_train)

    def fit_transform(self, X_train: csr_matrix) -> csr_matrix:
        """
        fits the Fisher_Vectorizer to X_train and returns the Fisher features Phi_train.
        X_train: csr_matrix (num_docs x vocab_size)
        returns: csr_matrix (num_docs x vocab_size) - Fisher features Phi_train
        """
        self.fit(X_train)
        return self.transform(X_train)