import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from src.preprocessing import clean_text

class SentimentModelTrainer:
    def __init__(self, data_path="data/dataset_ulasan.csv", test_size=0.2, random_state=42):
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.test_indices = None
        
        # Models and vectorizers
        self.count_vectorizer = None
        self.count_nb_model = None
        
        self.tfidf_vectorizer = None
        self.tfidf_nb_model = None
        
        # Results cache
        self.results = {}
        
    def load_and_prepare_data(self, df_input=None):
        if df_input is not None:
            self.df = df_input.copy()
        else:
            self.df = pd.read_csv(self.data_path)
            
        # Add cleaned text column
        self.df["cleaned_ulasan"] = self.df["ulasan"].apply(lambda x: clean_text(x, remove_stopwords=False))
        
        # Train-test split with fixed random state for 100% identical split
        X = self.df["cleaned_ulasan"]
        y = self.df["sentimen"]
        indices = self.df.index.values
        
        X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
            X, y, indices,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        self.X_train = X_tr
        self.X_test = X_te
        self.y_train = y_tr
        self.y_test = y_te
        self.test_indices = idx_te
        
        return len(self.X_train), len(self.X_test)

    def train_models(self):
        if self.X_train is None:
            self.load_and_prepare_data()
            
        # -------------------------------------------------------------
        # Experiment 1: Multinomial Naive Bayes + Word Frequency (CountVectorizer)
        # -------------------------------------------------------------
        self.count_vectorizer = CountVectorizer(ngram_range=(1, 2))
        X_tr_count = self.count_vectorizer.fit_transform(self.X_train)
        X_te_count = self.count_vectorizer.transform(self.X_test)
        
        self.count_nb_model = MultinomialNB(alpha=1.0)
        self.count_nb_model.fit(X_tr_count, self.y_train)
        
        y_pred_count = self.count_nb_model.predict(X_te_count)
        y_proba_count = self.count_nb_model.predict_proba(X_te_count)
        
        # -------------------------------------------------------------
        # Experiment 2: Multinomial Naive Bayes + TF-IDF (TfidfVectorizer)
        # -------------------------------------------------------------
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        X_tr_tfidf = self.tfidf_vectorizer.fit_transform(self.X_train)
        X_te_tfidf = self.tfidf_vectorizer.transform(self.X_test)
        
        self.tfidf_nb_model = MultinomialNB(alpha=1.0)
        self.tfidf_nb_model.fit(X_tr_tfidf, self.y_train)
        
        y_pred_tfidf = self.tfidf_nb_model.predict(X_te_tfidf)
        y_proba_tfidf = self.tfidf_nb_model.predict_proba(X_te_tfidf)
        
        # Classes: ['negatif', 'positif']
        classes = list(self.count_nb_model.classes_)
        
        # Compute metrics for Model 1 (Count Vectorizer)
        self.results["count"] = self._compute_metrics(
            "Naive Bayes (Frekuensi Kata)",
            self.y_test, y_pred_count, y_proba_count, classes, self.test_indices
        )
        
        # Compute metrics for Model 2 (TF-IDF)
        self.results["tfidf"] = self._compute_metrics(
            "Naive Bayes (TF-IDF)",
            self.y_test, y_pred_tfidf, y_proba_tfidf, classes, self.test_indices
        )
        
        return self.results

    def _compute_metrics(self, model_name, y_true, y_pred, y_proba, classes, test_indices):
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
        
        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        
        # Identify misclassified samples (Pemeriksaan Contoh Salah Prediksi)
        misclassified = []
        for idx, true_lbl, pred_lbl, proba in zip(test_indices, y_true, y_pred, y_proba):
            if true_lbl != pred_lbl:
                orig_text = self.df.loc[idx, "ulasan"]
                clean_text_val = self.df.loc[idx, "cleaned_ulasan"]
                kat = self.df.loc[idx, "kategori"] if "kategori" in self.df.columns else "N/A"
                
                # Confidence score of predicted label
                pred_idx = classes.index(pred_lbl)
                confidence = proba[pred_idx]
                
                misclassified.append({
                    "Index": idx,
                    "Kategori": kat,
                    "Ulasan Asli": orig_text,
                    "Ulasan Bersih": clean_text_val,
                    "Label Sebenarnya": true_lbl,
                    "Prediksi Model": pred_lbl,
                    "Kepercayaan": f"{confidence * 100:.1f}%"
                })
                
        df_misclassified = pd.DataFrame(misclassified)
        
        return {
            "name": model_name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "classification_report": report_dict,
            "confusion_matrix": cm,
            "classes": classes,
            "misclassified_df": df_misclassified,
            "total_test": len(y_true),
            "total_misclassified": len(misclassified)
        }

    def get_top_features(self, feature_type="tfidf", top_n=15):
        """Extract top words indicative of positive vs negative sentiment."""
        vectorizer = self.tfidf_vectorizer if feature_type == "tfidf" else self.count_vectorizer
        model = self.tfidf_nb_model if feature_type == "tfidf" else self.count_nb_model
        
        if vectorizer is None or model is None:
            return None
            
        feature_names = np.array(vectorizer.get_feature_names_out())
        classes = list(model.classes_)
        
        # Log probabilities for each class
        # classes[0] = 'negatif', classes[1] = 'positif'
        log_prob_neg = model.feature_log_prob_[0]
        log_prob_pos = model.feature_log_prob_[1]
        
        top_pos_idx = np.argsort(log_prob_pos)[::-1][:top_n]
        top_neg_idx = np.argsort(log_prob_neg)[::-1][:top_n]
        
        top_positive_words = [(feature_names[i], np.exp(log_prob_pos[i])) for i in top_pos_idx]
        top_negative_words = [(feature_names[i], np.exp(log_prob_neg[i])) for i in top_neg_idx]
        
        return {
            "positive": pd.DataFrame(top_positive_words, columns=["Kata/Frasa", "Skor Bobot"]),
            "negative": pd.DataFrame(top_negative_words, columns=["Kata/Frasa", "Skor Bobot"])
        }

if __name__ == "__main__":
    trainer = SentimentModelTrainer()
    results = trainer.train_models()
    print("--- COUNT VECTORIZER RESULTS ---")
    print(f"Accuracy: {results['count']['accuracy']:.4f}, F1: {results['count']['f1_score']:.4f}")
    print(f"Salah prediksi: {results['count']['total_misclassified']} dari {results['count']['total_test']}")
    
    print("\n--- TF-IDF VECTORIZER RESULTS ---")
    print(f"Accuracy: {results['tfidf']['accuracy']:.4f}, F1: {results['tfidf']['f1_score']:.4f}")
    print(f"Salah prediksi: {results['tfidf']['total_misclassified']} dari {results['tfidf']['total_test']}")
