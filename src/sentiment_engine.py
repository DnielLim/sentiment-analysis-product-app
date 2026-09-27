import pandas as pd
import numpy as np
from src.preprocessing import clean_text

class SentimentEngine:
    def __init__(self, trainer):
        self.trainer = trainer

    def predict_single(self, text: str, model_type: str = "tfidf"):
        """
        Predict sentiment for a single review string.
        model_type: 'count' or 'tfidf'
        """
        if not text or not text.strip():
            return {
                "label": "N/A",
                "probabilities": {"positif": 0.0, "negatif": 0.0},
                "confidence": 0.0,
                "cleaned_text": "",
                "highlighted_words": []
            }
            
        cleaned = clean_text(text)
        
        vectorizer = self.trainer.tfidf_vectorizer if model_type == "tfidf" else self.trainer.count_vectorizer
        model = self.trainer.tfidf_nb_model if model_type == "tfidf" else self.trainer.count_nb_model
        
        X_vec = vectorizer.transform([cleaned])
        pred_label = str(model.predict(X_vec)[0])
        probas = model.predict_proba(X_vec)[0]
        
        classes = [str(c) for c in model.classes_]
        prob_dict = {cls: float(prob) for cls, prob in zip(classes, probas)}
        
        # Extract features present in this review
        feature_names = vectorizer.get_feature_names_out()
        nonzero_indices = X_vec.nonzero()[1]
        
        words_contrib = []
        pos_idx = classes.index("positif")
        neg_idx = classes.index("negatif")
        
        for idx in nonzero_indices:
            word = feature_names[idx]
            pos_weight = float(np.exp(model.feature_log_prob_[pos_idx][idx]))
            neg_weight = float(np.exp(model.feature_log_prob_[neg_idx][idx]))
            words_contrib.append({
                "word": str(word),
                "positif_score": pos_weight,
                "negatif_score": neg_weight,
                "polarity": "positif" if pos_weight >= neg_weight else "negatif"
            })
            
        words_contrib.sort(key=lambda x: abs(x["positif_score"] - x["negatif_score"]), reverse=True)
        
        return {
            "label": pred_label,
            "probabilities": prob_dict,
            "confidence": prob_dict.get(pred_label, 0.0),
            "cleaned_text": cleaned,
            "highlighted_words": words_contrib
        }

    def predict_batch(self, df: pd.DataFrame, text_column: str = "ulasan", model_type: str = "tfidf"):
        """
        Predict sentiment for a pandas DataFrame containing review texts.
        """
        if text_column not in df.columns:
            raise ValueError(f"Kolom '{text_column}' tidak ditemukan dalam dataset.")
            
        df_res = df.copy()
        df_res["cleaned_text"] = df_res[text_column].apply(lambda x: clean_text(str(x)))
        
        vectorizer = self.trainer.tfidf_vectorizer if model_type == "tfidf" else self.trainer.count_vectorizer
        model = self.trainer.tfidf_nb_model if model_type == "tfidf" else self.trainer.count_nb_model
        
        X_vec = vectorizer.transform(df_res["cleaned_text"])
        preds = [str(p) for p in model.predict(X_vec)]
        probas = model.predict_proba(X_vec)
        
        classes = [str(c) for c in model.classes_]
        pos_idx = classes.index("positif")
        neg_idx = classes.index("negatif")
        
        df_res["prediksi_sentimen"] = preds
        df_res["probabilitas_positif"] = probas[:, pos_idx]
        df_res["probabilitas_negatif"] = probas[:, neg_idx]
        df_res["skor_kepercayaan"] = [float(probas[i, classes.index(p)]) for i, p in enumerate(preds)]
        
        return df_res
