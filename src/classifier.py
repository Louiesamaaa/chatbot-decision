import gzip
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

def load_abcd_data(path = "../data/abcd_v1.1.json.gz"):
    with gzip.open(path) as f:
        data = json.load(f) 
    return data

def customer_messages(convos, k):
    msgs = [text for speaker, text in convos["original"] if speaker == "customer"]
    return " ".join(msgs[:k])

class IntentClassifier:
    def __init__(self):

        self.model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df =2, sublinear_tf = True),
            LogisticRegression(C=5.0, max_iter=2000, random_state= 6)
        )

    def fit(self,convos, max_msgs = 3, label_fn = None):

        if label_fn is None:
            label_fn = lambda c: c["scenario"]["flow"]

        texts, labels = [], []
        for c in convos:
            for k in range(1, max_msgs + 1):
                texts.append(customer_messages(c, k))
                labels.append(label_fn(c))
        self.model.fit(texts, labels)

        return self

    def predict(self, text):

        probs = self.model.predict_proba([text])[0]
        best = int(np.argmax(probs))

        return self.model.classes_[best], float(probs[best])

    def evaluate(self,convos,k, label_fn = None):

        if label_fn is None:    
            label_fn = lambda c: c["scenario"]["flow"]

        texts = [customer_messages(c, k) for c in convos]
        labels = np.array([label_fn(c) for c in convos])
        probs = self.model.predict_proba(texts)
        preds = self.model.classes_[np.argmax(probs, axis=1)]

        return {"accuracy": float(np.mean(preds == labels)),
                "avg_confidence": float(np.mean(np.max(probs, axis=1)))}

    def save(self, path = "results/intent_classifier.joblib"):
        joblib.dump(self.model, path)
        print(f"Saved model to {path}")

    @classmethod
    def load(cls, path= "results/intent_classifier.joblib"):
        clf = cls()
        clf.model = joblib.load(path)
        return clf