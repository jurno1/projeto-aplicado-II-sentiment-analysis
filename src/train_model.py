
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

DATA_PATH = "data/amazon_reviews.csv"  # expected CSV with columns 'text' and 'label' (1 positive, 0 negative)
MODELS_DIR = "models"

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at {DATA_PATH}. Please place a CSV with columns 'text' and 'label'.")
        return

    df = pd.read_csv(DATA_PATH)
    if 'text' not in df.columns or 'label' not in df.columns:
        print("CSV precisa conter colunas 'text' e 'label'.")
        return

    X = df['text'].astype(str)
    y = df['label']

    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
    X_vec = tfidf.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    nb = MultinomialNB()
    nb.fit(X_train, y_train)

    logreg = LogisticRegression(max_iter=2000)
    logreg.fit(X_train, y_train)

    os.makedirs(MODELS_DIR, exist_ok=True)
    pickle.dump(tfidf, open(os.path.join(MODELS_DIR, "tfidf.pkl"), "wb"))
    pickle.dump(nb, open(os.path.join(MODELS_DIR, "nb_model.pkl"), "wb"))
    pickle.dump(logreg, open(os.path.join(MODELS_DIR, "logreg_model.pkl"), "wb"))

    print("Modelos treinados e salvos em /models")

if __name__ == '__main__':
    main()
