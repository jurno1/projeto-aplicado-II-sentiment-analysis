
import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dashboard de Análise de Sentimentos", layout="wide")

# -------------------------
# Helpers: load / train
# -------------------------
TFIDF_PATH = "models/tfidf.pkl"
NB_PATH = "models/nb_model.pkl"
LOGREG_PATH = "models/logreg_model.pkl"

@st.cache_data
def load_pickle(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def train_fallback_models():
    # Pequeno dataset de exemplo para fallback (texto em inglês)
    X = [
        "Great product, very happy with the purchase.",
        "Terrible quality, broke after one day.",
        "Works as expected. Satisfied.",
        "Very disappointed. Will not buy again.",
        "Excellent! Highly recommend to everyone.",
        "Bad item. Not as described. Waste of money.",
        "I love it, perfect item.",
        "Awful. Received damaged and no support.",
        "Very good value for money.",
        "Product stopped working after a week."
    ]
    y = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    Xtr = tfidf.fit_transform(X_train)

    nb = MultinomialNB()
    nb.fit(Xtr, y_train)

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(Xtr, y_train)

    # Create simple pipeline objects for convenience
    nb_pipe = make_pipeline(tfidf, nb)
    logreg_pipe = make_pipeline(tfidf, logreg)

    # Evaluate quickly
    Xte = X_test
    if len(Xte) > 0:
        ypred_nb = nb_pipe.predict(Xte)
        ypred_lr = logreg_pipe.predict(Xte)
    else:
        ypred_nb = ypred_lr = y_test

    return tfidf, nb, logreg, {
        "nb": compute_metrics(y_test, ypred_nb) if len(Xte)>0 else {"accuracy": None},
        "logreg": compute_metrics(y_test, ypred_lr) if len(Xte)>0 else {"accuracy": None}
    }

def compute_metrics(y_true, y_pred):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred)*100, 2),
        "precision": round(precision_score(y_true, y_pred, zero_division=0)*100, 2),
        "recall": round(recall_score(y_true, y_pred, zero_division=0)*100, 2),
        "f1": round(f1_score(y_true, y_pred, zero_division=0)*100, 2)
    }

# -------------------------
# Load or fallback
# -------------------------
tfidf = load_pickle(TFIDF_PATH)
nb_model = load_pickle(NB_PATH)
logreg_model = load_pickle(LOGREG_PATH)
trained_in_app = False
metrics_info = None

if tfidf is None or nb_model is None or logreg_model is None:
    # Treina fallback e salva localmente (opcional)
    tfidf_f, nb_f, logreg_f, metrics_info = train_fallback_models()
    try:
        save_pickle(tfidf_f, TFIDF_PATH)
        save_pickle(nb_f, NB_PATH)
        save_pickle(logreg_f, LOGREG_PATH)
    except Exception:
        pass
    tfidf, nb_model, logreg_model = tfidf_f, nb_f, logreg_f
    trained_in_app = True

# Expected metrics for reporting
expected_metrics = {
    "Naive Bayes": {"accuracy": 82.0, "precision": 81.0, "recall": 83.0, "f1": 82.0},
    "Regressão Logística": {"accuracy": 78.0, "precision": 79.0, "recall": 77.0, "f1": 78.0}
}

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("Navegação")
page = st.sidebar.radio("Ir para", ["Visão Geral", "Predict", "Metrics", "Data"])
st.sidebar.markdown("---")
st.sidebar.write("Modelos carregados:")
st.sidebar.write(f"- Naive Bayes: {'Sim' if nb_model is not None else 'Não'}")
st.sidebar.write(f"- Regressão Logística: {'Sim' if logreg_model is not None else 'Não'}")
if trained_in_app:
    st.sidebar.info("Modelos de exemplo foram treinados localmente (fallback).")

# -------------------------
# Pages
# -------------------------
if page == "Visão Geral":
    st.title("Dashboard de Análise de Sentimentos")
    st.markdown(
        """
        **Projeto:** Sistema de Análise de Sentimentos para Atendimento ao Cliente

        **Objetivo:** Classificar automaticamente comentários como *Positivos* ou *Negativos*

        **Modelos testados:** Naive Bayes (baseline) e Regressão Logística
        """
    )

    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Resumo executivo")
        st.write(
            "O sistema processa comentários (reviews) e retorna a classificação de sentimento. "
            "O modelo Naive Bayes apresentou melhor desempenho no conjunto de referência (82% de acurácia), "
            "enquanto a Regressão Logística apresentou 78%."
        )
        st.markdown("**Produto**: Protótipo com dashboard (Streamlit) que aceita entrada de texto e CSVs, e exibe estatísticas em tempo real.")

    with col2:
        st.subheader("Comparativo rápido")
        df_comp = pd.DataFrame({
            "Modelo": ["Naive Bayes", "Regressão Logística"],
            "Acurácia (%)": [expected_metrics["Naive Bayes"]["accuracy"], expected_metrics["Regressão Logística"]["accuracy"]],
            "F1-Score (%)": [expected_metrics["Naive Bayes"]["f1"], expected_metrics["Regressão Logística"]["f1"]]
        }).set_index("Modelo")
        st.table(df_comp)

    st.markdown("---")
    st.subheader("Como usar")
    st.markdown(
        """
        1. Vá para a aba **Predict** para testar textos avulsos.
        2. Suba um CSV com coluna `text` na aba **Data** para predições em lote.
        3. Visualize métricas e gráficos na aba **Metrics`.
        """
    )

elif page == "Predict":
    st.title("Previsão de Sentimento — Texto Único ou Lote (CSV)")
    st.markdown("Escolha o modelo para predizer e insira o texto abaixo, ou envie um CSV com uma coluna chamada `text`.")
    model_choice = st.selectbox("Modelo", ["Naive Bayes", "Regressão Logística"])
    input_mode = st.radio("Entrada", ["Texto único", "Upload CSV"])

    if input_mode == "Texto único":
        txt = st.text_area("Digite o comentário aqui", height=150)
        if st.button("Predizer"):
            if not txt.strip():
                st.warning("Insira algum texto para predizer.")
            else:
                X_vec = tfidf.transform([txt])
                if model_choice == "Naive Bayes":
                    pred = nb_model.predict(X_vec)[0]
                    prob = nb_model.predict_proba(X_vec).max()
                    proba = nb_model.predict_proba(X_vec)[0]
                else:
                    pred = logreg_model.predict(X_vec)[0]
                    prob = logreg_model.predict_proba(X_vec).max()
                    proba = logreg_model.predict_proba(X_vec)[0]

                label = "Positivo" if pred == 1 else "Negativo"
                st.metric("Predição", f"{label}", delta=f"{prob*100:.1f}% prob.")

                dfp = pd.DataFrame({"Classe": ["Negativo", "Positivo"], "Probabilidade": proba})
                st.table(dfp)

    else:
        uploaded_file = st.file_uploader("Upload CSV (coluna 'text')", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if "text" not in df.columns:
                st.error("CSV precisa conter uma coluna chamada 'text'.")
            else:
                st.write("Exemplo de linhas carregadas:")
                st.dataframe(df.head())
                if st.button("Predizer em lote"):
                    texts = df['text'].astype(str).tolist()
                    X_vec = tfidf.transform(texts)

                    if model_choice == "Naive Bayes":
                        preds = nb_model.predict(X_vec)
                        probs = nb_model.predict_proba(X_vec)
                    else:
                        preds = logreg_model.predict(X_vec)
                        probs = logreg_model.predict_proba(X_vec)

                    df['pred'] = np.where(preds==1, "Positivo", "Negativo")
                    df['prob_pos'] = probs[:,1]
                    st.success("Previsões geradas.")
                    st.dataframe(df.head())

                    # download
                    towrite = BytesIO()
                    df.to_csv(towrite, index=False)
                    towrite.seek(0)
                    st.download_button("Download CSV com Predição", towrite, file_name="predicoes_com_sentimento.csv", mime="text/csv")

elif page == "Metrics":
    st.title("Métricas e Gráficos de Comparação")
    st.markdown("Aqui estão as métricas consolidadas e gráficos comparativos entre os modelos testados.")

    # Table with expected metrics (from user)
    metrics_df = pd.DataFrame([
        ["Naive Bayes", expected_metrics["Naive Bayes"]["accuracy"], expected_metrics["Naive Bayes"]["precision"], expected_metrics["Naive Bayes"]["recall"], expected_metrics["Naive Bayes"]["f1"]],
        ["Regressão Logística", expected_metrics["Regressão Logística"]["accuracy"], expected_metrics["Regressão Logística"]["precision"], expected_metrics["Regressão Logística"]["recall"], expected_metrics["Regressão Logística"]["f1"]],
    ], columns=["Modelo", "Acurácia (%)", "Precisão (%)", "Recall (%)", "F1-Score (%)"]).set_index("Modelo")

    st.table(metrics_df)

    # Plot comparativo (Acurácia e F1)
    fig, ax = plt.subplots(figsize=(7,4))
    x = np.arange(len(metrics_df.index))
    width = 0.35
    acc = metrics_df["Acurácia (%)"].values
    f1 = metrics_df["F1-Score (%)"].values

    ax.bar(x - width/2, acc, width, label='Acurácia (%)')
    ax.bar(x + width/2, f1, width, label='F1-Score (%)')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df.index)
    ax.set_ylabel("Percentual (%)")
    ax.set_title("Comparativo: Acurácia vs F1-Score")
    ax.legend()
    st.pyplot(fig)

    st.markdown("**Interpretação:** Naive Bayes apresenta vantagem em Acurácia e F1, sendo uma escolha eficiente para baseline e aplicações com necessidade de resposta rápida.")

elif page == "Data":
    st.title("Upload de Dados e Pré-visualização")
    st.markdown("Faça upload de um CSV com a coluna `text` para análise em lote. O app retornará predições e permitirá download com resultados.")
    uploaded_file = st.file_uploader("Enviar CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(f"Linhas: {len(df)}")
        if "text" not in df.columns:
            st.error("O arquivo precisa conter uma coluna chamada 'text'.")
        else:
            st.dataframe(df.head())
            model_choice = st.selectbox("Modelo para predição em lote", ["Naive Bayes", "Regressão Logística"])

            if st.button("Rodar predição em lote"):
                texts = df['text'].astype(str).tolist()
                X_vec = tfidf.transform(texts)

                if model_choice == "Naive Bayes":
                    preds = nb_model.predict(X_vec)
                    probs = nb_model.predict_proba(X_vec)
                else:
                    preds = logreg_model.predict(X_vec)
                    probs = logreg_model.predict_proba(X_vec)

                df['pred'] = np.where(preds==1, "Positivo", "Negativo")
                df['prob_pos'] = probs[:,1]
                st.success("Predições adicionadas ao DataFrame.")
                st.dataframe(df.head())

                # download
                towrite = BytesIO()
                df.to_csv(towrite, index=False)
                towrite.seek(0)
                st.download_button("Download CSV com Predição", towrite, file_name="predicoes_com_sentimento.csv", mime="text/csv")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("Dashboard gerado para o Projeto Aplicado II — Etapa 3. Ajuste os modelos e o TF-IDF conforme o dataset real para melhores resultados.")
