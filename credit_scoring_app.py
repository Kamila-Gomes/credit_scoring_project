import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score, roc_curve


class OutlierRemover(BaseEstimator, TransformerMixin):
    """Clipping de outliers usando IQR"""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_out = X.copy()
        num_cols = X_out.select_dtypes(include='number').columns
        for col in num_cols:
            Q1 = X_out[col].quantile(0.25)
            Q3 = X_out[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            X_out[col] = X_out[col].clip(lower, upper)
        return X_out


def carregar_dados(uploaded_file):
    """Carrega base CSV ou FTR"""
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.ftr'):
        return pd.read_feather(uploaded_file)
    else:
        st.error("Formato não suportado. Envie um arquivo .csv ou .ftr.")
        return None

def carregar_modelo(model_file):
    """Carrega modelo pickle"""
    return pickle.load(model_file)

def escorar(modelo, df):
    """Gera probabilidade de inadimplência"""
    X = df.drop(columns=['index', 'data_ref', 'mau'], errors='ignore')
    model_cols = modelo.feature_names_in_ if hasattr(modelo, "feature_names_in_") else X.columns
    for col in model_cols:
        if col not in X.columns:
            X[col] = 0  
    df['prob_inadimplencia'] = modelo.predict_proba(X[model_cols])[:, 1]
    return df

def calcular_metricas(y_true, prob):
    """Calcula AUC, KS e Gini"""
    auc = roc_auc_score(y_true, prob)
    fpr, tpr, _ = roc_curve(y_true, prob)
    ks = max(tpr - fpr)
    gini = 2 * auc - 1
    return auc, ks, gini, fpr, tpr

st.set_page_config(page_title="Credit Scoring Dashboard", layout="wide")
st.title("💳 Credit Scoring Dashboard")
st.markdown("""
Aplicativo para **escoragem e avaliação de risco de crédito**.  
Envie os arquivos de base e modelo na barra lateral para começar.
""")

st.sidebar.header("📂 Entrada de Dados")
uploaded_file = st.sidebar.file_uploader("Carregue a base (.ftr ou .csv)", type=["csv", "ftr"])

st.sidebar.header("⚙️ Modelo Treinado")
model_file = st.sidebar.file_uploader("Carregue o modelo (.pkl)", type=["pkl"])

if uploaded_file is None:
    st.info("👈 Carregue um arquivo de base (.csv ou .ftr) para começar.")
elif model_file is None:
    st.info("👈 Carregue o modelo (.pkl) para escorar os dados.")
else:

    df = carregar_dados(uploaded_file)
    modelo = carregar_modelo(model_file)

    if df is not None and modelo is not None:
        st.success(f"✅ Base e modelo carregados com sucesso! ({df.shape[0]} linhas e {df.shape[1]} colunas)")


        st.markdown("---")
        st.subheader("🛠 Pré-processamento")
        st.markdown("Aplicando clipping de outliers nas colunas numéricas...")
        preprocess = OutlierRemover()
        df = preprocess.fit_transform(df)
        st.success("✅ Pré-processamento concluído")

        try:
            df = escorar(modelo, df)
            st.success("✅ Escoragem realizada com sucesso")
        except Exception as e:
            st.error(f"❌ Erro na escoragem: {e}")

        st.markdown("---")
        st.subheader("🔮 Top 10 Clientes Mais Arriscados")

        id_cols = ['Cliente_ID', 'CNPJ emit.', 'Emitente', 'No.']
        if not any(col in df.columns for col in id_cols):
            df.insert(0, "Cliente_ID", range(1, len(df) + 1))

        colunas_candidatas = ['Cliente_ID', 'CNPJ emit.', 'Emitente', 'No.', 'prob_inadimplencia']
        cols_para_mostrar = [c for c in colunas_candidatas if c in df.columns]

        df_show = df[cols_para_mostrar].sort_values(by='prob_inadimplencia', ascending=False).head(10).copy()

        styled_df = (
            df_show.style
                .format({"prob_inadimplencia": "{:.2%}"})
                .background_gradient(subset=['prob_inadimplencia'], cmap="RdYlGn_r")
                .set_table_styles([
                    {"selector": "th", "props": [("text-align", "center"), ("font-weight", "bold"), ("color", "#4DA6FF"), ("font-size", "13px")]},
                    {"selector": "td", "props": [("text-align", "center"), ("padding", "6px 12px"), ("font-size", "12px")]}
                ])
        )
        st.table(styled_df)


        st.subheader("📊 Distribuição Geral das Probabilidades")
        col1, col2 = st.columns([1.5, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.hist(df['prob_inadimplencia'], bins=30, color="#4DA6FF", edgecolor="white")
            ax.set_title("Distribuição Geral", fontsize=13)
            ax.set_xlabel("Probabilidade", fontsize=11)
            ax.set_ylabel("Frequência", fontsize=11)
            plt.tight_layout()
            st.pyplot(fig)
        with col2:
            prob_media = df['prob_inadimplencia'].mean()
            st.markdown("<div style='margin-top:40px'></div>", unsafe_allow_html=True)
            st.markdown(f"<h5 style='color:lightblue;'>Probabilidade Média</h5>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color:white;'>{prob_media:.2%}</h2>", unsafe_allow_html=True)


        if 'mau' in df.columns:
            st.markdown("---")
            st.subheader("🧠 Métricas de Performance do Modelo")
            auc, ks, gini, fpr, tpr = calcular_metricas(df['mau'], df['prob_inadimplencia'])
            col1, col2, col3 = st.columns(3)
            col1.metric("AUC", f"{auc:.3f}")
            col2.metric("KS", f"{ks:.3f}")
            col3.metric("Gini", f"{gini:.3f}")

            col_g1, col_g2 = st.columns(2)
            with col_g1:
                fig1, ax1 = plt.subplots(figsize=(3.8,2.5))
                ax1.plot(fpr, tpr, color='blue', label=f'AUC = {auc:.3f}')
                ax1.plot([0,1],[0,1],'--',color='gray')
                ax1.set_xlabel('Falso Positivo')
                ax1.set_ylabel('Verdadeiro Positivo')
                ax1.set_title('Curva ROC', fontsize=10)
                ax1.legend(fontsize=8)
                st.pyplot(fig1)
            with col_g2:
                fig2, ax2 = plt.subplots(figsize=(3.8,2.5))
                sns.kdeplot(data=df, x='prob_inadimplencia', hue='mau', fill=True, alpha=0.5, ax=ax2)
                ax2.set_title('Distribuição por Classe (mau=0/1)', fontsize=10)
                ax2.set_xlabel('Probabilidade')
                st.pyplot(fig2)

        st.markdown("---")
        st.subheader("⬇️ Download dos Resultados")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Baixar CSV com probabilidades", csv, "base_escorada.csv", "text/csv")
