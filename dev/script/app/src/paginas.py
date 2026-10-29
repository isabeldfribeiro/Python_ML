import numpy as np
import pandas as pd
import streamlit as st
from joblib import load
from pydantic import validate_call
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from src.acesso_data import acesso_data_banco
from src.modelos.clustering import Clustering
from src.modelos.classifier import Classifier
from src.modelos.regression import Regression

VARIAVEIS_CLUSTER = ["n_tetraciclina", "n_polimixinas", "n_degradacao_aromaticos"]
COR_DESTAQUE = "#c6efce"


class Paginas:
    @validate_call
    def busca_dados_cluster(_self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Função de busca dos dados de treino e teste.
        Retorna:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            DataFrames dos datasets de teste, treino e validação.
        """
        banco = acesso_data_banco()
        link = banco + "df_estruturada/"
        df_test = pd.read_csv(link + "df_test.csv")
        df_train = pd.read_csv(link + "df_train.csv")
        df_valid = pd.read_csv(link + "df_valid.csv")

        return df_train, df_valid, df_test

    @validate_call
    def busca_dados_class(_self) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Função de busca dos dados de classificação.
        Retorna:
            tuple[dict[str, pd.DataFrame], pd.DataFrame]:
            DataFrames (Dataset, Classe_Predita) de cada modelo e o
            DataFrame de teste com a classe observada ("grupo").
        """
        banco = acesso_data_banco()
        link = banco + "df_resultados/"
        modelos = ["GBM", "k-NN", "NB", "NN", "RF", "SVM"]
        datasets = {m: pd.read_csv(link + f"classified_{m}.csv") for m in modelos}

        df_test = pd.read_csv(link + "classified_Teste.csv")

        return datasets, df_test

    @validate_call
    def busca_dados_regr(_self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Função de busca dos dados de treino e teste.
        Retorna:
            tuple[pd.DataFrame, pd.DataFrame]:
            DataFrames dos datasets de treino e teste.
        """
        banco = acesso_data_banco()
        colunas = [
            "MIC_imipenem",
            "Multiple Linear Regression",
            "Suport Vector Machine",
            "Nayve Bayes",
            "Neural Network",
        ]
        df_train = load(banco + "train_resultados_regr")[colunas].copy()
        df_test = load(banco + "test_resultados_regr")[colunas].copy()

        return df_train, df_test

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------
    def __clustering(
        _self,
        variavel_1: str,
        variavel_2: str,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
    ) -> tuple[dict, str, str, str]:
        """
        Função de teste dos modelos de clustering.
        Parametros:
            self: Referência para a própria classe.
            variavel_1: Primeira variável para o clustering.
            variavel_2: Segunda variável para o clustering.
            df_train: DataFrame de treino.
            df_test: DataFrame de teste.
        Retorna:
            tuple[dict, str, str, str]: Resultados dos modelos e
                mensagens de sucesso.
        """
        i = variavel_1
        j = variavel_2

        cluster = Clustering(c1=i, c2=j, df=df_train, modelo="kmeans")
        resultados = {}
        resultados["K-means"], msn_kmenans = cluster.testar_modelo(
            best_param={"n_clusters": 3, "init": "k-means++", "n_init": 10},
            df_test=df_test,
        )

        cluster = Clustering(c1=i, c2=j, df=df_train, modelo="dbscan")
        resultados["DBSCAN"], msn_dbscan = cluster.testar_modelo(
            best_param={"eps": 0.5, "min_samples": 10, "metric": "euclidean"},
            df_test=df_test,
        )

        cluster = Clustering(c1=i, c2=j, df=df_train, modelo="agglomerative")
        resultados["Agg"], msn_agg = cluster.testar_modelo(
            best_param={"n_clusters": 2, "metric": "euclidean", "linkage": "average"},
            df_test=df_test,
        )

        return resultados, msn_kmenans, msn_dbscan, msn_agg

    # ------------------------------------------------------------------
    # Classificação
    # ------------------------------------------------------------------
    def _resumo_classificacao(_self, df_test: pd.DataFrame, modelos: list) -> pd.DataFrame:
        """Calcula as métricas de cada classificador no conjunto de teste."""
        y_true = df_test["grupo"]
        linhas = []
        for m in modelos:
            y_pred = df_test[m]
            linhas.append(
                {
                    "Modelo": m,
                    "Acurácia": accuracy_score(y_true, y_pred),
                    "Precisão": precision_score(
                        y_true, y_pred, average="macro", zero_division=0
                    ),
                    "Recall": recall_score(
                        y_true, y_pred, average="macro", zero_division=0
                    ),
                    "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
                }
            )
        return pd.DataFrame(linhas)

    def __classifier(_self) -> None:
        dados, df_test_class = _self.busca_dados_class()
        modelos = list(dados.keys())

        resumo = _self._resumo_classificacao(df_test_class, modelos)
        melhor = resumo.loc[resumo["Acurácia"].idxmax()]

        st.subheader("Resumo dos classificadores (conjunto de teste)")
        cards = st.columns(5)
        cards[0].metric("Melhor modelo", melhor["Modelo"])
        cards[1].metric("Acurácia", f"{melhor['Acurácia']:.1%}")
        cards[2].metric("Precisão", f"{melhor['Precisão']:.1%}")
        cards[3].metric("Recall", f"{melhor['Recall']:.1%}")
        cards[4].metric("F1", f"{melhor['F1']:.1%}")

        tabela = resumo.set_index("Modelo").style.format("{:.3f}")
        tabela = tabela.highlight_max(axis=0, color=COR_DESTAQUE)
        st.dataframe(tabela)
        st.caption("Métricas com média macro. Em verde: melhor valor de cada coluna.")

        st.divider()
        st.subheader("Detalhes por modelo")
        abas = st.tabs(modelos)
        for aba, nome in zip(abas, modelos):
            with aba:
                classificador = Classifier(df=dados[nome])
                col = st.columns(2, gap="large")
                with col[0]:
                    with st.container(border=True):
                        classificador.testar_hipotese()
                with col[1]:
                    with st.container(border=True):
                        classificador.matrix_confusao(
                            df=df_test_class, obs="grupo", pred=nome
                        )
        return None

    # ------------------------------------------------------------------
    # Regressão
    # ------------------------------------------------------------------
    def _resumo_regressao(
        _self, df_test: pd.DataFrame, modelos: list, y_true: str
    ) -> pd.DataFrame:
        """Calcula RMSE, MAE e R² de cada regressor no conjunto de teste."""
        linhas = []
        for m in modelos:
            linhas.append(
                {
                    "Modelo": m,
                    "RMSE": float(np.sqrt(mean_squared_error(df_test[y_true], df_test[m]))),
                    "MAE": mean_absolute_error(df_test[y_true], df_test[m]),
                    "R²": r2_score(df_test[y_true], df_test[m]),
                }
            )
        return pd.DataFrame(linhas)

    def __regressor(_self) -> None:
        df_train_regr, df_test_regr = _self.busca_dados_regr()
        y_true = "MIC_imipenem"
        modelos = list(df_train_regr.columns[1:5])

        resumo = _self._resumo_regressao(df_test_regr, modelos, y_true)
        melhor = resumo.loc[resumo["RMSE"].idxmin()]

        st.subheader("Resumo dos regressores (conjunto de teste)")
        cards = st.columns(4)
        cards[0].metric("Melhor modelo", melhor["Modelo"])
        cards[1].metric("RMSE", f"{melhor['RMSE']:.3f}")
        cards[2].metric("MAE", f"{melhor['MAE']:.3f}")
        cards[3].metric("R²", f"{melhor['R²']:.3f}")

        tabela = resumo.set_index("Modelo").style.format("{:.3f}")
        tabela = tabela.highlight_min(subset=["RMSE", "MAE"], color=COR_DESTAQUE)
        tabela = tabela.highlight_max(subset=["R²"], color=COR_DESTAQUE)
        st.dataframe(tabela)
        st.caption("Em verde: melhor valor de cada coluna (menor erro, maior R²).")

        st.divider()
        st.subheader("Detalhes por modelo")
        abas = st.tabs(modelos)
        for aba, nome in zip(abas, modelos):
            with aba:
                regressor = Regression(
                    y_pred_treino=df_train_regr[nome], y_pred_teste=df_test_regr[nome]
                )
                col = st.columns(2, gap="large")
                with col[0]:
                    with st.container(border=True):
                        regressor.testar_hipotese(y_pred=nome)
                with col[1]:
                    with st.container(border=True):
                        regressor.obter_grafico(
                            df_test=df_test_regr, y_true=y_true, y_pred=nome
                        )
        return None

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    def __datasets(_self) -> None:
        pasta = acesso_data_banco() + "df_estruturada/"
        conjuntos = {"Treino": "train", "Validação": "valid", "Teste": "test"}

        dfs = {}
        for nome, sufixo in conjuntos.items():
            x = pd.read_csv(pasta + f"X_{sufixo}.csv")
            y = pd.read_csv(pasta + f"y_{sufixo}.csv")
            dfs[nome] = pd.concat([y, x], axis=1)

        total = sum(len(d) for d in dfs.values())
        cards = st.columns(len(dfs) + 1)
        for card, (nome, df) in zip(cards, dfs.items()):
            card.metric(f"{nome}", f"{len(df)} amostras", f"{len(df) / total:.0%}")
        cards[-1].metric("Total", f"{total} amostras")

        st.divider()
        abas = st.tabs([f"Dados de {nome}" for nome in dfs])
        for aba, (nome, df) in zip(abas, dfs.items()):
            with aba:
                st.caption(f"{df.shape[0]} linhas × {df.shape[1]} colunas")
                st.dataframe(df, height=450)
        return None

    # ------------------------------------------------------------------
    # Página principal
    # ------------------------------------------------------------------
    def metricas(self) -> str:
        """
        Função de cálculo das métricas de clustering.
        Parametros:
            self: Referência para a própria classe.
        Retorna:
            str: Mensagem de sucesso.
        """
        abas = st.tabs(
            [
                "Clustering",
                "Classificação",
                "Regressão",
                "Datasets",
            ]
        )
        with abas[0]:
            df_train, _, df_test = self.busca_dados_cluster()

            with st.container(border=True):
                st.markdown("**Escolha as variáveis do clustering**")
                sel = st.columns(2)
                variavel_1 = sel[0].selectbox(
                    "Primeira variável:", VARIAVEIS_CLUSTER, index=0
                )
                variavel_2 = sel[1].selectbox(
                    "Segunda variável:", VARIAVEIS_CLUSTER, index=1
                )

            if variavel_1 == variavel_2:
                st.warning("Escolha duas variáveis diferentes para comparar.")
            else:
                resultados, msn_kmenans, msn_dbscan, msn_agg = self.__clustering(
                    variavel_1=variavel_1,
                    variavel_2=variavel_2,
                    df_train=df_train,
                    df_test=df_test,
                )
                graficos = [
                    ("K-means", msn_kmenans, resultados["K-means"]),
                    ("DBSCAN", msn_dbscan, resultados["DBSCAN"]),
                    ("Agglomerative", msn_agg, resultados["Agg"]),
                ]
                col = st.columns(3, gap="medium")
                for c, (titulo, msn, fig) in zip(col, graficos):
                    with c:
                        with st.container(border=True):
                            st.markdown(f"#### {titulo}")
                            st.markdown(f"<p>{msn}</p>", unsafe_allow_html=True)
                            st.pyplot(fig=fig)

        with abas[1]:
            self.__classifier()

        with abas[2]:
            self.__regressor()

        with abas[3]:
            self.__datasets()

        return "Métricas Calculadas"
