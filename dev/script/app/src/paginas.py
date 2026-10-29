import os
import pandas as pd
import streamlit as st
from joblib import load
from pydantic import validate_call

from src.acesso_data import acesso_data_banco
from src.modelos.clustering import Clustering
from src.modelos.classifier import Classifier
from src.modelos.regression import Regression


class Paginas:
    @validate_call
    def busca_dados_cluster(_self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Função de busca dos dados de treino e teste.
        Retorna:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            DataFrames dos datasets de teste, treino e validação.
        """
        # Importando os dados de treino e teste
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
        # Importando os dados de treino e teste
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
        # K-means ------------------------------------------------
        i = variavel_1
        j = variavel_2

        cluster = Clustering(c1=i, c2=j, df=df_train, modelo="kmeans")
        resultados = {}
        resultados["K-means"], msn_kmenans = cluster.testar_modelo(
            best_param={"n_clusters": 3, "init": "k-means++", "n_init": 10},
            df_test=df_test,
        )

        # DBSCAN -------------------------------------------------
        i = variavel_1
        j = variavel_2

        cluster = Clustering(c1=i, c2=j, df=df_train, modelo="dbscan")
        resultados["DBSCAN"], msn_dbscan = cluster.testar_modelo(
            best_param={"eps": 0.5, "min_samples": 10, "metric": "euclidean"},
            df_test=df_test,
        )

        # Agglomerative ------------------------------------------
        i = variavel_1
        j = variavel_2

        cluster = Clustering(c1=i, c2=j, df=df_train, modelo="agglomerative")
        resultados["Agg"], msn_agg = cluster.testar_modelo(
            best_param={"n_clusters": 2, "metric": "euclidean", "linkage": "average"},
            df_test=df_test,
        )

        return resultados, msn_kmenans, msn_dbscan, msn_agg

    def __classifier(_self) -> None:
        dados, df_test_class = _self.busca_dados_class()

        col = st.columns((5, 0.5, 5, 0.5, 5))
        dados_metricas = {
            0: "GBM",
            2: "k-NN",
            4: "NB",
        }
        for k, v in dados_metricas.items():
            with col[k]:
                st.write("=" * 69)
                st.markdown(
                    f"<h4 style='text-align: center; '>{v}</h4>", unsafe_allow_html=True
                )
                st.write("=" * 69)
                classificador = Classifier(df=dados[v])
                classificador.testar_hipotese()
                st.divider()
                classificador.matrix_confusao(df=df_test_class, obs="grupo", pred=v)

        st.divider()

        col = st.columns((5, 0.5, 5, 0.5, 5))
        dados_metricas = {0: "NN", 2: "RF", 4: "SVM"}
        for k, v in dados_metricas.items():
            with col[k]:
                st.write("=" * 69)
                st.markdown(
                    f"<h4 style='text-align: center; '>{v}</h4>", unsafe_allow_html=True
                )
                st.write("=" * 69)
                classificador = Classifier(df=dados[v])
                classificador.testar_hipotese()
                st.divider()
                classificador.matrix_confusao(df=df_test_class, obs="grupo", pred=v)
        return None

    def __regressor(_self) -> None:
        df_train_regr, df_test_regr = _self.busca_dados_regr()
        modelos = {
            0: df_train_regr.columns[1],
            2: df_train_regr.columns[2],
            4: df_train_regr.columns[3],
            6: df_train_regr.columns[4],
        }
        col = st.columns((5, 0.5, 5, 0.5, 5, 0.5, 5))
        for k, v in modelos.items():
            with col[k]:
                regressor = Regression(
                    y_pred_treino=df_train_regr[v], y_pred_teste=df_test_regr[v]
                )
                regressor.testar_hipotese(y_pred=v)
                st.divider()
                regressor.obter_grafico(df_test=df_test_regr, y_true="MIC_imipenem", y_pred=v)
        return None

    def __datasets(_self) -> None:
        pasta = acesso_data_banco() + "df_estruturada/"
        conjuntos = {"Treino": "train", "Validação": "valid", "Teste": "test"}

        col = st.columns((2, 0.05, 2, 0.05, 2))
        tamanhos = []
        for pos, (nome, sufixo) in zip((0, 2, 4), conjuntos.items()):
            x = pd.read_csv(pasta + f"X_{sufixo}.csv")
            y = pd.read_csv(pasta + f"y_{sufixo}.csv")
            df = pd.concat([y, x], axis=1)
            tamanhos.append(len(df))
            with col[pos]:
                st.markdown(
                    f"<h4 style='text-align: center; '>Dados de {nome}</h4>",
                    unsafe_allow_html=True,
                )
                st.dataframe(df)

        col = st.columns((2, 10))
        with col[0]:
            metr = pd.DataFrame(
                {
                    "Métrica": list(conjuntos.keys()),
                    "Tamanho": tamanhos,
                }
            )
            st.write("Tamanho dos datasets:")
            st.dataframe(metr)
        return None

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
            col = st.columns((2, 2, 0.05, 2, 0.05, 2))
            with col[0]:
                # Create a dropdown menu
                variavel_1 = st.radio(
                    "Primeira variável:", ["n_tetraciclina", "n_polimixinas", "n_degradacao_aromaticos"]
                )
                variavel_2 = st.radio(
                    "Segunda variável:", ["n_polimixinas", "n_degradacao_aromaticos", "n_tetraciclina"]
                )

            resultados, msn_kmenans, msn_dbscan, msn_agg = self.__clustering(
                variavel_1=variavel_1,
                variavel_2=variavel_2,
                df_train=df_train,
                df_test=df_test,
            )
            with col[1]:
                # Testando os modelos de clustering via K-means
                st.markdown(f"<p>{msn_kmenans}</p>", unsafe_allow_html=True)
                st.pyplot(fig=resultados["K-means"])

            with col[3]:
                # Testando os modelos de clustering via DBSCAN
                st.markdown(f"<p>{msn_dbscan}</p>", unsafe_allow_html=True)
                st.pyplot(fig=resultados["DBSCAN"])

            with col[5]:
                # Testando os modelos de clustering via Agglomerative
                st.markdown(f"<p>{msn_agg}</p>", unsafe_allow_html=True)
                st.pyplot(fig=resultados["Agg"])

        with abas[1]:
            self.__classifier()

        with abas[2]:
            self.__regressor()

        with abas[3]:
            self.__datasets()

        return "Métricas Calculadas"
