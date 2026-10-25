# Métricas de avaliação de clustering
from sklearn.metrics import adjusted_rand_score

# Bibliotecas para clustering
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# Testagem dos modelos
from scipy.stats import chisquare
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

# Bibliotecas para o modelo de clustering
import numpy as np
import pandas as pd

# Geração de gráficos
import seaborn as sns


class Clustering:
    """
    Classe para realizar clustering em uma bivariada de um DataFrame.
    """

    def __init__(self, c1: str, c2: str, df: pd.DataFrame, modelo: str) -> None:
        self.c1 = c1
        self.c2 = c2
        self.df = df
        self.modelo = modelo

        return None

    def grafico_scatter(
        self,
        hue: str | list | np.ndarray,
        df_grafico: pd.DataFrame,
        jitter: float = 0.08,
    ) -> sns.JointGrid:
        """
        Gráfico de dispersão (jointplot) com jitter, para variáveis de
        contagem, colorido pelos clusters (hue).
        """
        dados = df_grafico[[self.c1, self.c2]].copy().astype(float)
        dados["grupo"] = df_grafico["grupo"].values
        dados = dados.dropna().copy()

        # Valores reais (antes do jitter), usados nos ticks
        ticks_x = np.sort(dados[self.c1].unique())
        ticks_y = np.sort(dados[self.c2].unique())

        # Ruído pequeno para reduzir a sobreposição dos pontos
        rng = np.random.default_rng(42)
        dados[self.c1] += rng.normal(0, jitter, len(dados))
        dados[self.c2] += rng.normal(0, jitter, len(dados))

        p = sns.jointplot(
            data=dados,
            x=self.c1,
            y=self.c2,
            hue="grupo",
            height=6,
            ratio=4,
            marginal_ticks=True,
            space=0.05,
        )
        p.ax_joint.set_xticks(ticks_x)
        p.ax_joint.set_yticks(ticks_y)
        p.ax_joint.set_xticklabels([f"{int(v)}" for v in ticks_x])
        p.ax_joint.set_yticklabels([f"{int(v)}" for v in ticks_y])
        p.fig.suptitle(f"{self.c1} × {self.c2}", y=1.02)
        p.set_axis_labels(xlabel=self.c1, ylabel=self.c2)
        return p

    def __testar_kmeans(
        self, best_param: dict, df_test: pd.DataFrame
    ) -> tuple[sns.JointGrid, str]:
        """
        Função para testar a consistência dos clusters obtidos no conjunto
        de treino com os clusters obtidos no conjunto de validação,
        utilizando o teste Qui-Quadrado de Aderência e o
        Índice de Rand Ajustado (ARI).
            Parâmetros:
                best_param: dict
                    Dicionário contendo os melhores parâmetros para
                    o modelo
                    de clustering.
                df_test: pd.DataFrame, opcional
                    DataFrame contendo os dados de validação. Padrão
                    é df_test.
            Retorna:
                bool - Resultado do teste de consistência dos clusters.
        """
        # Modelo treinado no Treino
        kmeans_train = KMeans(**best_param).fit(self.df[[self.c1, self.c2]])
        labels_train = kmeans_train.labels_

        # Previsão dos clusters da Validação usando os centroides do Treino
        labels_val_predicted = kmeans_train.predict(df_test[[self.c1, self.c2]])

        # Modelo independente treinado diretamente na Validação
        # (para comparação via ARI)
        kmeans_val_direct = KMeans(**best_param).fit(df_test[[self.c1, self.c2]])
        labels_val_direct = kmeans_val_direct.labels_

        # 4. Cálculo das Frequências
        # Proporções observadas no Treino (proporções esperadas)
        train_counts = pd.Series(labels_train).value_counts().sort_index()
        train_proportions = train_counts / len(self.df[[self.c1, self.c2]])

        # Frequências esperadas na Validação (Proporção do Treino * N_val)
        expected_val_counts = train_proportions * len(df_test[[self.c1, self.c2]])

        # Frequências observadas reais na Validação
        observed_val_counts = (
            pd.Series(labels_val_predicted)
            .value_counts()
            .reindex(range(best_param["n_clusters"]), fill_value=0)
            .sort_index()
        )

        # 5. Execução do Teste Qui-Quadrado de Aderência
        chi2_stat, p_value = chisquare(
            f_obs=observed_val_counts, f_exp=expected_val_counts
        )

        # 6. Avaliação de Alinhamento Estrutural via Adjusted Rand Index (ARI)
        ari_score = adjusted_rand_score(labels_val_predicted, labels_val_direct)

        # --- Exibição dos Resultados ---
        msn_resul = "=" * 50
        msn_resul += "<br>"
        msn_resul += "<h6 style='text-align: center;'>K-means</h6>"
        msn_resul += "=" * 50 + "<br>"
        msn_resul += "Clusters identificados no Treino:       "
        msn_resul += f"{np.unique(labels_train)}<br><br>"
        msn_resul += "Frequências Esperadas na Validação (Treino): <br>"
        msn_resul += f"{expected_val_counts.values}<br><br>"
        msn_resul += "Frequências Observadas na Validação: <br>"
        msn_resul += f"       {observed_val_counts.values}<br>"
        msn_resul += "-" * 50 + "<br>"
        msn_resul += f"Estatística Qui-Quadrado (χ²): {chi2_stat:.4f}<br>"
        msn_resul += f"p-valor:                        {p_value:.4f}<br>"
        msn_resul += "-" * 50 + "<br>"
        msn_resul += "<br>"

        if p_value > 0.05:
            mensagem = "Conclusão 1: Não rejeitamos H0. A distribuição dos "
            mensagem += "clusters na validação "
            mensagem += "<b style='color: green;'>É IGUAL</b>"
            mensagem += " à do treino (p > 0.05)."
            mensagem += "<br>"
            msn_resul += mensagem
        else:
            mensagem = "Conclusão:  2Rejeitamos H0. A distribuição dos "
            mensagem += "clusters na validação "
            mensagem += "<b style='color: red;'>É DIFERENTE</b>"
            mensagem += " da do treino (p <= 0.05)."
            mensagem += "<br>"
            msn_resul += mensagem
        msn_resul += "<br>" + "-" * 50 + "<br>"
        msn_resul += "<br>"

        if ari_score > 0.5:
            mensagem = "Conclusão: O modelo de clustering do Treino e o "
            mensagem += "modelo da Validação "
            mensagem += "<b style='color: green;'>SÃO SEMELHANTES</b>"
            mensagem += f" (ARI = {ari_score:.4f})."
            mensagem += "<br>"
            msn_resul += mensagem
        else:
            mensagem = "Conclusão: O modelo de clustering do Treino e o "
            mensagem += "modelo da Validação "
            mensagem += "<b style='color: red;'>NÃO SÃO SEMELHANTES</b>"
            mensagem += f" (ARI = {ari_score:.4f})."
            mensagem += "<br>"
            msn_resul += mensagem

        msn_resul += "<br>"
        grafico = self.grafico_scatter(hue=labels_val_predicted, df_grafico=df_test)

        return grafico, msn_resul

    def __testar_dbscan(
        self, best_param: dict, df_test: pd.DataFrame
    ) -> tuple[sns.JointGrid, str]:
        """
        Função para testar a consistência dos clusters obtidos no conjunto
        de treino com os clusters obtidos no conjunto de validação,
        utilizando o teste Qui-Quadrado de Aderência e o
        Índice de Rand Ajustado (ARI).
            Parâmetros:
                self.c1: str - Nome da primeira variável a ser utilizada
                no clustering.
                self.c2: str - Nome da segunda variável a ser utilizada
                no clustering.
                best_param: dict
                    Dicionário contendo os melhores parâmetros para
                    o modelo
                    de clustering.
                self.df: pd.DataFrame, opcional
                    DataFrame contendo os dados de treino. Padrão
                    é self.df.
                df_test: pd.DataFrame, opcional
                    DataFrame contendo os dados de validação. Padrão
                    é df_test.
            Retorna:
                bool - Resultado do teste de consistência dos clusters.
        """
        dbscan_train = DBSCAN(**best_param).fit(self.df[[self.c1, self.c2]])
        labels_train = dbscan_train.labels_

        # 4. Projeção dos clusters do Treino para a Validação via KNN
        # Treina-se um modelo de K-Vizinhos Mais Próximos (k=1) com os
        # rótulos do DBSCAN no Treino
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(self.df[[self.c1, self.c2]], labels_train)
        labels_val_predicted = knn.predict(df_test[[self.c1, self.c2]])

        # DBSCAN executado de forma independente na Validação
        # (para avaliação de estrutura via ARI)
        dbscan_val_direct = DBSCAN(**best_param).fit(df_test[[self.c1, self.c2]])
        labels_val_direct = dbscan_val_direct.labels_

        # 5. Mapeamento e Contagem de Frequências (incluindo o rótulo de
        # ruído -1, se houver)
        all_clusters = np.unique(np.concatenate([labels_train, labels_val_predicted]))

        # Proporções no Treino
        train_counts = (
            pd.Series(labels_train)
            .value_counts()
            .reindex(all_clusters, fill_value=0)
            .sort_index()
        )
        train_proportions = train_counts / len(self.df[[self.c1, self.c2]])

        # Frequências esperadas na Validação (Proporção do Treino * N_val)
        expected_val_counts = train_proportions * len(df_test[[self.c1, self.c2]])

        # Frequências observadas na Validação
        observed_val_counts = (
            pd.Series(labels_val_predicted)
            .value_counts()
            .reindex(all_clusters, fill_value=0)
            .sort_index()
        )

        # 6. Execução do Teste Qui-Quadrado de Aderência
        chi2_stat, p_value = chisquare(
            f_obs=observed_val_counts, f_exp=expected_val_counts
        )

        # 7. Avaliação de Estrutura via Adjusted Rand Index (ARI)
        ari_score = adjusted_rand_score(labels_val_predicted, labels_val_direct)

        # --- Exibição dos Resultados ---
        msn_resul = "=" * 50 + "<br>"
        msn_resul += "<h6 style='text-align: center;'>DBSCAN</h6>"
        msn_resul += "=" * 50 + "<br>"
        msn_resul += "Clusters identificados no Treino:       "
        msn_resul += f"{np.unique(labels_train)}<br><br>"
        msn_resul += "Frequências Esperadas na Validação (Treino): <br>"
        msn_resul += f"{expected_val_counts.values}<br><br>"
        msn_resul += "Frequências Observadas na Validação: <br>"
        msn_resul += f"       {observed_val_counts.values}<br>"
        msn_resul += "-" * 50 + "<br>"
        msn_resul += f"Estatística Qui-Quadrado (χ²): {chi2_stat:.4f}<br>"
        msn_resul += f"p-valor:                        {p_value:.4f}<br>"
        msn_resul += "-" * 50 + "<br>"
        msn_resul += "<br>"

        if p_value > 0.05:
            mensagem = "Conclusão 1: Não rejeitamos H0. A distribuição dos "
            mensagem += "clusters na validação "
            mensagem += "<b style='color: green;'>É IGUAL</b>"
            mensagem += " à do treino (p > 0.05)."
            mensagem += "<br>"
            msn_resul += mensagem
        else:
            mensagem = "Conclusão:  2Rejeitamos H0. A distribuição dos "
            mensagem += "clusters na validação "
            mensagem += "<b style='color: red;'>É DIFERENTE</b>"
            mensagem += " da do treino (p <= 0.05)."
            mensagem += "<br>"
            msn_resul += mensagem
        msn_resul += "<br>" + "-" * 50 + "<br>"
        msn_resul += "<br>"

        if ari_score > 0.5:
            mensagem = "Conclusão: O modelo de clustering do Treino e o "
            mensagem += "modelo da Validação "
            mensagem += "<b style='color: green;'>SÃO SEMELHANTES</b>"
            mensagem += f" (ARI = {ari_score:.4f})."
            mensagem += "<br>"
            msn_resul += mensagem
        else:
            mensagem = "Conclusão: O modelo de clustering do Treino e o "
            mensagem += "modelo da Validação "
            mensagem += "<b style='color: red;'>NÃO SÃO SEMELHANTES</b>"
            mensagem += f" (ARI = {ari_score:.4f})."
            mensagem += "<br>"
            msn_resul += mensagem

        msn_resul += "<br>"
        grafico = self.grafico_scatter(hue=labels_val_predicted, df_grafico=df_test)

        return grafico, msn_resul

    def __testar_agg(
        self, best_param: dict, df_test: pd.DataFrame
    ) -> tuple[sns.JointGrid, str]:
        """
        Função para testar a consistência dos clusters obtidos no conjunto
        de treino com os clusters obtidos no conjunto de validação,
        utilizando o
        teste Qui-Quadrado de Aderência e o Índice de Rand Ajustado (ARI).
            Parâmetros:
                self.c1: str - Nome da primeira variável a ser utilizada
                no clustering.
                self.c2: str - Nome da segunda variável a ser utilizada
                no clustering.
                best_param: dict
                    Dicionário contendo os melhores parâmetros para
                    o modelo
                    de clustering.
                self.df: pd.DataFrame, opcional
                    DataFrame contendo os dados de treino.
                    Padrão é self.df.
                df_test: pd.DataFrame, opcional
                    DataFrame contendo os dados de validação.
                    Padrão é df_test.
            Retorna:
                bool - Resultado do teste de consistência dos clusters.
        """
        agg_train = AgglomerativeClustering(**best_param)
        labels_train = agg_train.fit_predict(self.df[[self.c1, self.c2]])

        # 4. Projeção dos clusters do Treino para a Validação via Centroides
        # Calcula o centroide de cada cluster hierárquico e atribui os dados
        # de validação ao centroide mais próximo
        clf_centroid = NearestCentroid()
        clf_centroid.fit(self.df[[self.c1, self.c2]], labels_train)
        labels_val_predicted = clf_centroid.predict(df_test[[self.c1, self.c2]])

        # AgglomerativeClustering executado diretamente na Validação
        # (para avaliação estrutural via ARI)
        agg_val_direct = AgglomerativeClustering(**best_param)
        labels_val_direct = agg_val_direct.fit_predict(df_test[[self.c1, self.c2]])

        # 5. Cálculo das Frequências
        all_clusters = np.unique(labels_train)

        # Proporções no Treino
        train_counts = (
            pd.Series(labels_train)
            .value_counts()
            .reindex(all_clusters, fill_value=0)
            .sort_index()
        )
        train_proportions = train_counts / len(self.df[[self.c1, self.c2]])

        # Frequências esperadas na Validação (Proporção do Treino * N_val)
        expected_val_counts = train_proportions * len(df_test[[self.c1, self.c2]])

        # Frequências observadas na Validação
        observed_val_counts = (
            pd.Series(labels_val_predicted)
            .value_counts()
            .reindex(all_clusters, fill_value=0)
            .sort_index()
        )

        # 6. Execução do Teste Qui-Quadrado de Aderência
        chi2_stat, p_value = chisquare(
            f_obs=observed_val_counts, f_exp=expected_val_counts
        )

        # 7. Avaliação de Estrutura via Adjusted Rand Index (ARI)
        ari_score = adjusted_rand_score(labels_val_predicted, labels_val_direct)

        # --- Exibição dos Resultados ---
        msn_resul = "=" * 50 + "<br>"
        msn_resul += "<h6 style='text-align: center;'>AGGLOMERATIVE</h6>"
        msn_resul += "=" * 50 + "<br>"
        msn_resul += "Clusters identificados no Treino:       "
        msn_resul += f"{np.unique(labels_train)}<br><br>"
        msn_resul += "Frequências Esperadas na Validação (Treino): <br>"
        msn_resul += f"{expected_val_counts.values}<br><br>"
        msn_resul += "Frequências Observadas na Validação: <br>"
        msn_resul += f"       {observed_val_counts.values}<br>"
        msn_resul += "-" * 50 + "<br>"
        msn_resul += f"Estatística Qui-Quadrado (χ²): {chi2_stat:.4f}<br>"
        msn_resul += f"p-valor:                        {p_value:.4f}<br>"
        msn_resul += "-" * 50 + "<br>"
        msn_resul += "<br>"

        if p_value > 0.05:
            mensagem = "Conclusão 1: Não rejeitamos H0. A distribuição dos "
            mensagem += "clusters na validação "
            mensagem += "<b style='color: green;'>É IGUAL</b>"
            mensagem += " à do treino (p > 0.05)."
            mensagem += "<br>"
            msn_resul += mensagem
        else:
            mensagem = "Conclusão 2: Rejeitamos H0. A distribuição dos "
            mensagem += "clusters na validação "
            mensagem += "<b style='color: red;'>É DIFERENTE</b>"
            mensagem += " da do treino (p <= 0.05)."
            mensagem += "<br>"
            msn_resul += mensagem
        msn_resul += "<br>" + "-" * 50 + "<br>"
        msn_resul += "<br>"

        if ari_score > 0.5:
            mensagem = "Conclusão: O modelo de clustering do Treino e o "
            mensagem += "modelo da Validação "
            mensagem += "<b style='color: green;'>SÃO SEMELHANTES</b>"
            mensagem += f" (ARI = {ari_score:.4f})."
            mensagem += "<br>"
            msn_resul += mensagem
        else:
            mensagem = "Conclusão: O modelo de clustering do Treino e o "
            mensagem += "modelo da Validação "
            mensagem += "<b style='color: red;'>NÃO SÃO SEMELHANTES</b>"
            mensagem += f" (ARI = {ari_score:.4f})."
            mensagem += "<br>"
            msn_resul += mensagem

        msn_resul += "<br>"
        grafico = self.grafico_scatter(hue=labels_val_predicted, df_grafico=df_test)

        return grafico, msn_resul

    def testar_modelo(
        self, best_param: dict, df_test: pd.DataFrame
    ) -> tuple[sns.JointGrid, str]:
        """
        Função para testar a consistência dos clusters obtidos no conjunto
        de treino com os clusters obtidos no conjunto de validação,
        utilizando o teste Qui-Quadrado de Aderência e o
        Índice de Rand Ajustado (ARI).
            Parâmetros:
                best_param: dict
                    Dicionário contendo os melhores parâmetros para o
                    modelo
                    de clustering.
                df_test: pd.DataFrame, opcional
                    DataFrame contendo os dados de validação. Padrão é
                    df_test.
            Retorna:
                bool: True se a distribuição dos clusters na validação é
                consistente com a do treino, False caso contrário.
        """
        if self.modelo == "kmeans":
            grafico, msn_resul = self.__testar_kmeans(
                best_param=best_param, df_test=df_test
            )
        elif self.modelo == "dbscan":
            grafico, msn_resul = self.__testar_dbscan(
                best_param=best_param, df_test=df_test
            )
        elif self.modelo == "agglomerative":
            grafico, msn_resul = self.__testar_agg(
                best_param=best_param, df_test=df_test
            )

        return grafico, msn_resul
