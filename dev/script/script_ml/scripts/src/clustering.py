# Métricas de avaliação de clustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score

# Bibliotecas para clustering
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score

# Testagem dos modelos
from scipy.stats import chisquare
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

# Bibliotecas para o modelo de clustering
import numpy as np
import pandas as pd

# Geração de gráficos
import matplotlib.pyplot as plt
import seaborn as sns


def silhouette_scorer_func(estimator, X) -> float:
    """
    Função para calcular a pontuação de Silhouette para um estimador de
    clustering.
        Parâmetros:
            estimator: Estimador de clustering (ex: KMeans)
            X: Dados de entrada para o cálculo da pontuação
        Retorna:
            float: Pontuação de Silhouette média para os dados fornecidos
    """
    labels = estimator.fit_predict(X)
    # Silhouette precisa de pelo menos 2 clusters distintos no conjunto
    # de validação
    if len(np.unique(labels)) > 1:
        return silhouette_score(X, labels)
    return -1.0  # Retorna pontuação baixa caso haja falha de separação no fold


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

    def __calcular_metricas(
        self, X_scaled: np.ndarray, pred: np.ndarray, obs: np.ndarray
    ) -> tuple:
        """
        Função para calcular métricas de avaliação de clustering.
            Parâmetros:
                X_scaled: np.array - Dados escalonados utilizados para o
                clustering
                pred: np.array - Rótulos previstos pelo modelo de
                clustering
                obs: np.array - Rótulos observados (verdadeiros) para
                comparação
            Retorna:
                None - Apenas imprime as métricas calculadas
        """
        ss = np.nan
        chs = np.nan
        dbs = np.nan
        ari = np.nan
        ami = np.nan

        try:
            # Calcula a métrica de Silhouette
            ss = round(silhouette_score(X_scaled, pred), 3)
            # Calcula a métrica de Calinski-Harabasz
            chs = round(calinski_harabasz_score(X_scaled, pred), 3)

            # Calcula a métrica de Davies-Bouldin
            dbs = round(davies_bouldin_score(X_scaled, pred), 3)

            # Calcula a métrica de Adjusted Rand Score
            ari = round(adjusted_rand_score(obs, pred), 3)

            # Calcula a métrica de Adjusted Mutual Info Score
            ami = round(adjusted_mutual_info_score(obs, pred), 3)
        except Exception:
            pass

        return (ss, chs, dbs, ari, ami)

    def __grafico_scatter(self, hue: str | list | np.ndarray, jitter: float = 0.08) -> None:
        """
        Gera um gráfico de dispersão (jointplot) com jitter, a partir de
        duas variáveis de contagem e uma variável de agrupamento (hue).
            Parâmetros:
                hue: str|list|np.array - Nome da coluna do DataFrame ou
                                        array com os valores de agrupamento.
                jitter: float - Desvio padrão do ruído aplicado aos pontos.
            Retorna:
                None
        """
        if isinstance(hue, str):
            hue_col = hue
            dados_plot = self.df[[self.c1, self.c2, hue_col]].copy()
        else:
            hue_col = "grupo"
            dados_plot = self.df[[self.c1, self.c2]].copy()
            dados_plot[hue_col] = np.asarray(hue)

        dados_plot = dados_plot.dropna().copy()
        dados_plot[self.c1] = dados_plot[self.c1].astype(float)
        dados_plot[self.c2] = dados_plot[self.c2].astype(float)

        # Valores reais de contagem (antes do jitter), usados nos ticks
        ticks_x = np.sort(dados_plot[self.c1].unique())
        ticks_y = np.sort(dados_plot[self.c2].unique())

        # Pequeno jitter para reduzir a sobreposição dos pontos
        np.random.seed(42)
        dados_plot[self.c1] += np.random.normal(0, jitter, len(dados_plot))
        dados_plot[self.c2] += np.random.normal(0, jitter, len(dados_plot))

        p = sns.jointplot(
            data=dados_plot,
            x=self.c1,
            y=self.c2,
            hue=hue_col,
            height=6,
            ratio=4,
            marginal_ticks=True,
            space=0.05,
        )

        # Ticks nos valores observados, sem fixar o intervalo
        p.ax_joint.set_xticks(ticks_x)
        p.ax_joint.set_yticks(ticks_y)
        p.ax_joint.set_xticklabels([f"{int(v)}" for v in ticks_x])
        p.ax_joint.set_yticklabels([f"{int(v)}" for v in ticks_y])

        p.fig.suptitle(f"{self.c1} × {self.c2}", y=1.02)
        p.set_axis_labels(xlabel=self.c1, ylabel=self.c2)

        plt.show()
        return None

    def __grafico_box(self, cluster_labels: np.ndarray) -> None:
        _, ax = plt.subplots(1, 2, figsize=(13, 5))
        plt.suptitle("Diferença das classes clusterizadas")
        g = sns.boxplot(x=cluster_labels, y=self.c1, data=self.df, ax=ax[0])
        g.set(
            title=f" {self.c1} para Clusters",
            # xlabel = cluster_labels,
            ylabel=self.c1,
        )
        g = sns.boxplot(x=cluster_labels, y=self.c2, data=self.df, ax=ax[1])
        g.set(
            title=f" {self.c2} para Clusters",
            # xlabel = cluster_labels,
            ylabel=self.c2,
        )

        return None

    def __param_grid(self) -> dict:
        if self.modelo == "kmeans":
            retorno = {
                "n_clusters": [2, 3, 4, 5, 6],
                "init": ["k-means++", "random"],
                "n_init": ["auto", 10, 20, 30],
            }
        elif self.modelo == "dbscan":
            retorno = {
                "eps": [0.1, 0.2, 0.3, 0.4, 0.5],
                "min_samples": [3, 5, 10, 15],
                "metric": ["euclidean", "manhattan"],
            }
        elif self.modelo == "agglomerative":
            retorno = {
                "n_clusters": [2, 3, 4, 5, 6],
                "metric": ["euclidean"],
                "linkage": ["ward", "complete", "average", "single"],
            }
        return retorno

    def cross_validation(
        self, n_splits: int = 5, random_state: int = 42, print_metricas: bool = True
    ) -> None | tuple:
        """
        Função para realizar validação cruzada utilizando a pontuação de
        Silhouette para um par de variáveis em um DataFrame.
            Parâmetros:
                self.c1: str - Nome da primeira variável
                self.c2: str - Nome da segunda variável
                self.df: pd.DataFrame - DataFrame contendo as variáveis
                n_splits: int - Número de folds para a validação cruzada.
                                Default é 5.
                random_state: int - Semente para reprodução dos resultados.
                                Default é 42.
                print_metricas: bool - Se True, imprime as métricas de
                                avaliação do clustering. Default é True.
            Retorna:
                None
        """
        # Preparação dos Dados Sintéticos
        X_raw = self.df[[self.c1, self.c2]]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

        # Grid Search Manual integrando cross_val_score

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        best_score = -1.0
        best_params = {}
        results = []

        param_grid = self.__param_grid()
        param_grid_keys = list(param_grid.keys())
        # Iteração sobre a grade de hiperparâmetros
        for a in param_grid[param_grid_keys[0]]:
            for b in param_grid[param_grid_keys[1]]:
                for c in param_grid[param_grid_keys[2]]:
                    if self.modelo == "kmeans":
                        # Instância do modelo com a combinação atual
                        model = KMeans(
                            n_clusters=a,
                            init=b,
                            n_init=c,
                            random_state=random_state,
                        )
                    elif self.modelo == "dbscan":
                        # Instância do modelo com a combinação atual
                        model = DBSCAN(eps=a, min_samples=b, metric=c, n_jobs=-1)
                    elif self.modelo == "agglomerative":
                        model = AgglomerativeClustering(
                            n_clusters=a,
                            metric=b,
                            linkage=c,
                        )
                    # Execução do cross_val_score do scikit-learn
                    scores = cross_val_score(
                        estimator=model,
                        X=X_scaled,
                        cv=kf,
                        # Usando a função scorer
                        scoring=silhouette_scorer_func,
                        n_jobs=-1,  # Paralelização dos folds
                    )

                    mean_score = np.mean(scores)
                    std_score = np.std(scores)

                    results.append(
                        {
                            param_grid_keys[0]: a,
                            param_grid_keys[1]: b,
                            param_grid_keys[2]: c,
                            "mean_score": mean_score,
                            "std_score": std_score,
                        }
                    )

                    if mean_score > best_score:
                        best_score = float(mean_score)
                        best_params = {
                            param_grid_keys[0]: a,
                            param_grid_keys[1]: b,
                            param_grid_keys[2]: c,
                        }

        # Treinamento do modelo final no dataset completo com os melhores
        # hiperparâmetros
        if self.modelo == "kmeans":
            best_model = KMeans(**best_params, random_state=random_state)
        elif self.modelo == "dbscan":
            best_model = DBSCAN(**best_params, n_jobs=-1)
        elif self.modelo == "agglomerative":
            best_model = AgglomerativeClustering(**best_params)

        cluster_labels = best_model.fit_predict(X_scaled)

        # Mostra as métricas de avaliação do clustering
        metricas = {
            "X_scaled": X_scaled,
            "pred": cluster_labels,
            "obs": np.array(self.df["grupo"].values),
        }
        if print_metricas:
            # Resultados e Ajuste Final
            print("=== MELHOR CONFIGURAÇÃO ENCONTRADA VIA cross_val_score ===")
            print(f"Parâmetros: {best_params}")
            print(f"Silhouette Score Médio (CV): {best_score:.4f}\n")

            # Calcular métricas de avaliação do clustering com os melhores
            # hiperparâmetros

            ss, chs, dbs, ari, ami = self.__calcular_metricas(**metricas)
            print(f"Silhouette Score -> {ss}")
            print(f"Calinski Harabasz Score -> {chs}")
            print(f"Davies Bouldin Score -> {dbs}")
            print(f"Adjusted Rand Score -> {ari}")
            print(f"Adjusted Mutual Info Score -> {ami}")

            # Graficos de relação entre as variáveis e os clusters
            self.__grafico_scatter(hue="grupo")
            self.__grafico_scatter(hue=cluster_labels)
            self.__grafico_box(cluster_labels=cluster_labels)
            plt.show()

            return None
        return cluster_labels, self.__calcular_metricas(**metricas)

    def __testar_kmeans(self, best_param: dict, df_test: pd.DataFrame) -> bool:
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
        print("=" * 50)
        print(" RESULTADOS DA ANÁLISE DE COMPARAÇÃO DE CLUSTERS")
        print("=" * 50)
        print("Frequências Esperadas na Validação (Treino): ")
        print(f"{expected_val_counts.values}")
        print("Frequências Observadas na Validação:")
        print(f"       {observed_val_counts.values}")
        print("-" * 50)
        print(f"Estatística Qui-Quadrado (χ²): {chi2_stat:.4f}")
        print(f"p-valor:                        {p_value:.4f}")
        print("-" * 50)

        alpha = 0.05
        if p_value > alpha:
            print("Conclusão: Não rejeitamos H0.", end=" ")
            print("A distribuição dos clusters na validação", end=" ")
            print("É IGUAL à do treino (p > 0.05).")
            retorno = True
        else:
            print("Conclusão: Rejeitamos H0. ", end=" ")
            print("A distribuição dos clusters na validação", end=" ")
            print("É DIFERENTE da do treino (p <= 0.05).")
            retorno = False

        print("-" * 50)
        mensagem = "Adjusted Rand Index (ARI) entre modelo de Treino e modelo "
        mensagem += f"da Validação: {ari_score:.4f} (ARI próximo de 1.0"
        mensagem += " indica que a partição geométrica gerada é"
        mensagem += " virtualmente idêntica)"
        print(mensagem)

        return retorno

    def __testar_dbscan(self, best_param: dict, df_test: pd.DataFrame) -> bool:
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
        print("=" * 55)
        print(" RESULTADOS DA COMPARAÇÃO DE CLUSTERS (DBSCAN)")
        print("=" * 55)
        print("Clusters identificados no Treino:       ")
        print(f"{np.unique(labels_train)}")
        print("Frequências Esperadas na Validação:     ")
        print(f"{expected_val_counts.values.round(2)}")
        print("Frequências Observadas na Validação:    ")
        print(f"{observed_val_counts.values}")
        print("-" * 55)
        print(f"Estatística Qui-Quadrado (χ²): {chi2_stat:.4f}")
        print(f"p-valor:                        {p_value:.4f}")
        print("-" * 55)

        alpha = 0.05
        if p_value > alpha:
            print("Conclusão: Não rejeitamos H0. A distribuição dos clusters")
            print("na validação É IGUAL à do treino (p > 0.05).")
            retorno = True
        else:
            mensagem = "Conclusão: Rejeitamos H0. A distribuição dos "
            mensagem += "clusters na validação É DIFERENTE da do treino  "
            mensagem += "(p <= 0.05)."
            print(mensagem)
            retorno = False

        print("-" * 55)
        print(
            f"Adjusted Rand Index (ARI) entre predição e DBSCAN direto na "
            f"Validação: {ari_score:.4f}"
        )

        return retorno

    def __testar_agg(self, best_param: dict, df_test: pd.DataFrame) -> bool:
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
        print("=" * 60)
        print(" RESULTADOS DA COMPARAÇÃO (AGGLOMERATIVE CLUSTERING)")
        print("=" * 60)
        print("Clusters no Treino:                  ")
        print(f"{all_clusters}")
        print("Frequências Esperadas na Validação:  ")
        print(f"{expected_val_counts.values.round(2)}")
        print("Frequências Observadas na Validação:  ")
        print(f"{observed_val_counts.values}")
        print("-" * 60)
        print(f"Estatística Qui-Quadrado (χ²): {chi2_stat:.4f}")
        print(f"p-valor:                        {p_value:.4f}")
        print("-" * 60)

        alpha = 0.05
        if p_value > alpha:
            mensagem = "Conclusão: Não rejeitamos H0. A distribuição dos "
            mensagem += "clusters na validação É IGUAL à do treino (p > 0.05)."
            print(mensagem)
            retorno = True
        else:
            mensagem = "Conclusão: Rejeitamos H0. A distribuição dos clusters "
            mensagem += "na validação É DIFERENTE da do treino (p <= 0.05)."
            print(mensagem)
            retorno = False

        print("-" * 60)
        print(
            f"Adjusted Rand Index (ARI) entre projeção e agrupamento "
            f"direto: {ari_score:.4f}"
        )

        return retorno

    def testar_modelo(self, best_param: dict, df_test: pd.DataFrame) -> bool:
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
                    Dicionário contendo os melhores parâmetros para o
                    modelo
                    de clustering.
                self.df: pd.DataFrame, opcional
                    DataFrame contendo os dados de treino. Padrão é
                    self.df.
                df_test: pd.DataFrame, opcional
                    DataFrame contendo os dados de validação. Padrão é
                    df_test.
            Retorna:
                bool: True se a distribuição dos clusters na validação é
                consistente com a do treino, False caso contrário.
        """
        if self.modelo == "kmeans":
            retorno = self.__testar_kmeans(best_param=best_param, df_test=df_test)
        elif self.modelo == "dbscan":
            retorno = self.__testar_dbscan(best_param=best_param, df_test=df_test)
        elif self.modelo == "agglomerative":
            retorno = self.__testar_agg(best_param=best_param, df_test=df_test)

        return retorno
