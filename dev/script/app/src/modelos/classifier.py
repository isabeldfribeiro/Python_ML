import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)


class Classifier:
    """
    Classificador genérico que encapsula diferentes modelos de
    aprendizado de máquina.
        Parâmetros
        ----------
        df : pd.DataFrame
            DataFrame contendo as features e a coluna alvo 'classe'.

        Métodos
        -------
        classify(modelo)
            Treina e avalia o modelo especificado.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

        return None

    def testar_hipotese(self) -> None:
        """
        Testa a hipótese de que a distribuição das classes no treino e no
        teste é estatisticamente igual (homogênea)
        ou diferente (heterogênea) usando o teste Qui-Quadrado.

        Parâmetros:
            df : pd.DataFrame
                DataFrame contendo as colunas 'Dataset' e 'Classe_Predita'.
            modelo : str
                Nome do modelo para exibição nos resultados.
        Retorna:
            None
        """
        # 3. Criando a Tabela de Contingência (Frequências Observadas)
        tabela_contingencia = pd.crosstab(
            self.df["Dataset"],
            self.df["Classe_Predita"],
        )
        st.write("Tabela de Contingência (Observada):")
        st.dataframe(tabela_contingencia.T)
        st.write("-" * 40)

        # 4. Realizando o Teste Qui-Quadrado
        chi2, p_valor, graus, _ = chi2_contingency(tabela_contingencia)

        # 5. Exibindo os resultados
        st.write(f"Estatística Qui-Quadrado: {chi2:.4f}")
        st.write(f"Graus de Liberdade: {graus}")
        st.write(f"Valor-p: {p_valor:.4f}")

        # 6. Tomada de Decisão (usando nível de significância de 5%)
        alpha = 0.05
        st.write("\nConclusão:")
        if p_valor < alpha:
            mensagem = "Rejeitamos H0: "
            mensagem += "A distribuição das classes no treino e no teste é "
            mensagem += "<b style='color: red;'>ESTATISTICAMENTE DIFERENTE "
            mensagem += "(Heterogênea).</b>"
            st.markdown(mensagem, unsafe_allow_html=True)
        else:
            mensagem = "Falhamos em rejeitar H0: "
            mensagem += "A distribuição das classes no treino e no teste é "
            mensagem += "<b style='color: green;'>ESTATISTICAMENTE IGUAL "
            mensagem += "(Homogênea).</b>"
            st.markdown(mensagem, unsafe_allow_html=True)

        return None

    def matrix_confusao(self, df: pd.DataFrame, obs: str, pred: str) -> None:
        """
        Exibe a matriz de confusão e o classification report.
            Parâmetros
            ----------
            df : pd.DataFrame
                DataFrame com a classe observada e a classe predita.
            obs : str
                Nome da coluna com a classe observada.
            pred : str
                Nome da coluna com a classe predita.
        """
        y_obs = df[obs]
        y_pred = df[pred]

        # Mesma ordem de classes na matriz e nos rótulos dos eixos
        labels = sorted(y_obs.unique())
        accuracy = accuracy_score(y_obs, y_pred)
        cm = confusion_matrix(y_obs, y_pred, labels=labels)

        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, cmap=plt.cm.Greens)
        ax.set_title(f"Accuracy {accuracy:.2f}")
        st.pyplot(fig)
        plt.close(fig)

        st.write("\nClassification Report:")
        st.dataframe(
            pd.DataFrame(classification_report(y_obs, y_pred, output_dict=True)).T
        )

        return None
