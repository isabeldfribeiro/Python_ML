import numpy as np
import pandas as pd
from sklearn import metrics
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay


class Avalia_modelo:
    """
    Classe que avalia o modelo de classificação
    """

    def __init__(
        self,
        df: pd.DataFrame,
        indep: pd.Index,
        modelo,
        target: str = "target",
        troca_classe: bool = False,
    ) -> None:
        self.df = df
        self.target = target
        self.indep = indep
        self.modelo = modelo
        self.observed = df.loc[:, target]
        self.troca_classe = troca_classe
        if self.troca_classe:
            self.predito = self.__retorna_classe(modelo.predict(df.loc[:, indep]))
        else:
            self.predito = modelo.predict(df.loc[:, indep])

    def __retorna_classe(self, pred: pd.Series) -> list:
        """
        Função que retorna a classe de predição
        :param pred (pd.Series): Previsão do modelo
        :return (list): Classe de predição
        """
        retorno = [0 if i <= 0.5 else 1 for i in pred]
        return retorno

    def __matrix_confusao(self) -> None:
        accuracy = accuracy_score(self.observed, self.predito)
        cm = confusion_matrix(self.observed, self.predito)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=self.observed.value_counts().index,
        )
        disp.plot(cmap=plt.cm.Greens)
        plt.title(f"{self.target}'s cluster: Accuracy {accuracy:.2f}")
        plt.show()
        print("\nClassification Report:")
        print(classification_report(self.observed, self.predito))
        return None

    def __metricas_pontuais(self) -> None:
        #  Acurácia
        acc = metrics.accuracy_score(self.observed, self.predito)

        # Precision
        precision = metrics.precision_score(self.observed, self.predito)

        # Recall
        recall = metrics.recall_score(self.observed, self.predito)

        # F1-score
        f1_score = metrics.f1_score(self.observed, self.predito)

        # KS
        index = self.df[self.df[self.target] == 1][self.indep]
        if self.troca_classe:
            score_pop1 = self.__retorna_classe(self.modelo.predict(index))
        else:
            score_pop1 = self.modelo.predict(index)

        index = self.df[self.df[self.target] == 0][self.indep]
        if self.troca_classe:
            score_pop2 = self.__retorna_classe(self.modelo.predict(index))
        else:
            score_pop2 = self.modelo.predict(index)

        ks = ks_2samp(score_pop1, score_pop2).statistic

        # AUC
        fpr, tpr, _ = metrics.roc_curve(self.observed, self.predito)
        auc = metrics.auc(fpr, tpr)

        # Gini
        gini = 2 * auc - 1

        print(
            pd.DataFrame(
                {
                    "Acurácia": [acc],
                    "Precision": [precision],
                    "Recall": [recall],
                    "F1-Score": [f1_score],
                    "KS": [ks],
                    "AUC": [auc],
                    "GINI": [gini],
                }
            ).round(4)
        )

        return None

    def __grafico_ks(self) -> None:
        # Plotagem da métrica KS (Kolmogorov-Smirnov)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        index = self.df[self.df[self.target] == 1][self.indep]
        score_pop1 = self.modelo.predict(index)
        ax.plot(
            np.sort(score_pop1),
            np.linspace(0, 1, len(score_pop1), endpoint=False),
            label="Cluster 1",
        )

        index = self.df[self.df[self.target] == 0][self.indep]
        score_pop2 = self.modelo.predict(index)
        ax.plot(
            np.sort(score_pop2),
            np.linspace(0, 1, len(score_pop2), endpoint=False),
            label="Cluster 0",
        )
        ax.legend()

        plt.title(f"Gráfico KS {self.target}")
        ax.set_xlabel("P")
        ax.set_ylabel("Função Distribuição Acumulada")
        plt.show()

        return None

    def __curva_roc(self) -> None:
        flag_serie = self.df.loc[:, self.target]
        # Plotando a curva ROC
        fpr, tpr, _ = metrics.roc_curve(flag_serie, self.predito)

        plt.figure()
        lw = 2

        fpr, tpr, _ = metrics.roc_curve(flag_serie, self.predito)
        auc_ = metrics.auc(fpr, tpr)
        plt.plot(
            fpr, tpr, color="darkorange", lw=lw, label=f"ROC curve (area = {auc_})"
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Receiver Operating Characteristic (ROC) for {self.target}")
        plt.legend(loc="lower right")
        plt.show()

        return None

    def metricas(self) -> None:
        """
        Calcula as métricas pontuais do modelo
        """
        self.__matrix_confusao()
        self.__metricas_pontuais()
        self.__curva_roc()
        self.__grafico_ks()
        return None
