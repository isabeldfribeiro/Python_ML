import pandas as pd


class Line_equation:
    """
    Classe que representa uma equação de reta, a partir de um DataFrame
    fornecido, contendo duas colunas, uma para o eixo x e outra para o
    eixo y.
    """

    def __init__(self, dados: pd.DataFrame, coluna_x: str, coluna_y: str) -> None:
        self.dados = dados
        self.coluna_x = coluna_x
        self.coluna_y = coluna_y

    def __colunas_base(self) -> tuple:
        """
        Função que retorna os pontos mínimo e máximo das colunas x e y
        do DataFrame fornecido.
            Retorna:
                tuple: pontos mínimo e máximo das colunas x e y.
        """
        x_base = self.dados.loc[:, self.coluna_x]
        y_base = self.dados.loc[:, self.coluna_y]
        a = (x_base.min(), y_base.min())
        b = (x_base.max(), y_base.max())

        return a, b

    def __coeficiente_angular(self) -> tuple:
        """
        m = ya - yb
            ---------
            xa - xb

        m = delta y / delta x

        Função que calcula o coeficiente angular da reta formada pelos
        pontos a e b, que são os pontos mínimo e máximo das colunas x e y
        do DataFrame fornecido.
            Retorna:
                tuple: coeficiente angular da reta (delta_y, delta_x).
        """
        a = self.__colunas_base()[0]
        b = self.__colunas_base()[1]

        delta_x = a[0] - b[0]
        delta_y = a[1] - b[1]

        if delta_x < 0:
            delta_x = delta_x * -1

        if delta_y < 0:
            delta_y = delta_y * -1

        return (delta_y, delta_x)

    def equacao_da_reta_calculo(
        self, varia_coeficiente: int | None = None, new_y: float | None = None
    ) -> float:
        """
        y - y0 = m(x - x0)

        Função que calcula o valor de y para um dado x, usando a equação da
        reta, a partir dos pontos mínimo e máximo das colunas x e y do
        DataFrame fornecido.
        y é calculado a partir do ponto mínimo da reta, e o coeficiente
        angular é calculado a partir dos pontos mínimo e máximo das
        colunas x e y.
        y = (((m + m_perc) * x) + c)/m[1]
            Parametros:
                varia_coeficiente: valor que será somado ao coeficiente
                angular da reta, para alterar a inclinação da reta.
                new_y: valor de y que será usado no cálculo, caso seja
                fornecido. Caso contrário, o valor de y do ponto mínimo
                da reta será usado.
            Retorna:
                y: valor de y correspondente ao valor de x fornecido.
        """
        a = self.__colunas_base()[0]
        b = self.__colunas_base()[1]

        x = b[0]
        if new_y is not None:
            x = new_y
        ponto = a

        m = self.__coeficiente_angular()
        x0 = m[0] * ponto[0]
        y0 = m[1] * ponto[1]
        c = -x0 + y0

        m_perc = 0
        if varia_coeficiente is not None:
            m_perc = (m[0] / 10) * varia_coeficiente

        y = (((m[0] + m_perc) * x) + c) / m[1]

        return y

    def reta(
        self, x: float, intercept: int = 5, slope: int = 5, bias: int = 0
    ) -> float:
        """
        Função que calcula o valor de y para um dado x, usando a equação
        da reta: y = intercept + slope * x + bias.
            Parametros:
                x: valor de x para o qual queremos calcular y.
                intercept: valor do intercepto da reta (ponto onde a reta
                cruza o eixo y).
                slope: valor do coeficiente angular da reta (inclinação
                da reta).
                bias: valor do viés da reta (deslocamento vertical).
            Retorna:
                y: valor de y correspondente ao valor de x fornecido.
        """
        y = intercept + x * slope + bias

        return y
