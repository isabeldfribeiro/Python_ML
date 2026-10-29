from unittest.mock import patch

import pandas as pd
import pytest

from src.modelos.classifier import Classifier


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Dataset": ["treino", "treino", "teste", "teste"],
            "Classe_Predita": ["a", "b", "a", "b"],
        }
    )


@pytest.mark.parametrize(
    "p_valor, esperado",
    [
        (0.5, "Falhamos em rejeitar H0"),
        (0.01, "Rejeitamos H0"),
    ],
)
def test_testar_hipotese_ramos(p_valor, esperado):
    # Given
    obj = Classifier(_df())

    # When
    with patch("src.modelos.classifier.st") as st_mock, patch(
        "src.modelos.classifier.chi2_contingency",
        return_value=(1.0, p_valor, 1, None),
    ):
        resultado = obj.testar_hipotese()

    # Then
    assert resultado is None
    mensagem = st_mock.markdown.call_args[0][0]
    assert mensagem.startswith(esperado)
