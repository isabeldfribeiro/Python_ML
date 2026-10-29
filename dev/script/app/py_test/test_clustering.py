from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.modelos.clustering import Clustering

plt.switch_backend("Agg")

MODELOS = {
    "kmeans": ("_Clustering__testar_kmeans", {"n_clusters": 2, "random_state": 0}),
    "dbscan": ("_Clustering__testar_dbscan", {"eps": 1.5, "min_samples": 3}),
    "agg": ("_Clustering__testar_agg", {"n_clusters": 2}),
}


def _dados(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.poisson(1, size=(n, 2))
    b = rng.poisson(6, size=(n, 2))
    df = pd.DataFrame(np.vstack([a, b]), columns=["c1", "c2"])
    df["grupo"] = ["A"] * n + ["B"] * n
    return df


@pytest.mark.parametrize("modelo", list(MODELOS))
@pytest.mark.parametrize("p_value", [0.5, 0.01])
@pytest.mark.parametrize("ari", [0.9, 0.1])
def test_testar_modelo_ramos(modelo, p_value, ari):
    # Given
    metodo, params = MODELOS[modelo]
    obj = Clustering("c1", "c2", _dados(seed=0), modelo)
    df_test = _dados(seed=1)

    # When
    with (
        patch("src.modelos.clustering.chisquare", return_value=(1.0, p_value)),
        patch("src.modelos.clustering.adjusted_rand_score", return_value=ari),
    ):
        grafico, msg = getattr(obj, metodo)(params, df_test)
    plt.close("all")

    # Then
    assert grafico is not None
    assert ("É IGUAL" in msg) == (p_value > 0.05)
    assert ("É DIFERENTE" in msg) == (p_value <= 0.05)
    assert ("SÃO SEMELHANTES" in msg.replace("NÃO SÃO SEMELHANTES", "")) == (ari > 0.5)
    assert ("NÃO SÃO SEMELHANTES" in msg) == (ari <= 0.5)
