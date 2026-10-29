from unittest.mock import patch

from src.acesso_data import acesso_data_banco, acesso_data_icon, acesso_data_test


def test_acesso_data_test_usa_caminho_local():
    with patch("src.acesso_data.os.path.isfile", return_value=True):
        assert acesso_data_test() == "./app/data/img/DataLab_Logo_i.jpg"


def test_acesso_data_icon_usa_caminho_local():
    with patch("src.acesso_data.os.path.isfile", return_value=True):
        assert acesso_data_icon() == "./app/data/img/PythonML.ico"


def test_acesso_data_banco_usa_caminho_local():
    with patch("src.acesso_data.os.path.isdir", return_value=True):
        assert acesso_data_banco() == "./app/data/"
