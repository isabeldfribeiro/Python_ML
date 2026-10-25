from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent


def acesso_data_test() -> str:
    """
    Função para acessar a imagem (logo) da aplicação.
    """
    return str(RAIZ / "data" / "img" / "bioinfofos.jpg")


def acesso_data_icon() -> str:
    """
    Função para acessar o ícone da aplicação.
    """
    return str(RAIZ / "data" / "img" / "PythonML.ico")


def acesso_data_banco() -> str:
    """
    Função para acessar o banco de dados de treino.
    """
    return str(RAIZ / "data") + "/"
