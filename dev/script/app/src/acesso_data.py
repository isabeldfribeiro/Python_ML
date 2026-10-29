import os


def acesso_data_test() -> str:
    """
    Função para acessar o ícone da aplicação.
    """
    path_img_local = "./app/data/img/DataLab_Logo_i.jpg"
    path_img = "./dev/script/app/data/img/DataLab_Logo_i.jpg"

    if os.path.isfile(path_img_local):
        path_img = path_img_local

    return path_img


def acesso_data_icon() -> str:
    """
    Função para acessar o ícone da aplicação.
    """
    path_img_local = "./app/data/img/PythonML.ico"
    path_img = "./dev/script/app/data/img/PythonML.ico"

    if os.path.isfile(path_img_local):
        path_img = path_img_local

    return path_img


def acesso_data_banco() -> str:
    """
    Função para acessar o banco de dados de treino.
    """
    path_banco_local = "./app/data/"
    path_banco = "./dev/script/app/data/"

    if os.path.isdir(path_banco_local):
        path_banco = path_banco_local

    return path_banco
