import os

from src.paginas import Paginas
from src.acesso_data import acesso_data_test, acesso_data_banco


def test_acesso_data_img():
    # Given
    teste_local = "./app/data/img/DataLab_Logo_i.jpg"

    # When
    teste_saida = acesso_data_test()
    if os.path.isfile(teste_saida):
        teste_local = teste_saida

    # Then
    assert teste_local == teste_saida


def test_acesso_data_banco():
    # Given
    teste_local = "./app/data/"

    # When
    teste_saida = acesso_data_banco()
    if os.path.isdir(teste_saida):
        teste_local = teste_saida

    # Then
    assert teste_local == teste_saida


def test_busca_dados_cluster():
    # Given
    teste_entrada = 3

    # When
    iniciando = Paginas()
    df_train, df_valid, df_test = iniciando.busca_dados_cluster()
    teste_saida = len([df_train, df_valid, df_test])

    # Then
    assert teste_entrada == teste_saida


def test_busca_dados_class():
    # Given
    teste_entrada_1 = 6
    teste_entrada_2 = 100

    # When
    iniciando = Paginas()
    teste_saida_1 = len(iniciando.busca_dados_class()[0])
    teste_saida_2 = len(iniciando.busca_dados_class()[1])

    # Then
    assert teste_entrada_1 == teste_saida_1
    assert teste_entrada_2 == teste_saida_2


def test_busca_dados_regr():
    # Given
    teste_entrada_1 = 700
    teste_entrada_2 = 100

    # When
    iniciando = Paginas()
    teste_saida_1 = len(iniciando.busca_dados_regr()[0])
    teste_saida_2 = len(iniciando.busca_dados_regr()[1])

    # Then
    assert teste_entrada_1 == teste_saida_1
    assert teste_entrada_2 == teste_saida_2
