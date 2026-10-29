from src.paginas import pagina_inicial
from loguru import logger


def test_pagina_inicial():
    logger.debug("Iniciando teste: test_pagina_inicial")
    # Given
    teste_entrada = "Página inicial carregada com sucesso."

    # When
    teste_saida = pagina_inicial(mensagem=teste_entrada)

    # Then
    assert teste_entrada == teste_saida
    logger.success("Teste test_pagina_inicial finalizado com sucesso.")


def test_pagina_inicial_erro():
    logger.debug("Iniciando teste: test_pagina_inicial_erro")
    # Given
    teste_entrada = 10

    # When
    try:
        teste_saida = pagina_inicial(mensagem=teste_entrada)
    except Exception as e:
        logger.warning(f"Exceção esperada no teste capturada: {e}")
        teste_saida = str(e)

    # Then
    assert teste_entrada != teste_saida
    logger.success("Teste test_pagina_inicial_erro finalizado com sucesso.")
