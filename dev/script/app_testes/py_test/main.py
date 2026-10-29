import sys
from src.paginas import pagina_inicial
from pydantic import ValidationError
from loguru import logger


if __name__ == "__main__":
    # Remove a configuração padrão do terminal (opcional,
    # para não duplicar se for mudar o formato)
    logger.remove()

    # Define um formato personalizado destacando a data (Ex: 14/08/2026 15:47)
    formato_personalizado = "{time:DD/MM/YYYY HH:mm:ss} | {level} | {message}"

    # Configura o log para o terminal com a data formatada
    logger.add(sys.stderr, format=formato_personalizado)

    # Configura o log para o arquivo com a data formatada
    logger.add("./app_testes/log.log", rotation="1 MB", format=formato_personalizado)

    logger.debug("Iniciando a execução do script...")

    try:
        resultado_erro = pagina_inicial(mensagem=10)
    except ValidationError as e:
        logger.error(f"Falha de validação capturada: {e}")

    try:
        resultado_sucesso = pagina_inicial(
            mensagem="Página inicial carregada com sucesso."
        )
        logger.success("Função executada perfeitamente.")
        print(resultado_sucesso)
    except ValidationError as e:
        logger.error(f"Falha de validação capturada: {e}")
