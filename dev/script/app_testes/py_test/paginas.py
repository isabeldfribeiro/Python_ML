from pydantic import validate_call
from loguru import logger


@validate_call
def pagina_inicial(mensagem: str) -> str:
    """
    Função de inicialização da página inicial.
    Parametros:
        mensagem (str): Mensagem a ser exibida.
    Retorna:
        str: Mensagem recebida.
    """
    # Log indicando que a função processou o dado com sucesso
    logger.info(f"Página inicial carregada com a mensagem: {mensagem}")
    return mensagem
