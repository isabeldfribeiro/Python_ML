# Dokerfile para uso de kernel do Jupyter Lab com Python 3.10 no VSCode
FROM python:3.10

# Devenvolvedor do projeto
LABEL maintainer="Guilherme F Silveira"

# Definir o diretório de trabalho dentro do container
WORKDIR /usr/src/myapp

# Copiar o arquivo de requisitos para o diretório de trabalho
COPY ./requirements.txt .

# Instalar as dependências do kernel no Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copiar todos os arquivos do projeto para o diretório de trabalho
COPY . .

# Expor a porta 8888 para acessar o kernel
EXPOSE 8888
EXPOSE 8080

# Comando para iniciar o Jupyter Lab quando o container for executado
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]

# Definindo o comando de entrada para iniciar a aplicação Streamlit
# ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8080", "--server.address=0.0.0.0"]
