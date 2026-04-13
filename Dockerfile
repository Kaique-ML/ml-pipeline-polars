# 1. Imagem base (Python 3.11 slim para ser leve)
FROM python:3.11.3-slim

# 2. Define o diretório de trabalho dentro do container
WORKDIR /app

# 3. Copia apenas o arquivo de requisitos primeiro (otimiza o cache do Docker)
COPY requirements.txt .

# 4. Instala as dependências (sem usar o venv local, o Docker já é um ambiente isolado)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copia todo o resto do projeto para dentro do container
COPY . .

# 6. Informa que a API usará a porta 8000
EXPOSE 8000

# O comando final não é necessário aqui porque você já o definiu no docker-compose.yml
# Mas, se quiser deixar por padrão:
CMD ["uvicorn", "serving.main:app", "--host", "0.0.0.0", "--port", "8000"]