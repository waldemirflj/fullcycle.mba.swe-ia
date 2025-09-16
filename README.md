## MBA Engenharia de Software com IA - Full Cycle

### Desafio - Ingestão e Busca Semântica com LangChain e PostgreSQL

Este projeto implementa ingestão e busca semântica usando LangChain + PostgreSQL (pgVector) para responder perguntas com base no conteúdo de um PDF.

---

#### Na raiz do projeto, execute os comandos abaixo:  

```bash
# Sobe os serviços definidos no docker-compose (ex: PostgreSQL com pgVector)
docker compose up -d

# Cria o ambiente virtual (.venv) isolado para instalar as dependências do projeto
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Cria o arquivo .env a partir do modelo .env.example
# (edite esse arquivo para configurar variáveis como chaves da OpenAI e conexão ao PostgreSQL)
cp .env.example .env

# Executa o processo de ingestão:
# - lê o PDF
# - gera embeddings com LangChain + OpenAI
# - armazena os vetores no PostgreSQL (pgVector)
python src/ingest.py

# Inicia o ChatBot:
# - recebe perguntas do usuário
# - retorna respostas contextuais baseadas no conteúdo do PDF
python src/chat.py
```

#### Estrutura do Projeto
```bash
├── docker-compose.yml  # Banco de dados Postgres + pgVector
├── document.pdf        # PDF para ingestão e busca
├── requirements.txt    # Dependências do projeto
└── src
    ├── chat.py         # CLI
    ├── ingest.py       # Processo de ingestão
    ├── search.py       # Processo de busca e resposta
    └── utils
        ├── logger.py   # utilitário de logs
```

#### Tecnologias
- Pip - v23.2.1
- Pyenv - v2.6.6
- Python - v3.12.0
- Docker - v28.4.0
