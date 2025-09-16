import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector

from utils.logger import get_logger

# Configuração do logger
logger = get_logger(__name__)

class IngestionError(Exception):
    """Exceção customizada para erros."""
    pass

def validate_environment_variables() -> None:
    """Valida se todas as variáveis de ambiente necessárias estão definidas."""

    required_keys = [
        "OPENAI_API_KEY",
        "OPENAI_EMBEDDING_MODEL",
        "PGVECTOR_URL",
        "PGVECTOR_COLLECTION",
        "PDF_PATH",
        "TEXT_SPLITTER_CHUNK_SIZE",
        "TEXT_SPLITTER_CHUNK_OVERLAP",
        "TEXT_SPLITTER_ADD_START_INDEX"
    ]

    missing_keys = [key for key in required_keys if not os.getenv(key)]
    if missing_keys:
        raise IngestionError(
            f"Variáveis de ambiente não definidas: {', '.join(missing_keys)}"
        )

def get_int_env(key: str) -> int:
    """Converte variável de ambiente para inteiro com validação."""

    try:
        value = int(os.getenv(key))
        if value <= 0:
            raise ValueError
        return value
    except (ValueError, TypeError):
        raise IngestionError(f"Variável {key} deve ser um número inteiro positivo.")

def get_bool_env(key: str) -> bool:
    """Converte variável de ambiente para bool."""

    value = os.getenv(key, "").lower()
    return value in ("true", "1", "yes", "on")

def load_pdf_documents(pdf_path: str) -> List[Document]:
    """Carrega um PDF e retorna lista de documentos."""

    file = Path(pdf_path)

    if not file.exists():
        raise IngestionError(f"Arquivo não encontrado: {pdf_path}")
    if file.suffix.lower() != ".pdf":
        raise IngestionError(f"O arquivo não é um PDF válido: {pdf_path}")

    logger.info(f"Carregando PDF: {pdf_path}")
    docs = PyPDFLoader(str(file)).load()

    if not docs:
        raise IngestionError("O PDF está vazio ou não pôde ser processado.")

    logger.info(f"PDF carregado com sucesso - {len(docs)} páginas.")
    return docs

def split_documents(docs: List[Document], chunk_size: int, chunk_overlap: int, add_start_index: bool) -> List[Document]:
    """Divide documentos em chunks menores."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=add_start_index,
    )

    splits = splitter.split_documents(docs)

    if not splits:
        raise IngestionError("Nenhum chunk foi gerado.")

    logger.info(f"Documentos divididos em {len(splits)} chunks.")
    return splits

def clean_document_metadata(splits: List[Document]) -> List[Document]:
    """Remove metadados vazios dos documentos."""

    logger.info("Limpando metadados vazios...")
    enriched = []

    for documento in splits:
        # Filtra apenas metadados que têm valor (não são vazios ou None)
        metadados_limpos = {}
        for chave, valor in documento.metadata.items():
            if valor != "" and valor is not None:
                metadados_limpos[chave] = valor

        # Cria um novo documento com os metadados filtrados
        enriched.append(Document(
            page_content=documento.page_content,
            metadata=metadados_limpos
        ))

    return enriched

def generate_document_ids(documents: List[Document]) -> List[str]:
    """Gera IDs sequenciais para documentos."""

    return [f"doc-{i}" for i in range(len(documents))]

def create_embeddings(model: str) -> OpenAIEmbeddings:
    """Inicializa embeddings."""

    logger.info(f"Inicializando embeddings com modelo: {model}")
    return OpenAIEmbeddings(model=model)

def create_vector_store(embeddings: OpenAIEmbeddings, collection: str, url: str) -> PGVector:
    """Cria banco vetorial PGVector."""

    logger.info("Conectando ao PGVector...")
    return PGVector(
        embeddings=embeddings,
        collection_name=collection,
        connection=url,
        use_jsonb=True,
    )

def ingest_pdf() -> None:
    """Fluxo principal de ingestão de PDF -> embeddings -> PGVector."""

    load_dotenv()
    validate_environment_variables()

    pdf_path = os.getenv("PDF_PATH")
    pg_url = os.getenv("PGVECTOR_URL")
    pg_collection = os.getenv("PGVECTOR_COLLECTION")
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL")

    chunk_size = get_int_env("TEXT_SPLITTER_CHUNK_SIZE")
    chunk_overlap = get_int_env("TEXT_SPLITTER_CHUNK_OVERLAP")
    add_start_index = get_bool_env("TEXT_SPLITTER_ADD_START_INDEX")

    docs = load_pdf_documents(pdf_path)
    splits = split_documents(docs, chunk_size, chunk_overlap, add_start_index)
    enriched = clean_document_metadata(splits)
    ids = generate_document_ids(enriched)

    embeddings = create_embeddings(embedding_model)
    store = create_vector_store(embeddings, pg_collection, pg_url)
    store.add_documents(documents=enriched, ids=ids)

    logger.info("Ingestão de PDF concluída com sucesso!")


if __name__ == "__main__":
    try:
        ingest_pdf()
    except IngestionError as e:
        logger.error(f"Erro controlado: {e}")
        exit(1)
    except KeyboardInterrupt:
        print("\n")
        logger.warning("-" * 50)
        logger.warning("Execução interrompida pelo usuário.")
        exit(0)
    except Exception as e:
        print("\n")
        logger.critical("-" * 50)
        logger.critical(f"Erro inesperado: {e}")
        exit(2)