import os
from typing import List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

from utils.logger import get_logger

# Configuração do logger
logger = get_logger(__name__)

# # Constantes de configuração
# DEFAULT_SIMILARITY_RESULTS = 10
# DEFAULT_TEMPERATURE = 0.5
# DEFAULT_MODEL = "gpt-5-nano"
# FALLBACK_RESPONSE = "Não tenho informações necessárias para responder sua pergunta."

class SearchError(Exception):
    """Exceção customizada para erros."""
    pass

def validate_environment_variables() -> None:
    """Valida se todas as variáveis de ambiente necessárias estão definidas."""

    required_keys = [
        "OPENAI_API_KEY",
        "OPENAI_EMBEDDING_MODEL",

        "PGVECTOR_URL",
        "PGVECTOR_COLLECTION",

        "MODEL_SIMILARITY",
        "MODEL_TEMPERATURE",
        "MODEL_NAME",

        "TEXT_SPLITTER_CHUNK_SIZE",
        "TEXT_SPLITTER_CHUNK_OVERLAP",
        "TEXT_SPLITTER_ADD_START_INDEX"
    ]

    missing_keys = [key for key in required_keys if not os.getenv(key)]
    if missing_keys:
        raise SearchError(
            f"Variáveis de ambiente não definidas: {', '.join(missing_keys)}"
        )

def get_float_env(key: str) -> float:
    """
    Converte uma string numérica para float.

    Args:
        key (str): Valor em string (ex: "0.5")

    Returns:
        float: Valor convertido

    Raises:
        ValueError: Se a string não puder ser convertida em float
    """

    try:
        return float(os.getenv(key).strip())
    except (ValueError, TypeError):
        raise SearchError(f"Variável {key} deve ser um número float.")

def get_int_env(key: str) -> int:
    """Converte variável de ambiente para inteiro com validação."""

    try:
        value = int(os.getenv(key))
        if value <= 0:
            raise ValueError
        return value
    except (ValueError, TypeError):
        raise SearchError(f"Variável {key} deve ser um número inteiro positivo.")

def get_prompt_template() -> PromptTemplate:
    """
    Retorna um PromptTemplate configurado para RAG com contexto.

    Returns:
        PromptTemplate: Template configurado para respostas baseadas em contexto
    """

    template = """
        CONTEXTO:
        {contexto}
        
        REGRAS:
        - Responda somente com base no CONTEXTO.
        - Se a informação não estiver explicitamente no CONTEXTO, responda: "Não tenho informações necessárias para responder sua pergunta."
        - Nunca invente ou use conhecimento externo.
        - Nunca produza opiniões ou interpretações além do que está escrito.
        
        EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
        Pergunta: "Qual é a capital da França?"
        Resposta: "Não tenho informações necessárias para responder sua pergunta."
        
        Pergunta: "Quantos clientes temos em 2024?"
        Resposta: "Não tenho informações necessárias para responder sua pergunta."
        
        Pergunta: "Você acha isso bom ou ruim?"
        Resposta: "Não tenho informações necessárias para responder sua pergunta."
        
        PERGUNTA DO USUÁRIO:
        {pergunta}
        
        RESPONDA A "PERGUNTA DO USUÁRIO"
    """

    return PromptTemplate(
        input_variables=["contexto", "pergunta"],
        template=template
    )

def create_llm_model(model_name: str, temperature: float) -> ChatOpenAI:
    """
    Cria e configura o modelo de linguagem.

    Args:
        temperature: Temperatura
        model_name: Nome do modelo OpenAI

    Returns:
        ChatOpenAI: Modelo configurado
    """

    return ChatOpenAI(
        model=model_name,
        temperature=temperature
    )

def create_embeddings(embedding: str) -> OpenAIEmbeddings:
    """
    Cria e configura o modelo de embeddings.

    Returns:
        OpenAIEmbeddings: Modelo de embeddings configurado
    """

    return OpenAIEmbeddings(
        model=embedding
    )

def create_vector_store(embeddings: OpenAIEmbeddings, collection: str, url: str) -> PGVector:
    """
    Cria e configura o armazenamento vetorial.

    Args:
        embeddings: Modelo de embeddings configurado
        collection:  PGVector collection
        url: PGVector URL

    Returns:
        PGVector: Store vetorial configurado
    """

    return PGVector(
        embeddings=embeddings,
        collection_name=collection,
        connection=url,
        use_jsonb=True,
    )

def search_similar_documents(store: PGVector, question: str, similarity: int) -> List[Document]:
    """
    Busca documentos similares à pergunta.

    Args:
        store: Store vetorial configurado
        question: Pergunta do usuário
        similarity: Número de documentos a retornar

    Returns:
        List[Document]: Lista de documentos similares
    """

    return store.similarity_search(
        question,
        k=similarity
    )

def format_context(documents: List[Document]) -> str:
    """
    Formata os documentos encontrados em contexto para o prompt.

    Args:
        documents: Lista de documentos encontrados

    Returns:
        str: Contexto formatado
    """

    if not documents:
        return "Nenhuma informação relevante encontrada."

    context_parts = []

    for i, doc in enumerate(documents, 1):
        content = doc.page_content.strip()

        # Só adiciona se não estiver vazio
        if content:
            context_parts.append(f"{i}. {content}")

    return "\n\n".join(context_parts)

def search_prompt(question=None):
    load_dotenv()
    validate_environment_variables()

    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL")
    pg_collection = os.getenv("PGVECTOR_COLLECTION")
    pg_url = os.getenv("PGVECTOR_URL")
    model_similarity = get_int_env("MODEL_SIMILARITY")
    model_temperature = get_float_env("MODEL_TEMPERATURE")
    model_name = os.getenv("MODEL_NAME")

    prompt_template = get_prompt_template()
    model = create_llm_model(model_name, model_temperature)
    embeddings = create_embeddings(embedding_model)
    store = create_vector_store(embeddings, pg_collection, pg_url)

    chain = prompt_template | model

    documents = search_similar_documents(store, question, model_similarity)
    contexto = format_context(documents)

    response = chain.invoke({
        "contexto": contexto,
        "pergunta": question
    })

    print("\n")
    logger.info(f"Pergunta: {question}")
    logger.info(f"Resposta: {response.content.strip()}")
    logger.info("-" * 50)
    print("\n")

    pass