from dotenv import load_dotenv

from search import search_prompt
from utils.logger import get_logger

# Carrega variáveis de ambiente
load_dotenv()

# Configuração do logger
logger = get_logger(__name__)

class ChatError(Exception):
    """Exceção customizada para erros."""
    pass

def new_question() -> None:
    """Captura a pergunta do usuário."""

    question = input("Digite sua pergunta: ").strip()

    if not question:
        raise ChatError("A pergunta não pode ser vazia.")

    search_prompt(question)

def main() -> None:
    """Fluxo principal do ChatBot."""

    logger.info("ChatBot Iniciado: Desafio do MBA em Engenharia de Software com IA - Full Cycle")
    logger.info("-" * 50)
    print("\n")

    while True:
        print("1. Faça uma pergunta.")
        print("2. Sair.")
        print("\n")

        try:
            option = int(input("Escolha uma das opções acima: ").strip())
        except ValueError:
            print("❌ - Entrada inválida.", "\n")
            continue

        if option == 1:
            try:
                new_question()
            except ChatError as e:
                print(f"⚠️  - {e}", "\n")

        elif option == 2:
            print("Saindo.")
            break
        else:
            print("❌ - Opção inválida, tente novamente.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n")
        logger.warning("Execução interrompida pelo usuário.")
        logger.warning("-" * 50)
        exit(0)
    except Exception as e:
        print("\n")
        logger.critical(f"Erro inesperado: {e}")
        logger.critical("-" * 50)
        exit(2)