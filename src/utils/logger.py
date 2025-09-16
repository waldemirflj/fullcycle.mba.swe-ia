import logging
import colorlog

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    # Cria o logger
    logger = logging.getLogger(name)

    # Se já foi configurado, retorna o existente
    if logger.handlers:
        return logger

    # Configura o handler com colorlog
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        }
    ))

    # Configura o logger
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False

    return logger

def setup_global_logger(level: int = logging.INFO) -> None:
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        }
    ))

    root_logger.addHandler(handler)
    root_logger.setLevel(level)

def quick_setup(level: int = logging.INFO) -> logging.Logger:
    return get_logger("main", level)

if __name__ == "__main__":
    logger = get_logger(__name__)

    # Exemplo de uso
    # logger.debug("Mensagem de DEBUG (ciano)")
    # logger.info("Mensagem de INFO (verde)")
    # logger.warning("Mensagem de WARNING (amarelo)")
    # logger.error("Mensagem de ERROR (vermelho)")
    # logger.critical("Mensagem de CRITICAL (vermelho negrito)")