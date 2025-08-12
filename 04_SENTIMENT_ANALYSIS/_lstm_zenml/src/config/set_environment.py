import os


def set_ENVIRONMENT() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
