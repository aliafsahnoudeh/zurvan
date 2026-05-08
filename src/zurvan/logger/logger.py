from enum import StrEnum


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class Logger:
    def log(self, message: str, level: LogLevel = LogLevel.INFO, env=None):
        pass
