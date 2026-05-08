import os
from datetime import datetime

from zurvan.logger.logger import LogLevel, Logger

_LEVEL_RANK = {
    LogLevel.DEBUG: 0,
    LogLevel.INFO: 1,
    LogLevel.WARNING: 2,
    LogLevel.ERROR: 3,
}


class LocalTextFileLogger(Logger):
    def __init__(self, name: str, level: LogLevel = LogLevel.INFO, path: str = None):
        self._level = level
        logs_dir = os.path.join(path or os.getcwd(), ".logs")
        os.makedirs(logs_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.txt"
        self._filepath = os.path.join(logs_dir, filename)

    def log(self, message: str, level: LogLevel = LogLevel.INFO, env=None):
        if _LEVEL_RANK[level] < _LEVEL_RANK[self._level]:
            return
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        env_info = f" [ENV: {env.__class__.__name__}]" if env else ""
        line = f"{timestamp} [{level}] {message}{env_info}\n"
        with open(self._filepath, "a", encoding="utf-8") as f:
            f.write(line)
