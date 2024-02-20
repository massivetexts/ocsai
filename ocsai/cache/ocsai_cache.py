from pathlib import Path
import logging


class Ocsai_Cache:
    base_cols = ["prompt", "response", "question", "type", "language", "model"]

    def __init__(self, cache_path=None, logger=None):
        self.cache_path = None
        if cache_path:
            self.cache_path = Path(cache_path)
            self.cache_path.mkdir(parents=True, exist_ok=True)

        self.logger = logger
        if not self.logger:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)

    def is_empty(self):
        raise NotImplementedError
