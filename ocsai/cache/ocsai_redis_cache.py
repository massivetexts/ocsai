import redis
import pandas as pd
import json
from .ocsai_cache import Ocsai_Cache
from ..utils import set_cache_dtypes


class Ocsai_Redis_Cache(Ocsai_Cache):
    def __init__(self, redis_url: str, logger=None):
        super().__init__(logger=logger)
        self.redis = redis.Redis.from_url(redis_url)
        self.base_cols = ["prompt", "response", "question", "type", "language", "model"]

    def _check_input_format(self, df: pd.DataFrame):
        for col in self.base_cols:
            if col not in df.columns:
                raise ValueError(f"Column {col} not found in input dataframe")

    def get_cache_scores(self, df: pd.DataFrame):
        self._check_input_format(df)
        df = set_cache_dtypes(df)

        pipeline = self.redis.pipeline()
        keys = []
        for _, row in df.iterrows():
            key = self._generate_cache_key(row)
            pipeline.get(key)
            keys.append(key)

        cached_results = pipeline.execute()

        cached_scores = []
        to_score = []

        for i, cached in enumerate(cached_results):
            if cached:
                cached_scores.append(json.loads(cached))
            else:
                to_score.append(df.iloc[i])

        to_score_df = pd.DataFrame(to_score, columns=self.base_cols)
        cache_results_df = pd.DataFrame(cached_scores,
                                        columns=self.base_cols + ["score", "confidence", "flags", "timestamp"])

        return to_score_df, cache_results_df

    def write(self, df: pd.DataFrame, min_size_to_write: int = 0):
        if df.empty:
            return

        pipeline = self.redis.pipeline()
        for _, row in df.iterrows():
            key = self._generate_cache_key(row)
            value = row.to_json()
            pipeline.set(key, value)

        pipeline.execute()

    def _generate_cache_key(self, row):
        return ':'.join([str(row[col]) for col in sorted(self.base_cols)])

    def __del__(self):
        pass  # Redis handles persistence automatically
