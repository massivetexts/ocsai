import redis
import pandas as pd
import json
import logging
from .ocsai_cache import Ocsai_Cache
from ..utils import set_cache_dtypes


class Ocsai_Redis_Cache(Ocsai_Cache):
    """Redis-backed cache for OCSAI scores.

    Uses Redis for storage with automatic LRU eviction when memory limits are set.
    Configure Redis with `maxmemory` and `maxmemory-policy allkeys-lru` for
    automatic cleanup of old cache entries.
    """

    KEY_PREFIX = "ocsai_cache:"

    def __init__(self, redis_url: str, logger=None):
        # Don't pass cache_path to parent since Redis doesn't use file paths
        self.cache_path = None
        self.logger = logger
        if not self.logger:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)

        self.redis = redis.Redis.from_url(redis_url, decode_responses=True)
        # Match base class base_cols (includes "method")
        self.base_cols = ["prompt", "response", "question", "type", "language", "model", "method"]

    def _check_input_format(self, df: pd.DataFrame):
        for col in self.base_cols:
            if col not in df.columns:
                raise ValueError(f"Column {col} not found in input dataframe")

    def is_empty(self) -> bool:
        """Check if the cache has any entries."""
        try:
            # Use SCAN with higher count to reliably find at least one key if any exist
            # SCAN's count is a hint, so we need a reasonable value
            cursor, keys = self.redis.scan(cursor=0, match=f"{self.KEY_PREFIX}*", count=100)
            if keys:
                return False
            # If first scan returned nothing, continue scanning until cursor returns to 0
            while cursor != 0:
                cursor, keys = self.redis.scan(cursor=cursor, match=f"{self.KEY_PREFIX}*", count=100)
                if keys:
                    return False
            return True
        except redis.RedisError as e:
            self.logger.warning(f"Redis error checking if empty: {e}")
            return True

    def get_cache_scores(self, df: pd.DataFrame):
        """For a dataframe, retrieve cached scores.

        Returns (to_score, cached) dataframes, matching the parquet cache interface.
        """
        self._check_input_format(df)
        df = set_cache_dtypes(df)

        if df.empty:
            empty_df = pd.DataFrame([], columns=self.base_cols)
            return empty_df, empty_df

        try:
            # Generate all keys and fetch in a single pipeline
            pipeline = self.redis.pipeline()
            for row in df.itertuples(index=False):
                key = self._generate_cache_key_from_tuple(row)
                pipeline.get(key)

            cached_results = pipeline.execute()
        except redis.RedisError as e:
            self.logger.warning(f"Redis error fetching cache, treating all as uncached: {e}")
            # On Redis failure, return all rows as needing scoring
            empty_cached = pd.DataFrame([], columns=self.base_cols + ["score", "confidence", "flags", "timestamp"])
            return df.copy(), set_cache_dtypes(empty_cached)

        to_score_indices = []
        cached_scores = []

        for i, cached in enumerate(cached_results):
            if cached:
                try:
                    cached_scores.append(json.loads(cached))
                except json.JSONDecodeError:
                    self.logger.warning(f"Invalid JSON in cache, will re-score")
                    to_score_indices.append(i)
            else:
                to_score_indices.append(i)

        # Preserve all columns from original DataFrame for rows needing scoring
        to_score_df = df.iloc[to_score_indices].copy() if to_score_indices else pd.DataFrame([], columns=df.columns)

        # Build cached results DataFrame from the list of dicts
        if cached_scores:
            cache_results_df = pd.DataFrame(cached_scores)
            cache_results_df = set_cache_dtypes(cache_results_df)
        else:
            cache_results_df = pd.DataFrame([], columns=self.base_cols + ["score", "confidence", "flags", "timestamp"])
            cache_results_df = set_cache_dtypes(cache_results_df)

        self.logger.debug(f"Cache hit: {len(cached_scores)}/{len(df)}, to score: {len(to_score_indices)}")

        return to_score_df, cache_results_df

    def write(self, df: pd.DataFrame, min_size_to_write: int = 0):
        """Write scores to Redis cache.

        Note: min_size_to_write is accepted for interface compatibility but
        Redis writes are always immediate (no in-memory buffering needed).
        """
        if df.empty:
            return

        try:
            pipeline = self.redis.pipeline()
            for row in df.itertuples(index=False):
                key = self._generate_cache_key_from_tuple(row)
                # Convert row to dict for JSON serialization
                row_dict = {col: getattr(row, col) for col in df.columns}
                # Handle special types that don't serialize well
                for k, v in row_dict.items():
                    if isinstance(v, (list, tuple)):
                        # Lists/tuples are fine for JSON
                        pass
                    elif hasattr(v, 'tolist'):  # numpy arrays
                        row_dict[k] = v.tolist()
                    elif pd.isna(v):
                        row_dict[k] = None
                value = json.dumps(row_dict)
                pipeline.set(key, value)

            pipeline.execute()
            self.logger.debug(f"Wrote {len(df)} entries to Redis cache")
        except redis.RedisError as e:
            self.logger.error(f"Redis error writing to cache: {e}")
            # Don't raise - cache write failure shouldn't break scoring

    def _generate_cache_key(self, row) -> str:
        """Generate a cache key from a row (dict-like or Series)."""
        key_parts = [str(row[col]) for col in sorted(self.base_cols)]
        return f"{self.KEY_PREFIX}{':'.join(key_parts)}"

    def _generate_cache_key_from_tuple(self, row) -> str:
        """Generate a cache key from a named tuple (from itertuples)."""
        key_parts = [str(getattr(row, col)) for col in sorted(self.base_cols)]
        return f"{self.KEY_PREFIX}{':'.join(key_parts)}"

    def clear(self):
        """Clear all cache entries. Use with caution."""
        try:
            cursor = 0
            deleted = 0
            while True:
                cursor, keys = self.redis.scan(cursor=cursor, match=f"{self.KEY_PREFIX}*", count=1000)
                if keys:
                    self.redis.delete(*keys)
                    deleted += len(keys)
                if cursor == 0:
                    break
            self.logger.info(f"Cleared {deleted} cache entries")
            return deleted
        except redis.RedisError as e:
            self.logger.error(f"Redis error clearing cache: {e}")
            raise

    def __del__(self):
        pass  # Redis handles persistence automatically
