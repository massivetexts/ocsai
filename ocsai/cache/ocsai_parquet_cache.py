from .ocsai_cache import Ocsai_Cache
import duckdb
import pandas as pd
import time
from pathlib import Path
from typing import Union


class Ocsai_Parquet_Cache(Ocsai_Cache):
    """The parquet cache is very basic: it writes each
    set of results to a different parquet file. Retrieval
    used duckdb joins.

    It has an in-memory fallback to avoid writing files for
    really small results.
    """

    def __init__(self, cache_path: Union[str, Path], logger=None):
        super().__init__(cache_path, logger)
        self.cache_path = cache_path
        self.in_memory_cache = pd.DataFrame(
            [], columns=self.base_cols + ["score", "timestamp"]
        )

    def is_empty(self):
        all_files = self.cache_path.glob("*.parquet")
        return len(list(all_files)) == 0

    def _check_input_format(self, df: pd.DataFrame):
        # TODO check that incoming dataframe has proper columns
        pass

    def get_cache_scores(self, df: pd.DataFrame):
        """For a dataframe, join from cache. Returns

        (scored, unscored) dataframes"""
        self._check_input_format(df)

        if self.is_empty():
            cache_results = pd.DataFrame(
                [], columns=self.base_cols + ["score", "timestamp"]
            )
            cache_results = df.merge(cache_results, how="left", on=self.base_cols)
        else:
            # Using IS NOT DISTINCT FROM to handle nulls
            col_match_sql = " AND ".join(
                [f"df.{x} IS NOT DISTINCT FROM cache.{x}" for x in self.base_cols]
            )
            cache_results = duckdb.query(
                "SELECT df.*, cache.score, cache.confidence, cache.flags, cache.timestamp FROM "
                f"df LEFT JOIN '{self.cache_path}/*.parquet' cache ON {col_match_sql}"
            ).to_df()

        # Add a join with the in-memory cache
        in_memory_results = df.merge(
            self.in_memory_cache, how="left", on=self.base_cols
        )
        cache_results = pd.concat([cache_results, in_memory_results])

        cache_results = cache_results.drop_duplicates(self.base_cols)
        cache_results = cache_results.astype({col: "object" for col in self.base_cols})
        # force non-response score to be 1.
        cache_results.loc[cache_results.response.str.strip() == "", "score"] = 1

        unscored = cache_results[cache_results.score.isna()]
        scored = cache_results[~cache_results.score.isna()]

        self.logger.debug(
            f"To score:{cache_results.score.isna().sum()} / {len(cache_results)}"
        )

        return unscored, scored

    def write(self, df: pd.DataFrame, min_size_to_write: int = 0):
        """Write new scores to the cache.

        The parquet scorer is very basic in this regard: it writes
        a timestamped file to the cache_path.

        If smaller than min_size_to_write, it will save the cache in-memory
        """
        
        if df.empty:
            return

        # append in-memory cache
        if len(self.in_memory_cache) > 0:
            total_cache = pd.concat([self.in_memory_cache, df]).drop_duplicates(self.base_cols)
            self.logger.info("Writing to in-memory cache. Total cache size:", len(total_cache))
        else:
            total_cache = df

        if len(total_cache) > min_size_to_write:
            total_cache.to_parquet(self.cache_path / f"results.{time.time()}.parquet")
            self.in_memory_cache = pd.DataFrame(
                [], columns=self.base_cols + ["score", "timestamp"]
            )

        else:
            # append DF to in-memory cache
            self.in_memory_cache = pd.concat([self.in_memory_cache, df])

    def __del__(self):
        self.write(self.in_memory_cache, min_size_to_write=0)
