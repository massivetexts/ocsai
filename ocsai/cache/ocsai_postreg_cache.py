import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from .ocsai_cache import Ocsai_Cache
from sqlalchemy import create_engine


class Ocsai_Postgres_Cache(Ocsai_Cache):
    """A cache that utilizes PostgreSQL for storage and retrieval."""

    def __init__(self, db_url, cache_table='cache', **kwargs):
        super().__init__(**kwargs)
        self.db_url = db_url
        self.cache_table = cache_table
        self.engine = create_engine(db_url)
        self._create_table()

    def _create_table(self):
        with self.engine.connect() as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.cache_table} (
                    prompt TEXT,
                    response TEXT,
                    question TEXT,
                    type TEXT,
                    language TEXT,
                    model TEXT,
                    score NUMERIC,
                    confidence NUMERIC,
                    flags TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (prompt, question, type, language, model)
                );
            """)

    def is_empty(self):
        with self.engine.connect() as conn:
            result = conn.execute(f"SELECT COUNT(*) FROM {self.cache_table}").fetchone()
        return result[0] == 0

    def get_cache_scores(self, df: pd.DataFrame):
        self._check_input_format(df)
        # Convert DataFrame to SQL
        df.to_sql('temp_df', self.engine, if_exists='replace', index=False)
        query = f"""
            SELECT temp_df.*, c.score, c.confidence, c.flags, c.timestamp
            FROM temp_df
            LEFT JOIN {self.cache_table} c ON temp_df.prompt = c.prompt AND temp_df.question = c.question AND 
                                              temp_df.type = c.type AND temp_df.language = c.language AND 
                                              temp_df.model = c.model
        """
        cache_results = pd.read_sql_query(query, self.engine)
        unscored = cache_results[cache_results.score.isnull()]
        scored = cache_results[~cache_results.score.isnull()]
        return unscored, scored

    def write(self, df: pd.DataFrame):
        if df.empty:
            return
        df.to_sql(self.cache_table, self.engine, if_exists='append', index=False)

    def _check_input_format(self, df: pd.DataFrame):
        # Ensure df contains all necessary columns
        required_cols = set(self.base_cols + ['score', 'confidence', 'flags'])
        assert required_cols.issubset(df.columns), "DataFrame missing required columns"
