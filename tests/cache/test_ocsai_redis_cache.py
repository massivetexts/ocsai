import pytest
import redis
import pandas as pd
import numpy as np
from ocsai.cache import Ocsai_Redis_Cache

# uses a docker image, so that we don't overwrite the actual redis database
from pytest_docker_tools import container, fetch

redis_image = fetch(repository='redis:latest')
redis_container = container(
    image="{redis_image.id}",
    scope="module",
    ports={"6379/tcp": None},  # Let Docker assign a random port
)


@pytest.fixture(scope="module")
def redis_client(redis_container):
    port = redis_container.ports["6379/tcp"][0]
    client = redis.Redis(host="localhost", port=port)
    client.flushdb()  # Clear the database before running tests
    yield client
    client.flushdb()  # Clear the database after running tests


@pytest.fixture
def cache(redis_client):
    redis_url = (
        f"redis://localhost:{redis_client.connection_pool.connection_kwargs['port']}"
    )
    return Ocsai_Redis_Cache(redis_url=redis_url)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame with all required columns."""
    return pd.DataFrame([{
        "prompt": "brick",
        "response": "use as a doorstop",
        "question": "What are creative uses for a brick?",
        "type": "uses",
        "language": "eng",
        "model": "gpt-4",
        "method": "top",
    }])


@pytest.fixture
def sample_df_with_scores(sample_df):
    """Sample DataFrame with score columns added."""
    df = sample_df.copy()
    df["score"] = 0.85
    df["confidence"] = 3
    df["flags"] = [[]]
    df["timestamp"] = 1627849187.123
    return df


class TestCacheInitialization:
    def test_cache_initialization(self, cache):
        assert cache.redis.ping() is True

    def test_base_cols_includes_method(self, cache):
        """Ensure base_cols matches the base class (includes 'method')."""
        expected = ["prompt", "response", "question", "type", "language", "model", "method"]
        assert cache.base_cols == expected

    def test_is_empty_on_fresh_cache(self, cache):
        cache.clear()
        assert cache.is_empty() is True


class TestKeyGeneration:
    def test_generate_cache_key(self, cache):
        row = {
            "prompt": "Test prompt",
            "response": "Test response",
            "question": "Test question",
            "type": "Test type",
            "language": "Test language",
            "model": "Test model",
            "method": "top",
        }
        key = cache._generate_cache_key(row)
        # Keys are sorted alphabetically by column name:
        # language, method, model, prompt, question, response, type
        expected_key = (
            "ocsai_cache:Test language:top:Test model:Test prompt:Test question:Test response:Test type"
        )
        assert key == expected_key

    def test_key_has_prefix(self, cache, sample_df):
        """Ensure all keys have the namespace prefix."""
        row = sample_df.iloc[0]
        key = cache._generate_cache_key(row)
        assert key.startswith("ocsai_cache:")


class TestWriteAndRead:
    def test_write_and_get_cache_scores(self, cache, sample_df_with_scores):
        cache.clear()

        # Write to cache
        cache.write(sample_df_with_scores)

        # Verify not empty
        assert cache.is_empty() is False

        # Retrieve from cache (query with just base cols)
        query_df = sample_df_with_scores[cache.base_cols].copy()
        to_score, cache_results = cache.get_cache_scores(query_df)

        assert len(to_score) == 0
        assert len(cache_results) == 1
        assert cache_results.iloc[0]["score"] == 0.85
        assert cache_results.iloc[0]["confidence"] == 3

    def test_write_empty_df(self, cache):
        """Writing an empty DataFrame should not error."""
        empty_df = pd.DataFrame([], columns=cache.base_cols + ["score", "confidence", "flags", "timestamp"])
        cache.write(empty_df)  # Should not raise

    def test_get_cache_scores_empty_df(self, cache):
        """Querying with empty DataFrame returns empty results."""
        empty_df = pd.DataFrame([], columns=cache.base_cols)
        to_score, cached = cache.get_cache_scores(empty_df)
        assert len(to_score) == 0
        assert len(cached) == 0


class TestPartialCacheRetrieval:
    def test_partial_cache_retrieval(self, cache):
        """Test that we correctly split cached vs uncached rows."""
        cache.clear()

        df_existing = pd.DataFrame([{
            "prompt": "Existing prompt",
            "response": "Existing response",
            "question": "Existing question",
            "type": "Existing type",
            "language": "Existing language",
            "model": "Existing model",
            "method": "top",
        }])
        df_existing["score"] = 0.8
        df_existing["confidence"] = 0.85
        df_existing["flags"] = [[]]
        df_existing["timestamp"] = 1627849187.123

        # Write existing data to cache
        cache.write(df_existing)

        df_new = pd.DataFrame([{
            "prompt": "New prompt",
            "response": "New response",
            "question": "New question",
            "type": "New type",
            "language": "New language",
            "model": "New model",
            "method": "top",
        }])

        combined_df = pd.concat([df_existing[cache.base_cols], df_new], ignore_index=True)

        # Retrieve from cache
        to_score, cache_results = cache.get_cache_scores(combined_df)

        assert len(to_score) == 1
        assert len(cache_results) == 1
        assert to_score.iloc[0]["prompt"] == "New prompt"
        assert cache_results.iloc[0]["prompt"] == "Existing prompt"
        assert cache_results.iloc[0]["score"] == 0.8


class TestColumnPreservation:
    def test_to_score_preserves_all_columns(self, cache):
        """The to_score DataFrame should preserve all columns from the original."""
        cache.clear()

        # Create a DataFrame with extra columns beyond base_cols
        df = pd.DataFrame([{
            "prompt": "test",
            "response": "test response",
            "question": "test question",
            "type": "uses",
            "language": "eng",
            "model": "gpt-4",
            "method": "top",
            "extra_col": "should be preserved",
            "another_col": 123,
        }])

        to_score, cached = cache.get_cache_scores(df)

        # to_score should have all original columns
        assert "extra_col" in to_score.columns
        assert "another_col" in to_score.columns
        assert to_score.iloc[0]["extra_col"] == "should be preserved"
        assert to_score.iloc[0]["another_col"] == 123


class TestSpecialValues:
    def test_handles_nan_values(self, cache):
        """Cache should handle NaN values correctly."""
        cache.clear()

        df = pd.DataFrame([{
            "prompt": "test",
            "response": "test response",
            "question": "",  # empty string
            "type": "uses",
            "language": "eng",
            "model": "gpt-4",
            "method": "top",
            "score": 0.5,
            "confidence": np.nan,  # NaN value
            "flags": [],
            "timestamp": 1627849187.123,
        }])

        cache.write(df)

        query_df = df[cache.base_cols].copy()
        to_score, cached = cache.get_cache_scores(query_df)

        assert len(cached) == 1
        assert cached.iloc[0]["score"] == 0.5
        # NaN should come back as NaN (or None after JSON round-trip)
        assert pd.isna(cached.iloc[0]["confidence"]) or cached.iloc[0]["confidence"] is None

    def test_handles_list_in_flags(self, cache):
        """Cache should correctly serialize/deserialize list values."""
        cache.clear()

        df = pd.DataFrame([{
            "prompt": "test",
            "response": "test response",
            "question": "test q",
            "type": "uses",
            "language": "eng",
            "model": "gpt-4",
            "method": "top",
            "score": 0.5,
            "confidence": 3,
            "flags": ["flag1", "flag2"],
            "timestamp": 1627849187.123,
        }])

        cache.write(df)

        query_df = df[cache.base_cols].copy()
        to_score, cached = cache.get_cache_scores(query_df)

        assert len(cached) == 1
        assert cached.iloc[0]["flags"] == ["flag1", "flag2"]


class TestClearCache:
    def test_clear_removes_all_entries(self, cache, sample_df_with_scores):
        """clear() should remove all cache entries."""
        # Write some data
        cache.write(sample_df_with_scores)
        assert cache.is_empty() is False

        # Clear
        deleted = cache.clear()
        assert deleted >= 1
        assert cache.is_empty() is True


class TestIntegrationWithScorer:
    """Tests that simulate how the scorer uses the cache."""

    def test_scorer_workflow(self, cache):
        """Simulate the full scorer workflow."""
        cache.clear()

        # 1. Scorer creates DataFrame from input (simulating base_scorer.py lines 409-414)
        # base_cols[:-2] = first 5 cols, then model and method added separately
        prompts = ["brick", "paper clip"]
        responses = ["doorstop", "bookmark"]
        questions = ["uses for brick?", "uses for paper clip?"]
        task_types = ["uses", "uses"]
        languages = ["eng", "eng"]

        df = pd.DataFrame(
            list(zip(prompts, responses, questions, task_types, languages)),
            columns=cache.base_cols[:-2],  # First 5 columns
        )
        df["model"] = "gpt-4"
        df["method"] = "top"

        # 2. Check cache
        to_score, cached = cache.get_cache_scores(df)
        assert len(to_score) == 2  # All need scoring
        assert len(cached) == 0

        # 3. Score and write results
        newly_scored = to_score.copy()
        newly_scored["score"] = [0.7, 0.8]
        newly_scored["confidence"] = [3, 4]
        newly_scored["flags"] = [[], ["creative"]]
        newly_scored["timestamp"] = 1627849187.123
        cache.write(newly_scored)

        # 4. Query again - should all be cached now
        to_score2, cached2 = cache.get_cache_scores(df)
        assert len(to_score2) == 0
        assert len(cached2) == 2

        # 5. Add one new item
        df_new = pd.DataFrame([{
            "prompt": "fork",
            "response": "back scratcher",
            "question": "uses for fork?",
            "type": "uses",
            "language": "eng",
            "model": "gpt-4",
            "method": "top",
        }])
        combined = pd.concat([df, df_new], ignore_index=True)

        to_score3, cached3 = cache.get_cache_scores(combined)
        assert len(to_score3) == 1  # Only new item
        assert len(cached3) == 2  # Previous items cached
        assert to_score3.iloc[0]["prompt"] == "fork"


class TestDuplicateInputRows:
    """Tests that duplicate input rows are properly deduplicated."""

    def _make_row(self, prompt="pencil", response="writing", question="uses?",
                  type_="uses", language="eng", model="gpt-4", method="top"):
        return {
            "prompt": prompt, "response": response, "question": question,
            "type": type_, "language": language, "model": model, "method": method,
        }

    def test_duplicate_uncached_rows_deduplicated(self, cache):
        """4 identical uncached rows should yield only 1 row in to_score."""
        cache.clear()
        rows = [self._make_row() for _ in range(4)]
        df = pd.DataFrame(rows)

        to_score, cached = cache.get_cache_scores(df)
        assert len(to_score) == 1
        assert len(cached) == 0

    def test_duplicate_cached_rows_deduplicated(self, cache):
        """4 identical cached rows should yield only 1 row in cache_results."""
        cache.clear()

        scored = pd.DataFrame([self._make_row()])
        scored["score"] = 0.75
        scored["confidence"] = 3
        scored["flags"] = [[]]
        scored["timestamp"] = 1627849187.123
        cache.write(scored)

        rows = [self._make_row() for _ in range(4)]
        df = pd.DataFrame(rows)

        to_score, cached = cache.get_cache_scores(df)
        assert len(to_score) == 0
        assert len(cached) == 1
        assert cached.iloc[0]["score"] == 0.75

    def test_mixed_duplicate_and_unique_rows(self, cache):
        """Mix of duplicated cached + duplicated uncached rows."""
        cache.clear()

        # Cache one item
        scored = pd.DataFrame([self._make_row(prompt="brick", response="doorstop")])
        scored["score"] = 0.8
        scored["confidence"] = 4
        scored["flags"] = [[]]
        scored["timestamp"] = 1627849187.123
        cache.write(scored)

        # Build input: 3x cached dupe + 4x uncached dupe + 1 unique uncached
        rows = (
            [self._make_row(prompt="brick", response="doorstop")] * 3
            + [self._make_row(prompt="pencil", response="writing")] * 4
            + [self._make_row(prompt="fork", response="eating")]
        )
        df = pd.DataFrame(rows)

        to_score, cached = cache.get_cache_scores(df)
        # 3 unique rows total: 1 cached, 2 uncached
        assert len(cached) == 1
        assert len(to_score) == 2

    def test_scorer_merge_with_duplicates(self, cache):
        """Simulate the full scorer merge path with duplicate input rows.

        Verifies that the output row count matches the input row count
        after the merge in originality_batch().
        """
        cache.clear()

        # Cache one item
        scored_row = self._make_row(prompt="brick", response="doorstop")
        scored = pd.DataFrame([scored_row])
        scored["score"] = 0.8
        scored["confidence"] = 4
        scored["flags"] = [[]]
        scored["timestamp"] = 1627849187.123
        cache.write(scored)

        # Input with duplicates (simulating originality_batch's df)
        rows = (
            [self._make_row(prompt="brick", response="doorstop")] * 3
            + [self._make_row(prompt="pencil", response="writing")] * 4
        )
        df = pd.DataFrame(rows)
        df = df.astype({col: "object" for col in cache.base_cols})

        to_score, cache_results = cache.get_cache_scores(df)

        # Simulate scoring the uncached items
        newly_scored = to_score.copy()
        newly_scored["score"] = [0.6]
        newly_scored["confidence"] = [3]
        newly_scored["flags"] = [[]]
        newly_scored["timestamp"] = 1627849187.123

        # Merge path from originality_batch
        right = pd.concat([cache_results, newly_scored])
        right = right.drop_duplicates(subset=cache.base_cols)
        final = df.merge(right, how="left", on=cache.base_cols)

        # Output row count must match input row count
        assert len(final) == len(df), (
            f"Merge produced {len(final)} rows but input had {len(df)}"
        )
        # All rows should have scores
        assert final["score"].notna().all()


if __name__ == "__main__":
    pytest.main()
