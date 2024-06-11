import pytest
import redis
import pandas as pd
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


def test_cache_initialization(cache):
    assert cache.redis.ping() is True


def test_generate_cache_key(cache):
    row = {
        "prompt": "Test prompt",
        "response": "Test response",
        "question": "Test question",
        "type": "Test type",
        "language": "Test language",
        "model": "Test model",
    }
    key = cache._generate_cache_key(row)
    expected_key = (
        "Test language:Test model:Test prompt:Test question:Test response:Test type"
    )
    assert key == expected_key


def test_write_and_get_cache_scores(cache):
    df = pd.DataFrame(
        [
            {
                "prompt": "Test prompt",
                "response": "Test response",
                "question": "Test question",
                "type": "Test type",
                "language": "Test language",
                "model": "Test model",
            }
        ]
    )
    df["score"] = 0.9
    df["confidence"] = 3
    df["flags"] = [[]]
    df["timestamp"] = 1627849187.123

    # Write to cache
    cache.write(df)

    # Retrieve from cache
    to_score, cache_results = cache.get_cache_scores(df)

    assert len(to_score) == 0
    assert len(cache_results) == 1
    assert cache_results.iloc[0]["score"] == 0.9
    assert cache_results.iloc[0]["confidence"] == 3


def test_partial_cache_retrieval(cache):
    df_existing = pd.DataFrame(
        [
            {
                "prompt": "Existing prompt",
                "response": "Existing response",
                "question": "Existing question",
                "type": "Existing type",
                "language": "Existing language",
                "model": "Existing model",
            }
        ]
    )
    df_existing["score"] = 0.8
    df_existing["confidence"] = 0.85
    df_existing["flags"] = [[]]
    df_existing["timestamp"] = 1627849187.123

    # Write existing data to cache
    cache.write(df_existing)

    df_new = pd.DataFrame(
        [
            {
                "prompt": "New prompt",
                "response": "New response",
                "question": "New question",
                "type": "New type",
                "language": "New language",
                "model": "New model",
            }
        ]
    )

    combined_df = pd.concat([df_existing, df_new])

    # Retrieve from cache
    to_score, cache_results = cache.get_cache_scores(combined_df)

    assert len(to_score) == 1
    assert len(cache_results) == 1
    assert to_score.iloc[0]["prompt"] == "New prompt"
    assert cache_results.iloc[0]["prompt"] == "Existing prompt"
    assert cache_results.iloc[0]["score"] == 0.8


if __name__ == "__main__":
    pytest.main()
