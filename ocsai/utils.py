import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import anthropic
import openai


def generic_llm(text,
                sysmsg,
                client: anthropic.Anthropic | openai.OpenAI,
                model: str = 'gpt-3.5-turbo',
                temperature: float = 0.0,
                max_tokens: int = 300,
                ) -> str:
    '''Run an openai or anthropic api call, based on the supplied client.'''
    common_args = {
        'model': model,
        'temperature': temperature,
        'max_tokens': max_tokens
    }
    if type(client) is anthropic.Anthropic:
        response = client.messages.create(
                system=sysmsg,
                messages=[
                    {'role': 'user', 'content': text}
                ],
                **common_args
            )

        content = response.content[0].text
    elif type(client) is openai.OpenAI:
        response = client.chat.completions.create(
            messages=[
                {'role': 'system', 'content': sysmsg},
                {'role': 'user', 'content': text}
            ],
            **common_args
        )
        content = response.choices[0].message.content
    else:
        raise ValueError("client must be either an anthropic.Anthropic or openai.OpenAI object.")
    return content

def can_render_md_html():
    try:
        from IPython import get_ipython

        # Check if IPython is running
        ipython = get_ipython()
        if ipython is None:
            return False  # Not in an IPython environment

        # Check if the environment is a Jupyter notebook
        # This is a heuristic check and may not be foolproof
        if "zmqshell" in str(type(ipython)):
            return True
        else:
            return False

    except ImportError:
        # IPython is not installed
        return False


def mprint(*messages):
    """If renderable, print as markdown; else print as text."""
    full_message = " ".join(str(message) for message in messages)
    renderable = can_render_md_html()
    if renderable:
        from IPython.display import Markdown, display

        display(Markdown(full_message))
    else:
        print(full_message)


def fingerprint_df(
    df, base_cols=["prompt", "response", "question", "type", "language", "model"]
) -> list[str]:
    import hashlib

    out_s = (df[base_cols]
        .astype(str)
        .apply(lambda x: hashlib.md5("".join(x).encode()).hexdigest(), axis=1)
    )
    if out_s.empty:
        return []

    try:
        return out_s.tolist()
    except AttributeError:
        print(out_s)
        raise


def set_cache_dtypes(df):
    # string cols
    for col in ["prompt", "response", "question", "type", "language", "model"]:
        if col in df.columns:
            df[col] = df[col].astype(object).fillna("").astype(str)

    # object cols
    for col in ["flags"]:
        if col in df.columns:
            df[col] = df[col].astype(object)

    # numeric cols (allowing errors)
    for col in ["score", "confidence"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")

    # float cols
    for col in ["timestamp"]:
        if col in df.columns:
            df[col] = df[col].astype(float)

    return df


def upgrade_cache(
    cache_path,
    chunksize=1000000,
    base_cols=["prompt", "response", "question", "type", "language", "model"],
    raise_on_error=True,
):

    # upgrade cache - open each parquet file in the directory, ensure
    # that it has the right columns, and if not, add them. Then write
    # a new sharded parquet file.

    cache_path = Path(cache_path)
    cache_files = list(cache_path.glob("*.parquet"))
    data_collector = None
    all_hashes = set()

    try:
        all_new_files = []
        chunk_n = 0
        for cache_file in tqdm(cache_files):
            df = pd.read_parquet(cache_file)
            df = set_cache_dtypes(df)
            # drop rows where score is null or none or na
            df = df[~df["score"].replace("", pd.NA).isna()]
            # fingerprint the dataframe and remove any hashes that are already in all_hashes
            hashes = fingerprint_df(df, [c for c in base_cols if c in df.columns])
            hashes = [h in all_hashes for h in hashes]
            hashes_to_drop = sum(hashes)
            if hashes_to_drop > 0:
                print(
                    f"Found {hashes_to_drop} hashes already in all_hashes, dropping them"
                )
                df = df[~hashes]
            all_hashes.update(hashes)

            defaults = {
                "confidence": np.nan,
                "question": None,
                "flags": None,
                "language": "eng",
                "type": "uses",
            }
            for col, default in defaults.items():
                if col not in df.columns:
                    df[col] = default

            # object types
            for col in ["prompt", "response", "question", "type", "language", "model"]:
                df[col] = df[col].astype("object")

            # float types
            for col in ["score", "confidence"]:
                df[col] = pd.to_numeric(df[col], errors="ignore")

            # check if data collector is none
            if data_collector is None:
                data_collector = df
            else:
                data_collector = pd.concat([data_collector, df])

            # DROP WHERE SCORE IS NULL
            data_collector = data_collector.dropna(subset=["score"])

            # Drop duplicates, based on the base columns
            data_collector = data_collector.drop_duplicates(subset=base_cols)

            # if data collector is bigger than chunk size, write
            # chunk_size number of rows to a file, and truncate them
            # from the data collector. This is done incrementally, to
            # avoid memory issues.
            if data_collector is not None:
                while len(data_collector) > chunksize:
                    chunk = data_collector.head(chunksize)
                    data_collector = data_collector.iloc[chunksize:]
                    ts = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
                    fname = cache_path / f"results.{ts}.{chunk_n}.parquet"
                    set_cache_dtypes(chunk).to_parquet(fname)
                    all_new_files.append(fname)
                    chunk_n += 1

        # Write the final chunk to a file
        if data_collector is not None and len(data_collector) > 0:
            ts = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
            fname = cache_path / f"results.{ts}.{chunk_n}.parquet"
            set_cache_dtypes(data_collector).to_parquet(fname)
            all_new_files.append(fname)

    except KeyboardInterrupt:
        print("KeyboardInterrupt, cleaning up")
        for f in all_new_files:
            try:
                f.unlink()
            except FileNotFoundError:
                print("File not found for unlinking:", f)
        if raise_on_error:
            raise

    # CHECK THAT NO DATA IS LOST
    original_ids = []
    for cache_file in tqdm(cache_files, desc="checking ids on old files"):
        df = pd.read_parquet(cache_file)
        df = df[df["score"].replace('', pd.NA).notna()]
        df = set_cache_dtypes(df)
        ids = fingerprint_df(df, base_cols)
        original_ids.extend(ids)
    original_ids = set(original_ids)

    new_ids = []
    for cache_file in tqdm(all_new_files, desc="checking ids on new files"):
        df = pd.read_parquet(cache_file)
        ids = fingerprint_df(df, base_cols)
        new_ids.extend(ids)
    new_ids = set(new_ids)

    print("Checking that all ids are preserved")
    print(f"Original ids: {len(original_ids)}")
    print(f"New ids: {len(new_ids)}")
    try:
        assert sorted(original_ids) == sorted(new_ids)
    except AssertionError:
        print("Something went wrong with optimization, deleting *new* files")
        print("len(original_ids):", len(original_ids), "len(new_ids): ", len(new_ids))
        print(
            "Example mismatches (og<->new):",
            [(x, y) for (x, y) in list(zip(original_ids, new_ids)) if x != y][:10],
        )
        for f in all_new_files:
            try:
                f.unlink()
            except FileNotFoundError:
                print("File not found for unlinking:", f)
        if raise_on_error:
            raise
    print("All ids preserved, deleting old files")
    for f in cache_files:
        try:
            f.unlink()
        except FileNotFoundError:
            print("File not found for unlinking:", f)
