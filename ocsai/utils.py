def can_render_md_html():
    try:
        from IPython import get_ipython

        # Check if IPython is running
        ipython = get_ipython()
        if ipython is None:
            return False  # Not in an IPython environment

        # Check if the environment is a Jupyter notebook
        # This is a heuristic check and may not be foolproof
        if 'zmqshell' in str(type(ipython)):
            return True
        else:
            return False

    except ImportError:
        # IPython is not installed
        return False
 

def mprint(*messages):
    '''If renderable, print as markdown; else print as text.'''
    full_message = ' '.join(str(message) for message in messages)
    renderable = can_render_md_html()
    if renderable:
        from IPython.display import Markdown, display
        display(Markdown(full_message))
    else:
        print(full_message)


def upgrade_cache(cache_path, chunksize=10000, raise_on_error=True):
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    from pathlib import Path

    # upgrade cache - open each parquet file in the directory, ensure
    # that it has the right columns, and if not, add them. Then write
    # a new sharded parquet file.

    cache_path = Path(cache_path)
    cache_files = list(cache_path.glob('*.parquet'))
    data_collector = None

    all_new_files = []
    chunk_n = 0
    for cache_file in tqdm(cache_files):
        df = pd.read_parquet(cache_file)
        # drop rows where score is null or none or na
        df = df.dropna(subset=['score'])

        defaults = {
            'confidence': np.nan,
            'question': None,
            'flags': [],
            'language': 'eng',
            'type': 'uses'
        }
        for col, default in defaults.items():
            if col not in df.columns:
                df[col] = default

        # object types
        for col in ['prompt', 'response', 'question', 'type', 'language', 'model']:
            df[col] = df[col].astype('object')

        # float types
        for col in ['score', 'confidence']:
            df[col] = pd.to_numeric(df[col], errors='raise')
        
        # check if data collector is none
        if data_collector is None:
            data_collector = df
        else:
            data_collector = pd.concat([data_collector, df])

        # if data collector is bigger than chunk size, write
        # chunk_size number of rows to a file, and truncate them
        # from the data collector. This is done incrementally, to
        # avoid memory issues.
        while len(data_collector) > chunksize:
            chunk = data_collector.head(chunksize)
            data_collector = data_collector.iloc[chunksize:]
            ts = pd.Timestamp.now().strftime('%Y%m%d%H%M%S')
            fname = cache_path / f'results.{ts}.{chunk_n}.parquet'
            chunk.to_parquet(fname)
            all_new_files.append(fname)
            chunk_n += 1

    # Write the final chunk to a file
    if len(data_collector) > 0:
        ts = pd.Timestamp.now().strftime('%Y%m%d%H%M%S')
        fname = cache_path / f'results.{ts}.{chunk_n}.parquet'
        data_collector.to_parquet(fname)
        all_new_files.append(fname)

    # CHECK THAT NO DATA IS LOST
    original_ids = []
    for cache_file in tqdm(cache_files, desc='checking ids on old files'):
        df = pd.read_parquet(cache_file).dropna(subset=['score'])
        
        ids = (df['prompt'].astype(str) +
               df['response'].astype(str) +
               df['score'].astype(str)
               ).tolist()
        original_ids.extend(ids)

    new_ids = []
    for cache_file in tqdm(all_new_files, desc='checking ids on new files'):
        df = pd.read_parquet(cache_file)
        ids = (df['prompt'].astype(str) +
               df['response'].astype(str) +
               df['score'].astype(str)
              ).tolist()
        new_ids.extend(ids)
    print("Checking that all ids are preserved")
    print(f"Original ids: {len(original_ids)}")
    print(f"New ids: {len(new_ids)}")
    try:
        assert sorted(original_ids) == sorted(new_ids)
    except AssertionError:
        print("Something went wrong with optimization, deleting *new* files")
        print("len(original_ids):", len(original_ids), "len(new_ids): ", len(new_ids))
        print("Example mismatches (og<->new):", [(x,y) for (x,y) in list(zip(original_ids, new_ids)) if x != y][:10])
        [f.unlink() for f in all_new_files]
        if raise_on_error:
            raise
    print("All ids preserved, deleting old files")
    [f.unlink() for f in cache_files]