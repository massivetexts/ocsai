'''
Download and process datasets for Ocsai, using an Ocsai dataset definition file.

Example usage:
```
data_dir = 'path/to/data/dir'  # Set your data directory
datasets = process_datasets('datasets_definition.yaml', data_dir)
```
'''

import pandas as pd
import yaml
import zipfile
from tqdm import tqdm
from pathlib import Path
import requests
from .preprocess import prep_general
from ..utils import mprint


def download_from_description(description, data_dir, extension='csv'):
    ''' Download a dataset from a description dictionary, which contains
    a name and a download. 
    
    The download value can be a dictionary, with `url` (required),
        `extension` (optional) and `archive_files` (representing the files
        to extract from an archive, optional). It can also be a string, 
        representing the url to download.

        You can include a list of multiple download dictionaries or url string.
    '''
    data_dir = Path(data_dir)
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
    
    urls = (description['meta']['download']
            if type(description['meta']['download']) is list
            else [description['meta']['download']]
            )
    
    name = description["name"]
    file_paths = []
    for i, download_desc in enumerate(urls):
        extract_files = None
        if type(download_desc) is dict:
            # overwrite extension parameter if it's in the dict
            if download_desc.get('extension'):
                extension = download_desc['extension']
            if download_desc.get('archive_files'):
                extract_files = download_desc['archive_files']
            download_url = download_desc['url']
        elif type(download_desc) is str:
            download_url = download_desc

        file_path = data_dir / f'{name}_{i}.{extension}'
        if not file_path.exists():
            file_path = download_file(download_url, file_path)
        
        # select files out of compressed archive
        if extract_files and extension == 'zip':
            for extract_file in extract_files:
                extract_file_path = file_path / extract_file
                if not extract_file_path.exists():
                    raise Exception(f"File {extract_file_path} not found in {file_path}")
                file_paths.append(extract_file_path)
        else:
            file_paths.append(file_path)

    return file_paths


def download_file(url, filename):
    filename = Path(filename)
    # for archived files, they're immediately decompressed
    dirpath = None  
    if (filename.suffix == '.zip'):
        dirpath = filename.parent / filename.stem
        if dirpath.exists():
            return dirpath
    elif filename.exists():
        return filename

    response = requests.get(url)

    with open(filename, 'wb') as file:
        file.write(response.content)

    # if extension is .zip, unzip to a directory 
    # with the same name
    if dirpath:
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(dirpath)
        # delete the zip file
        filename.unlink()
        print(dirpath)
        return dirpath
    else:
        return filename


def load_dataset(dataset, data_dir, verbose=True, include_rater_std=True):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
    download_url = dataset["download_url"]
    name = dataset["name"]
    file_path = data_dir / f'{name}.csv'

    # display message in markdown, if available (e.g. in Jupyter); else print
    if verbose:
        pname = dataset['inline'] if dataset.get('inline') else name
        mprint(f'# Loading *{pname}*')
        if dataset.get('citation'):
            mprint(dataset['citation'])

    if not file_path.exists():
        download_file(download_url, file_path)

    data = pd.read_csv(file_path)

    if dataset.get("type"):
        data['type'] = dataset['type']

    if dataset.get("null_marker"):
        # replace null marker with NaN
        data.response = data.response.replace(dataset["null_marker"], pd.NA)
        for col in dataset.get("rater_cols", []):
            data[col] = data[col].replace(dataset["null_marker"], pd.NA)

    if dataset.get("column_mappings"):
        data = data.rename(columns=dataset["column_mappings"])

    if dataset.get("additional_processing", {}).get("split_task_column"):
        data['prompt'] = data.task.apply(lambda x: x.split('_')[-1])

    if dataset.get("range"):
        original_range = dataset["range"]
    else:
        original_range = None
    
    prepped_data = prep_general(data,
                                name,
                                original_range=original_range,
                                include_rater_std=include_rater_std)
    return prepped_data


def load_dataset_definitions(definition_file, data_dir, verbose=True,
                             include_rater_std=False):
    with open(definition_file, 'r') as file:
        datasets_definition = yaml.safe_load(file)

    # datasets_definitions can be a list of dataset dictionaries, or
    # a single dataset dictionary
    if isinstance(datasets_definition, dict):
        datasets_definition = [datasets_definition]

    datasets = {}
    for dataset_def in tqdm(datasets_definition,
                            desc="Processing datasets"):
        datasets[dataset_def["name"]] = load_dataset(dataset_def,
                                                     data_dir,
                                                     verbose=verbose,
                                                     include_rater_std=include_rater_std)

    return datasets
