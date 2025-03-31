# Ocsai-Py

These are the materials for training and analyzing Ocsai - a model for evaluating various Originality Scoring tasks with Large Language models.

This includes:
- resources pertaining to the training of Ocsai 2
- tools for training automated scoring models
- materials related to confidence measures and weighted probabilistic scoring.

## Citations 

Forthcoming

## Data

Ocsai is trained on a great deal of data, from a number of past studies. Since not all the data is our own we don't redistribute all of it; rather, we provide code for downloading, normalizing, and cleaning the datasets.

See the Dataset preparation notebook at [cleanDatasets.ipynb](notebooks/cleanDatasets.ipynb) to see all include datasets and citations.

Please reach out to <peter.organisciak@du.edu> if you want help acquiring all of the data: we don't want it to be intimidating.

If you'd like to provide permission for us to bundle your data into this repository, let us know.


## Installing the Library

The code used in preparation is bundled in a library. It can be installed as follows: 

```
pip install git+https://www.github.com/massivetexts/ocsai.git
```

If you want a sandboxed virtual environment to try the code in this repository's notebooks:

```bash
git clone https://www.github.com/massivetexts/ocsai && cd ocsai
uv sync
```

This will make a virtual environment available in `./.venv`