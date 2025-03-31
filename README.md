# Ocsai-Py

These are the materials for training and analyzing Ocsai - a model for evaluating various Originality Scoring tasks with Large Language models.

This includes:
- resources pertaining to the training of Ocsai 2
- tools for training automated scoring models
- [materials related to confidence measures and weighted probabilistic scoring](./notebooks/evaluation/LogProbsOcsai1.ipynb).

## Citations 

Forthcoming

## Data

Ocsai is trained on a great deal of data, from a number of past studies. The training data for the paper [Beyond semantic distance: Automated scoring of divergent thinking greatly improves with large language models](https://github.com/massivetexts/ocsai/tree/main/data/ocsai1) is in the data folder, but please see the Dataset preparation notebook at [cleanDatasets.ipynb](notebooks/cleanDatasets.ipynb) to see all include datasets and citations.

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