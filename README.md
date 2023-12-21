# Ocsai Large

These are the materials for training and analyzing Ocsai Large - a model for evaluating various Originality Scoring tasks with Large Language models.

## Citation 

Forthcoming

## Data

Ocsai is trained on a great deal of data, from a number of past studies. Since not all the data is our own we don't redistribute all of it; rather, we provide code for downloading, normalizing, and cleaning the datasets.

See the Dataset preparation notebook at [cleanDatasets.ipynb](notebooks/cleanDatasets.ipynb) to see all include datasets and citations.

Please reach out to <peter.organisciak@du.edu> if you want help acquiring all of the data: we don't want it to be intimidating.

If you'd like to provide permission for us to bundle your data into this repository, let us know.

TODO include sample.csv

## Installing the Library

The code used in preparation is bundled in a library. It can be installed as follows: 

```bash
git clone https://www.github.com/massivetexts/ocsai
cd ocsai
pip install -e .
```