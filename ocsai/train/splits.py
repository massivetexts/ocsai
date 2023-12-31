import pandas as pd
import numpy as np


def default_split(size, train_size=0.8, val_size=0.05, test_size=0.15, random_seed=987):
    '''
    A traditionally randomized train/val/test split
    '''
    total = train_size + val_size + test_size
    if (total != 1):
        train_size = train_size/total
        val_size = val_size/total
        test_size = 1 - train_size - val_size

    names, probs = [], []
    for name, prob in zip(['train', 'val', 'test'], [train_size, val_size, test_size]):
        if prob > 0:
            names.append(name)
            probs.append(prob)
    rng = np.random.RandomState(random_seed)
    return rng.choice(names, size=size, p=probs)


def col_split(col, colname=None, include_in_train=[],
              train_size=.8, val_size=.05, test_size=.15, random_seed=987):
    '''
    Add a split based on the unique values of a column, where unique values can be *either*
        in the train set or the val+test set, but not both.

    Args:
        col (pd.Series): the column to split on
        include_in_train (list): a list of values to include in the train set. For example,
            in binning by prompt, 'brick' is so largely represented that we don't want it
            crowding out other options in val+test
        random_seed (int): random seed for reproducibility

    Returns:
        pd.Series: a column of 'train', 'val', or 'test' values
    '''
    total = train_size + val_size + test_size
    if (total != 1):
        train_size = train_size/total
        val_size = val_size/total
        test_size = 1 - train_size - val_size
    
    rng = np.random.RandomState(random_seed)
    if colname is None:
        colname = col.name

    values = col.unique()
    rng.shuffle(values)

    train_values = include_in_train
    trainmatch = col.isin(train_values)

    for value in values:
        if value in train_values:
            continue
        if trainmatch.sum() >= len(col)*train_size:
            break
        train_values.append(value)
        trainmatch = col.isin(train_values)
    
    vprob = val_size/(val_size+test_size)
    outseries = pd.Series(['train']*len(col), index=col.index)
    outseries[~trainmatch] = rng.choice(['val', 'test'],
                                        size=(~trainmatch).sum(),
                                        p=[vprob, 1-vprob])
    return outseries
