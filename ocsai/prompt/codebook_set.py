import openai
from .codebook import Codebook, Dataset, Labels, Label, LabeledDataset, write_codebook
from pathlib import Path
import json
import random
from .utils import hashname
from tqdm.auto import trange


class CodebookSet:

    def __init__(self,
                 data: Dataset,
                 truth: Labels,
                 seed: int = 42,
                 name: str | None = None,
                 client: openai.OpenAI = openai.OpenAI()):
        '''
        Run codebook related experiments, keeping all data together.

        Includes ELO-based scoring.
        '''
        self.data: list = list(data)
        self.truth: list = list(truth)
        self.name: str = name if name else hashname()
        self.seed: int = seed

        # Shuffle data and truth zip with random seed
        random.seed(seed)
        shuffled_data: LabeledDataset = list(zip(self.data, self.truth))
        random.shuffle(shuffled_data)

        # Split into train and test sets

        train_size = int(0.8 * len(shuffled_data))
        self.train_data: LabeledDataset = shuffled_data[:train_size]
        self.test_data: LabeledDataset = shuffled_data[train_size:]

        self.client = client
        self.codebooks: list[Codebook] = []

    def add_codebook(self, codebook: Codebook) -> None:
        self.codebooks.append(codebook)

    def write_codebooks(self,
                        n: int = 10,
                        examples_per_codebook: int = 100,
                        item_description: str = "Items to predict.",
                        label_description: str = "The predicted labels for the items.",
                        scale_details: str | None = None,
                        model: str = "gpt-3.5-turbo",
                        temperature: float = 0.5,
                        name=None) -> None:
        '''
        Write multiple codebooks.

        n: Number of codebooks to write.

        examples_per_codebook: Max number of examples to use for each codebook.
        '''

        for i in trange(n):
            # select random examples_per_codebook from train_data, without shuffling
            # train_data
            range_indices = list(range(len(self.train_data)))
            random.shuffle(range_indices)
            train_ex: Dataset = []
            train_X: list[Label] = []
            run_name = name + f"_{i}" if name else None
            for idx in range_indices[:examples_per_codebook]:
                ex, label = self.train_data[idx]
                train_ex.append(ex)
                train_X.append(label)

            new_codebook = write_codebook(train_ex,
                                          train_X,
                                          item_description=item_description,
                                          label_description=label_description,
                                          scale_details=scale_details,
                                          model=model,
                                          temperature=temperature,
                                          name=run_name,
                                          parent=None)

            self.add_codebook(new_codebook)

    def a_b_test(self, n: int, codebook1: Codebook, codebook2: Codebook) -> float:
        '''
        Compare two codebooks on the same sample of data.

        n: Number of randomly sampled data points to compare.
        '''
        raise NotImplementedError

    def to_json(self, file_path: str | Path | None = None) -> dict | None:
        outdict = {
            "data": self.data,
            "truth": self.truth,
            "seed": self.seed,
            "name": self.name,
            "codebooks": [c.to_json() for c in self.codebooks]
        }
        if file_path:
            with open(file_path, "w") as file:
                json.dump(outdict, file)
            return None
        else:
            return outdict

    @staticmethod
    def read_json(file_path: str | Path) -> "CodebookSet":
        with open(file_path, "r") as file:
            data = json.load(file)
        return CodebookSet.from_json(data)

    @staticmethod
    def from_json(data: dict) -> "CodebookSet":
        codebooks = [Codebook.from_json(c) for c in data["codebooks"]]
        experiment = CodebookSet(data=data["data"],
                                 truth=data["truth"],
                                 seed=data['seed'],
                                 name=data["name"]
                                 )
        experiment.codebooks = codebooks
        return experiment
