from typing import Literal
import openai
from .codebook import (
    Codebook,
    Dataset,
    EvalRun,
    Labels,
    Label,
    LabeledDataset,
    WriteProtocol,
    combine_codebooks,
    write_codebook,
)
from pathlib import Path
import json
import random
import anthropic
from .utils import hashname
from tqdm.auto import trange, tqdm
import pandas as pd


class CodebookSet:

    def __init__(
        self,
        data: Dataset,
        truth: Labels,
        seed: int = 42,
        name: str | None = None,
        client: openai.OpenAI | anthropic.Anthropic = openai.OpenAI(),
        train_proportion: float = 0.8,
        val_proportion: float = 0.1,
    ):
        """
        Run codebook related experiments, keeping all data together.

        Includes ELO-based scoring.
        """
        self.data: list = list(data)
        self.truth: list = list(truth)
        self.name: str = name if name else hashname()
        self.seed: int = seed

        # Shuffle data and truth zip with random seed
        random.seed(seed)
        shuffled_data: LabeledDataset = list(zip(self.data, self.truth))
        random.shuffle(shuffled_data)

        # Split into train and test sets

        train_size = int(train_proportion * len(shuffled_data))
        validation_size = int(val_proportion * len(shuffled_data))
        self.train_data: LabeledDataset = shuffled_data[:train_size]
        self.val_data: LabeledDataset = shuffled_data[
            train_size : train_size + validation_size
        ]
        self.test_data: LabeledDataset = shuffled_data[train_size + validation_size :]

        self.client = client
        self.codebooks: list[Codebook] = []

    def add_codebook(self, codebook: Codebook) -> None:
        self.codebooks.append(codebook)

    def write_codebooks(
        self,
        n: int = 10,
        examples_per_codebook: int = 100,
        item_description: str = "Items to predict.",
        label_description: str = "The predicted labels for the items.",
        scale_details: str | None = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.5,
        seed: int | None = None,
        protocol: WriteProtocol = "direct",
        name=None,
    ) -> None:
        """
        Write multiple codebooks.

        n: Number of codebooks to write.

        examples_per_codebook: Max number of examples to use for each codebook.
        """

        for i in trange(n):
            # select random examples_per_codebook from train_data, without shuffling
            # train_data
            range_indices = list(range(len(self.train_data)))
            if seed:
                random.seed(seed)
            random.shuffle(range_indices)
            train_ex: Dataset = []
            train_X: list[Label] = []
            run_name = name + f"_{i}" if name else None
            for idx in range_indices[:examples_per_codebook]:
                ex, label = self.train_data[idx]
                train_ex.append(ex)
                train_X.append(label)

            new_codebook = write_codebook(
                train_ex,
                train_X,
                item_description=item_description,
                label_description=label_description,
                scale_details=scale_details,
                model=model,
                temperature=temperature,
                name=run_name,
                protocol=protocol,
                parent=None,
                client=self.client,
            )

            self.add_codebook(new_codebook)

    def iterate_best_codebooks(
        self,
        n: int = 5,
        n_codebooks: int = 3,
        examples_per_codebook: int = 10,
        item_description: str = "Items to predict.",
        label_description: str = "The predicted labels for the items.",
        scale_details: str | None = None,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.5,
        include_predicted_labels: bool = False,
        debug: bool = False,
        seed: int | None = None,
    ) -> list[Codebook]:
        import random

        pool = self.val_data.copy()
        random.shuffle(pool)
        labeled_examples = pool[:examples_per_codebook]

        cbs_evals = []
        max_generation = max(
            [
                cb.metadata["generation"]
                for cb in self.codebooks
                if "generation" in cb.metadata
            ]
            + [0]
        )
        for cb in self.codebooks:
            try:
                evaluation = cb._fetch_evaluations()[0]
                cbs_evals.append((cb, evaluation))
                # max_generation = max(max_generation, evaluation["generation"]) if "generation" in evaluation else 0
            except IndexError:
                continue
        if "RMSE" in cbs_evals[0][1]["eval_results"]:
            cbs_evals = sorted(cbs_evals, key=lambda x: x[1]["eval_results"]["RMSE"])
        else:
            cbs_evals = sorted(
                cbs_evals, key=lambda x: x[1]["eval_results"]["accuracy"], reverse=True
            )

        # Take the top codebooks, then add a random selection from the next few places, for variety
        codebooks_to_iterate, evalruns = list(
            zip(
                *cbs_evals[: n_codebooks - 1]
                + [cbs_evals[random.randint(n_codebooks - 1, n_codebooks + 1)]]
            )
        )
        # reverse, so the best is at the end. This order effect was seen in the OPro paper
        codebooks_to_iterate = codebooks_to_iterate[::-1]
        evalruns = evalruns[::-1]

        new_cbs = []
        for i in range(n):
            new_cb = combine_codebooks(
                codebooks_to_iterate,
                evalruns,
                labeled_examples,
                model=model,
                debug=debug,
                generation=max_generation + 1,
                item_description=item_description,
                label_description=label_description,
                scale_details=scale_details,
                temperature=temperature,
                include_predicted_labels=include_predicted_labels,
                client=self.client,
            )
            self.codebooks.append(new_cb)
            new_cbs.append(new_cb)
        return new_cbs

    def evals(
        self,
        model: str | None = None,
        temperature: float | None = None,
        label_batch_size: int | None = None
    ) -> pd.DataFrame:
        ''' Format evaluations into a DataFrame.'''

        rows = []
        for cb in self.codebooks:
            # get all evals and filter later
            evals = cb._fetch_evaluations()
            for eval in evals:
                row = dict(cb.metadata.copy())
                row.update(eval['eval_results'])
                row.update({
                    'name': cb.name,
                    'evalmodel': eval['model'],  # model used to evaluate the codebook
                    'label_batch_size': 10, # evaluation batch size
                    'temperature': 0.0,
                    'test_n': len(eval['items']),
                })
                rows.append(row)
        df = pd.DataFrame(rows)
        # move some useful columns to the front
        firstcols = ['name', 'generation', 'model', 'protocol', 'parents', 'training_example_count',
                     'test_n', 'evalmodel', 'RMSE', 'pearsonr', 'label_batch_size', 'tokens', 'temperature']
        # renaming mainly for readability
        renamecols = {'training_example_count': 'train_n', 'generation': 'gen'}
        df = df[firstcols + [col for col in df.columns if col not in firstcols]].rename(columns=renamecols)
        if model:
            df = df[df['evalmodel'] == model]
        if temperature:
            df = df[df['temperature'] == temperature]
        if label_batch_size:
            df = df[df['label_batch_size'] == label_batch_size]
        return df

    def evaluate_codebooks(
        self,
        split: Literal["val", "test"] = "val",
        type: Literal["classification", "regression", "infer"] = "infer",
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        label_batch_size: int = 10,
    ) -> list[EvalRun]:
        """Evaluate all un-evaluated codebooks, on the validation or test set.
        Results are stored in the codebooks themselves.
        """
        if split == "val":
            data = self.val_data
        elif split == "test":
            data = self.test_data
        else:
            raise ValueError("split must be 'val' or 'test'.")
        items, targets = zip(*data)
        pbar = tqdm(self.codebooks, desc="Evaluating codebooks", leave=False)
        all_evals = []
        for codebook in pbar:
            try:
                evals = codebook._fetch_evaluations(
                    # items=items, # too specific
                    # targets=targets,
                    model=model,
                    temperature=temperature,
                    label_batch_size=label_batch_size,
                )
                if len(evals) > 0:
                    all_evals.append(evals[0])
                    continue
                eval: EvalRun = codebook.evaluate(
                    items,
                    targets,
                    model=model,
                    temperature=temperature,
                    label_batch_size=label_batch_size,
                )
                all_evals.append(eval)
            except KeyboardInterrupt:
                # gracefully stop the eval loop
                break
        return all_evals

    def a_b_test(self, n: int, codebook1: Codebook, codebook2: Codebook) -> float:
        """
        Compare two codebooks on the same sample of data.

        n: Number of randomly sampled data points to compare.
        """
        raise NotImplementedError

    def to_json(self, file_path: str | Path | None = None) -> dict | None:
        outdict = {
            "data": self.data,
            "truth": self.truth,
            "seed": self.seed,
            "name": self.name,
            "codebooks": [c.to_json() for c in self.codebooks],
        }
        if file_path:
            with open(file_path, "w") as file:
                json.dump(outdict, file, indent=4)
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
        experiment = CodebookSet(
            data=data["data"], truth=data["truth"], seed=data["seed"], name=data["name"]
        )
        experiment.codebooks = codebooks
        return experiment
