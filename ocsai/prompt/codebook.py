import json
import textwrap
from pathlib import Path
from typing import Literal, Union, Sequence
from typing_extensions import TypeAlias
import pandas as pd
import numpy as np
import openai
from .utils import strip_backticks

Item: TypeAlias = str
Labels: TypeAlias = Union[Sequence[str], Sequence[int], pd.Series, np.ndarray]
Dataset: TypeAlias = list[Item]


class ClassificationEvalResults:
    def __init__(self, accuracy: float, precision: float, recall: float, f1: float):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1


class RegressionEvalResults:
    def __init__(self, RMSE: float, pearsonr: float):
        self.RMSE = RMSE
        self.pearsonr = pearsonr

    def to_dict(self) -> dict:
        return {"RMSE": self.RMSE, "pearsonr": self.pearsonr}

    def __str__(self) -> str:
        return json.dumps(self.to_dict())

    def __repr__(self) -> str:
        return self.__str__()


EvalResults: TypeAlias = Union[ClassificationEvalResults, RegressionEvalResults]


class Codebook:
    def __init__(
        self,
        codebook: str,
        id: str | None = None,
        name: str | None = None,
        metadata: dict = {},
        client: openai.OpenAI = openai.OpenAI(),
    ):
        self.id = id
        self.name = name
        self.codebook = codebook
        self.metadata: dict = metadata
        self.client: openai.OpenAI = client

    def merge(self, other: "Codebook") -> "Codebook":
        # merge the codebook with another codebook
        raise NotImplementedError

    def to_json(self, file_path: str | Path | None = None) -> dict | None:
        outdict = {
            "codebook": self.codebook,
            "id": self.id,
            "name": self.name,
            "metadata": self.metadata,
        }
        if file_path:
            with open(file_path, "w") as file:
                json.dump(outdict, file)
            return None
        else:
            return outdict

    @staticmethod
    def read_json(file_path: str | Path) -> "Codebook":
        with open(file_path, "r") as file:
            data = json.load(file)
        return Codebook.from_json(data)

    @staticmethod
    def from_json(data: dict) -> "Codebook":
        # create a codebook from a json representation
        raise NotImplementedError

    def __str__(self) -> str:
        return self.codebook

    def __repr__(self) -> str:
        return self.codebook

    def correction(
        self, data: Dataset, truth: list[Labels], predicted: list[Labels]
    ) -> "Codebook":
        # iterate a codebook by observing the correct and incorrect labels it had created.
        raise NotImplementedError

    def summarize(self) -> "Codebook":
        # summarize a long codebook into a short one
        raise NotImplementedError

    @property
    def token_count(self) -> int:
        # return the number of tokens in the codebook, using tiktoken
        if "tokens" not in self.metadata:
            import tiktoken

            enc = tiktoken.encoding_for_model("gpt-4")
            self.metadata["tokens"] = len(enc.encode(self.codebook))
        return self.metadata["tokens"]

    def label(
        self, data: Dataset, model: str = "gpt-3.5-turbo", temperature: float = 0.0
    ) -> Labels:

        example_str = "- " + "\n- ".join(list(data))

        item_description = ""
        label_description = ""
        if "item_description" in self.metadata:
            item_description = self.metadata["item_description"]
        if "label_description" in self.metadata:
            label_description = self.metadata["label_description"]

        prompt_template = """
        Below is a set of items. Your goal is to label the item responses according to the codebook.

        {item_description}{label_description}

        Read the examples and write a detailed codebook outlining how to judge the responses.

        # CODEBOOK
        ------------

        {codebook}

        EXAMPLES TO LABEL
        -----------------

        {example_str}

        FORMAT DETAILS
        --------------

        Return all labels as a 1 dimensional JSON array of labels, surrounded by triple-backticks.
        """

        prompt_template = textwrap.dedent(prompt_template.strip())
        prompt = prompt_template.format(
            item_description=item_description,
            label_description=label_description,
            codebook=self.codebook,
            example_str=example_str,
        )

        client = openai.OpenAI()
        SYS_MSG = {"role": "system", "content": "Tag items acording to a codebook"}
        messages = [SYS_MSG, {"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            n=1,
            logprobs=None,
            max_tokens=1000,
        )
        content = response.choices[0].message.content
        if content:
            content = strip_backticks(content)
            data = json.loads(content)
            assert isinstance(data, list) and len(data) == len(
                list(data)
            ), "Output format didn't match"
            return data
        else:
            return []

    def evaluate(
        self,
        data: Dataset,
        truth: Labels,
        type: Literal["classification", "regression"],
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
    ) -> EvalResults:
        labels = self.label(data, model=model, temperature=temperature)
        return evaluate_labels(truth, labels, type)


def write_codebook(
    data: Dataset,
    truth: list[Labels],
    item_description: str | None,
    label_description: str | None,
    scale_details: str | None = None,
    format_details: str | None = None,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.0,
) -> Codebook:
    """
    Take a dataset of labels and extract a codebook.

    data: a list of items

    truth: a list of labels, one for each item in data

    item_description: a string describing what the items are, e.g.
         - "a list of responses to the question "What is a creativity use for a brick?"

    label_description: a string describing what the labels are. E.g.
        - "human-judged labels of creativity"

    scale_details: an optional string describing the scale, e.g.
        - "The scale is from 1.0 to 5.0, with 1.0 being not at all creative and 5.0 being very creative."

    format_details: an optional string describing additional considerations for the format of the labels, e.g.
        - "The labels should be in the form of a decimal number"
    """
    if item_description:
        item_description = item_description[0].lower() + item_description[1:]
        item_description = "- Item description: " + item_description
    else:
        item_description = "- Item description: a list of items"

    if label_description:
        label_description = label_description[0].lower() + label_description[1:]
        label_description = f"\n- Label description: {label_description}"
    else:
        label_description = "- Label description: human-judged labels for the items"

    if scale_details:
        scale_details = f"\n- Scale details: {scale_details}"
    else:
        scale_details = ""

    example_str = "\n".join(
        [f" - `{item}`: {label}" for item, label in zip(list(data), list(truth))]
    )

    prompt_template = """
    Below is a set of human-judged `item,label pairs. Your goal is to write a codebook for future human judges to use to label new items.

    {item_description}{label_description}{scale_details}

    Read the examples and write a detailed codebook outlining how to judge the responses.

    # EXAMPLES

    {example_str}

    # TASK

    Read all the examples and write a finely-detailed codebook describing how to label the items.

    {format_details}
    """
    prompt_template = textwrap.dedent(prompt_template.strip())
    prompt = prompt_template.format(
        item_description=item_description,
        label_description=label_description,
        scale_details=scale_details,
        example_str=example_str,
        format_details=format_details,
    )

    client = openai.OpenAI()
    SYS_MSG = {
        "role": "system",
        "content": "Please write a codebook for the following examples.",
    }
    messages = [SYS_MSG, {"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        n=1,
        logprobs=None,
        max_tokens=1000,
    )

    codebook = response.choices[0].message.content
    codebook = codebook if codebook else ""
    metadata = {
        "item_description": item_description,
        "label_description": label_description,
        "scale_details": scale_details,
        "format_details": format_details,
        "model": model,
        "source": "write_codebook",
        "temperature": temperature,
        "tokens": response.usage.completion_tokens if response.usage else 0,
    }
    return Codebook(codebook=codebook, metadata=metadata)


def evaluate_labels(
    truth: Labels,
    predicted: Labels,
    type: Literal["classification", "regression"]
) -> EvalResults:
    # grade the quality of the input on how descriptive it is, based on RMSE.
    if type == "classification":
        # calculate classification evaluation metrics (accuracy, precision, recall, f1)
        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        return ClassificationEvalResults(accuracy, precision, recall, f1)
    elif type == "regression":
        # calculate regression evaluation metrics (RMSE, pearsonr)
        RMSE = np.sqrt(np.mean((np.array(truth) - np.array(predicted)) ** 2))
        pearsonr = np.corrcoef(np.array(truth), np.array(predicted))[0, 1]
        return RegressionEvalResults(RMSE, pearsonr)
    else:
        raise ValueError(
            "Invalid evaluation type. Must be either "
            "'classification' or 'regression'."
        )
