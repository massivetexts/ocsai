import json
import textwrap
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, Union, MutableSequence
from typing_extensions import TypeAlias
import pandas as pd
import numpy as np
import openai
from .utils import strip_backticks, hashname

Item: TypeAlias = str
Label: TypeAlias = Union[str, int]
Labels: TypeAlias = Union[MutableSequence[Label], pd.Series, np.ndarray]
Dataset: TypeAlias = MutableSequence[Item]
LabeledDataset: TypeAlias = MutableSequence[tuple[Item, Label]]

WriteProtocol: TypeAlias = Literal["direct", "CoT"]
IterateProtocol: TypeAlias = Literal["summarize", "merge", "correct"]

class ClassificationEvalResults:
    def __init__(
        self,
        accuracy: np.floating,
        precision: np.floating,
        recall: np.floating,
        f1: np.floating,
    ):
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


class CodebookMetadata(TypedDict):
    item_description: NotRequired[str]
    label_description: NotRequired[str]
    scale_details: NotRequired[str | None]
    format_details: NotRequired[str | None]
    model: NotRequired[str]
    source: NotRequired[str]
    parents: NotRequired[list[str]]
    protocol: NotRequired[WriteProtocol | IterateProtocol]
    temperature: NotRequired[float]
    tokens: NotRequired[int]

class Codebook:
    def __init__(
        self,
        codebook: str,
        id: str | None = None,
        name: str | None = None,
        metadata: CodebookMetadata = {},
        client: openai.OpenAI = openai.OpenAI(),
    ):
        self.id = id
        self.name = name
        self.codebook = codebook
        self.metadata: CodebookMetadata = metadata
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
        # at a minimum, the json representation should have a "codebook" key
        return Codebook(
            codebook=data["codebook"],
            id=data["id"] if "id" in data else None,
            name=data["name"] if "name" in data else None,
            metadata=data["metadata"] if "metadata" in data else {},
        )

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
        self,
        data: Dataset,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        max_retries: int = 3,
        temp_backoff: float = 0.1,
    ) -> Labels:
        '''
        Label a dataset of items according to the codebook.

        data: a list of items to label
        model: A valid model for the openai API
        temperature: A temperature to use when sampling from the model
        max_retries: The maximum number of retries to attempt when the 
            model fails to return a valid response
        temp_backoff: The amount to increase the temperature by on each retry
        '''

        while max_retries >= 0:
            try:
                labels = self._label(data, model, temperature)
                break
            except ValueError:
                max_retries -= 1
                temperature += temp_backoff
                continue
        return labels

    def _label(
        self,
        data: Dataset,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.0
    ) -> Labels:
        ''' Label a dataset of items according to the codebook
          Use 'label' method instead,
        which includes error handling
        '''

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
            newdata = json.loads(content)
            if not isinstance(newdata, list):
                raise ValueError("Output format didn't match")
            if not len(newdata) == len(list(data)):
                raise ValueError("Output format didn't match: mismatched length")
            return newdata
        else:
            return []

    def evaluate(
        self,
        data: Dataset,
        truth: Labels,
        type: Literal["classification", "regression"],
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        label_batch_size: int = 20,
    ) -> EvalResults:
        '''
        
        '''
        all_labels: list[Label] = []
        batches: list[Dataset] = []
        if len(data) > label_batch_size:
            # split into batches
            for i in range(0, len(data), label_batch_size):
                batches.append(data[i:i + label_batch_size])
        else:
            batches = [data]

        for batch in batches:
            labels = self.label(batch, model=model, temperature=temperature)
            all_labels.extend(labels)
        return evaluate_labels(truth, labels, type)


def write_codebook(
    data: Dataset,
    truth: Labels,
    item_description: str | None,
    label_description: str | None,
    scale_details: str | None = None,
    format_details: str | None = None,
    protocol: WriteProtocol = "direct",
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.5,
    name: str | None = None,
    parent: str | None = None,
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
        - "The scale is from 1.0 to 5.0, with 1.0 being not at all creative and 5.0 being 
            very creative."

    format_details: an optional string describing additional considerations for the format 
    of the labels, e.g.
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
        max_tokens=5000,
    )

    codebook = response.choices[0].message.content
    codebook = codebook if codebook else ""
    
    metadata: CodebookMetadata = {
        "item_description": item_description,
        "label_description": label_description,
        "scale_details": scale_details,
        "format_details": format_details,
        "model": model,
        "source": "write_codebook",
        "protocol": protocol,
        "temperature": temperature,
        "parents": [parent] if parent else [],
        "tokens": response.usage.completion_tokens if response.usage else 0,
    }
    return Codebook(codebook=codebook, name=name, metadata=metadata)


def evaluate_labels(
    truth: Labels, predicted: Labels, type: Literal["classification", "regression"]
) -> EvalResults:
    if type == "classification":
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
        )

        accuracy = accuracy_score(truth, predicted)
        precision = precision_score(truth, predicted, average="weighted")
        recall = recall_score(truth, predicted, average="weighted")
        f1 = f1_score(truth, predicted, average="weighted")
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
