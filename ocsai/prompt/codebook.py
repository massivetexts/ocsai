import json
import textwrap
from pathlib import Path
from typing import Literal, TypedDict, MutableSequence
import anthropic
from typing_extensions import TypeAlias, NotRequired
import pandas as pd
import numpy as np
import openai
from tqdm.auto import tqdm
from .utils import strip_backticks, hashname

Item: TypeAlias = str
Label: TypeAlias = str | int | float
Labels: TypeAlias = MutableSequence[Label] | pd.Series | np.ndarray
Dataset: TypeAlias = MutableSequence[Item]
LabeledDataset: TypeAlias = MutableSequence[tuple[Item, Label]]

WriteProtocol: TypeAlias = Literal["direct", "direct-bullet", "CoT"]
IterateProtocol: TypeAlias = Literal["summarize", "merge", "correct"]


class ClassificationEvalResults(TypedDict):
    accuracy: np.floating | float
    precision: np.floating | float
    recall: np.floating | float
    f1: np.floating | float


class RegressionEvalResults(TypedDict):
    RMSE: float
    pearsonr: float


EvalResults: TypeAlias = ClassificationEvalResults | RegressionEvalResults


class EvalRun(TypedDict):
    eval_results: EvalResults
    items: Dataset
    targets: Labels
    predictions: Labels
    label_batch_size: int
    model: str
    temperature: float
    split: NotRequired[str]


class CodebookMetadata(TypedDict):
    item_description: NotRequired[str | None]
    label_description: NotRequired[str | None]
    scale_details: NotRequired[str | None]
    format_details: NotRequired[str | None]
    model: NotRequired[str]
    source: NotRequired[str]
    parents: NotRequired[list[str]]
    generation: NotRequired[int | None]
    protocol: NotRequired[WriteProtocol | IterateProtocol]
    temperature: NotRequired[float]
    tokens: NotRequired[int]
    timestamp: NotRequired[str]
    training_examples: NotRequired[LabeledDataset]
    training_example_count: NotRequired[int]


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
        if name:
            self.name = name
        else:
            self.name = hashname()
        self.codebook = codebook
        self.metadata: CodebookMetadata = metadata
        self.evaluations: list[EvalRun] = []

        if "timestamp" not in self.metadata:
            self.metadata["timestamp"] = pd.Timestamp.now().isoformat()
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
            "evaluations": self.evaluations,
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
        cb = Codebook(
            codebook=data["codebook"],
            id=data["id"] if "id" in data else None,
            name=data["name"] if "name" in data else None,
            metadata=data["metadata"] if "metadata" in data else {},
        )
        cb.evaluations = data["evaluations"] if "evaluations" in data else []
        return cb

    def __str__(self) -> str:
        return self.codebook

    def __repr__(self) -> str:
        return self.codebook

    def _repr_markdown_(self) -> str:
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
        max_retries: int = 4,
        temp_backoff: float = 0.15,
        label_batch_size: int = 10,
        type_check: Literal["float", "int", "str"] | None = None,
    ) -> Labels:
        """
        Label a dataset of items according to the codebook.

        data: a list of items to label
        model: A valid model for the openai API
        temperature: A temperature to use when sampling from the model
        max_retries: The maximum number of retries to attempt when the
            model fails to return a valid response
        temp_backoff: The amount to increase the temperature by on each retry
        label_batch_size: The number of items to label in each batch
        """
        all_labels: list[Label] = []
        batches: list[Dataset] = []
        if len(data) > label_batch_size:
            # split into batches
            for i in range(0, len(data), label_batch_size):
                batches.append(data[i : i + label_batch_size])
        else:
            batches = [data]

        pbar = tqdm(batches, desc="Labeling", leave=False)
        for batch in pbar:
            remaining_retries = max_retries
            while remaining_retries >= 0:
                try:
                    labels = self._label(
                        batch, model, temperature, type_check=type_check
                    )
                    if type_check == "str":
                        labels = [str(x) for x in labels]
                    elif type_check == "int":
                        labels = [int(x) for x in labels]
                    elif type_check == "float":
                        labels = [float(x) for x in labels]
                    all_labels.extend(labels)
                    break
                except ValueError:
                    remaining_retries = remaining_retries - 1
                    temperature += temp_backoff
                    continue
            if remaining_retries < 0:
                raise ValueError("Failed to label a batch after max retries")
        return all_labels

    def _label(
        self,
        data: Dataset,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        examples: LabeledDataset | None = None,
        type_check: Literal["float", "int", "str"] | None = None,
    ) -> Labels:
        """Label a dataset of items according to the codebook
          Use 'label' method instead,
        which includes error handling
        """

        example_str = "- " + "\n- ".join(list(data))

        item_description = ""
        label_description = ""
        if "item_description" in self.metadata:
            item_description = self.metadata["item_description"]
        if "label_description" in self.metadata:
            label_description = self.metadata["label_description"]

        if not examples:
            labeled_examples = ""
        else:
            labeled_examples = (
                "LABELED EXAMPLES\n"
                "------------\n"
                "\n".join([f" - `{item}`: {label}" for item, label in examples])
            )
        type_comment = ""
        if type_check == "str":
            type_comment = "Each label should be a string."
        elif type_check == "int":
            type_comment = "Each label should be an integer."
        elif type_check == "float":
            type_comment = "Each label should be a float."
        prompt_template = """
        Below is a set of items. Your goal is to label the item responses according to the codebook.

        {item_description}{label_description}

        # CODEBOOK
        ------------

        {codebook}
        {labeled_examples}
        EXAMPLES TO LABEL
        -----------------

        {example_str}

        FORMAT DETAILS
        --------------

        {type_comment} Return all labels as a 1 dimensional JSON array of labels, surrounded by triple-backticks.
        """

        prompt_template = textwrap.dedent(prompt_template)
        prompt = prompt_template.format(
            item_description=item_description,
            label_description=label_description,
            codebook=self.codebook,
            example_str=example_str,
            labeled_examples=labeled_examples,
            type_comment=type_comment,
        ).strip()

        SYS_MSG = {"role": "system", "content": "Tag items acording to a codebook"}
        messages = [SYS_MSG, {"role": "user", "content": prompt}]
        if type(self.client) is anthropic.Anthropic:
            response = self.client.messages.create(
                model=model,
                messages=messages[1:],
                system=SYS_MSG["content"],
                temperature=temperature,
                max_tokens=4096,
            )
            content = response.content[0].text
        elif type(self.client) is openai.OpenAI:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=4096,
            )
            content = response.choices[0].message.content

        if content:
            content = strip_backticks(content)
            if "[" not in content and "," in content:
                # try to fix it
                content = f"[{content}]"
            if "[" not in content and "," not in content:
                # see if it's newline separated
                if content.count("\n") >= (len(data) - 1):
                    content = content.replace("\n", ",")
                    content = f"[{content}]"
            try:
                newdata = json.loads(content)[: len(data)]
            except json.JSONDecodeError:
                print("Error decoding this response:", content)
                print("Input looks like:", example_str)
                import tiktoken

                enc = tiktoken.encoding_for_model("gpt-4")
                print(f"Input tokens: {len(enc.encode(example_str))}")
                print(f"Output tokens: {len(enc.encode(content))}")
                raise
            if not isinstance(newdata, list):
                raise ValueError("Output format didn't match")
            if not len(newdata) == len(list(data)):
                raise ValueError("Output format didn't match: mismatched length")
            return newdata
        else:
            return []

    def _fetch_evaluations(
        self,
        items: Dataset | None = None,
        targets: Labels | None = None,
        model: str | None = None,
        temperature: float | None = None,
        label_batch_size: int | None = None,
    ) -> list[EvalRun]:
        """Fetch saved evaluations"""
        matches = []
        targetvals = {
            "items": items,
            "targets": targets,
            "model": model,
            "temperature": temperature,
            "label_batch_size": label_batch_size,
        }
        for eval_run in self.evaluations:
            for key, keyvar in targetvals.items():
                if not keyvar:  # type: ignore
                    continue
                elif (key in eval_run) and (eval_run[key] == keyvar):
                    continue
                elif ((key == 'model') and ('model' in eval_run)):
                    if keyvar.startswith('gpt-4-turbo') and eval_run[key].startswith('gpt-4-turbo'):
                        continue
                    else:
                        break
                else:
                    break
            else:
                matches.append(eval_run)
        return matches

    def evaluate(
        self,
        data: Dataset,
        truth: Labels,
        type: Literal["classification", "regression", "infer"] = "infer",
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        label_batch_size: int = 10,
        force: bool = False,
    ) -> EvalRun:
        assert len(data) == len(truth), "Data and truth must be the same length"
        # see if we have already evaluated this data
        if not force:
            eval_runs = self._fetch_evaluations(
                items=data,
                targets=truth,
                model=model,
                temperature=temperature,
                label_batch_size=label_batch_size,
            )
            if len(eval_runs) > 0:
                return eval_runs[0]

        if type == "infer":
            if all((isinstance(x, int) or isinstance(x, float)) for x in truth):
                type = "regression"
            elif all(isinstance(x, str) for x in truth):
                type = "classification"
            else:
                raise ValueError(
                    "Could not infer type of labels. Must be either "
                    "'classification' or 'regression'."
                )
        if type == "classification":
            type_check = "str"
        elif type == "regression":
            type_check = "float"

        all_labels = self.label(
            data,
            model=model,
            temperature=temperature,
            label_batch_size=label_batch_size,
            type_check=type_check,
        )
        assert len(all_labels) == len(truth), "Labels and truth must be the same length"
        eval_results = evaluate_labels(truth, all_labels, type)

        new_eval_run: EvalRun = {
            "eval_results": eval_results,
            "items": data,
            "targets": truth,
            "predictions": all_labels,
            "label_batch_size": label_batch_size,
            "model": model,
            "temperature": temperature,
        }
        self.evaluations.append(new_eval_run)
        return new_eval_run


def combine_codebooks(
    codebooks: list[Codebook],
    evalruns: list[EvalRun],
    labeled_examples: LabeledDataset | None = None,
    model: str = "gpt-3.5-turbo",
    temperature: float = 1,
    name: str | None = None,
    item_description: str | None = None,
    label_description: str | None = None,
    scale_details: str | None = None,
    format_details: str | None = None,
    debug: bool = False,
    generation: int | None = None,
    include_predicted_labels: bool = False,
    type_check: Literal["float", "int", "str"] | None = None,
    client: openai.OpenAI | anthropic.Anthropic = openai.OpenAI(),
) -> Codebook:
    """Combine well-performing models, along with their evaluations,
    into a single codebook.

    evalruns: list of eval runs, one per codebook
    """
    snippets = codebook_prompt_snippets(
        labeled_examples,
        item_description,
        label_description,
        scale_details,
        format_details,
    )
    labeled_example_note = ""
    if labeled_examples:
        labeled_example_note = "There is also a small set of example items and labels."
        label_collector = pd.DataFrame(labeled_examples, columns=["item", "TRUTH"])
        example_data, example_labels = list(zip(*labeled_examples))

    codebook_strs = ""
    for i, (cb, evalrun) in enumerate(zip(codebooks, evalruns)):
        eval_results_str = ", ".join(
            [f"{k}: {np.round(v, 2)}" for k, v in evalrun["eval_results"].items()]
        )
        codebook_strs += f"## Codebook {i} (Performance: `{eval_results_str}`)\n\n```\n{cb.codebook}\n```\n\n"

        # label the examples by notebook - currently model is hardcoded
        if labeled_examples and include_predicted_labels:
            label_collector[f"Codebook {i} label"] = cb.label(
                list(example_data),
                model="gpt-4o",
                temperature=temperature,
                type_check=type_check,
            )

    label_example_str = ""
    if labeled_examples:
        label_example_str = (
            "# EXAMPLES\n\n```csv\n" + label_collector.to_csv(index=False) + "```\n\n"
        )

    prompt_template = """
    Your goal is to write an IMPROVED codebook for human judges to use to label new items. INTEGRATE/BORROW/COMBINE THE BEST ELEMENTS FROM THE PREVIOUS CODEBOOKS, WHILE IMPROVING ON THEM WHEN POSSIBLE.

    Below are examples of previous codebooks, as well as a measure of how well coders following those codebooks operated.{labeled_example_note}

    # TASK DETAILS

    {item_description}{label_description}{scale_details}

    Read the examples and write a detailed codebook outlining how to judge the responses.

    # CODEBOOKS

    {codebooks_strs}
    {label_example_str}

    # TASK

    Improve, integrate, and combine the existing codebooks, and write a succinct but finely-detailed codebook describing how to label the items.

    {format_details}
    """
    prompt_template = textwrap.dedent(prompt_template)
    prompt = prompt_template.format(
        item_description=snippets["item_description"],
        label_description=snippets["label_description"],
        scale_details=snippets["scale_details"],
        example_str=snippets["example_str"],
        format_details=snippets["format_details"],
        labeled_example_note=labeled_example_note,
        label_example_str=label_example_str,
        codebooks_strs=codebook_strs,
    ).strip()

    if debug:
        from IPython.display import display, Markdown

        display(Markdown(prompt))

    SYS_MSG = {
        "role": "system",
        "content": "Please write a codebook based on the following instructions.",
    }
    messages = [SYS_MSG, {"role": "user", "content": prompt}]

    if type(client) is anthropic.Anthropic:
        response = client.messages.create(
            model=model, messages=messages[1:], system=SYS_MSG['content'],
            temperature=temperature, max_tokens=4096
        )
        codebook = response.content[0].text
        usage = response.usage.output_tokens if response.usage else 0
    elif type(client) is openai.OpenAI:
        response = client.chat.completions.create(
            model=model, messages=messages,
            temperature=temperature, max_tokens=4096
        )
        codebook = response.choices[0].message.content
        usage = response.usage.completion_tokens if response.usage else 0
    else:
        raise ValueError("Invalid client type")
    codebook = codebook if codebook else ""

    metadata: CodebookMetadata = {
        "item_description": item_description,
        "label_description": label_description,
        "scale_details": scale_details,
        "format_details": format_details,
        "model": model,
        "source": "combine_codebooks",
        "generation": generation,
        "protocol": "merge",
        "temperature": temperature,
        "parents": [cb.name for cb in codebooks],
        "tokens": usage,
        "timestamp": pd.Timestamp.now().isoformat(),
        # "training_examples": [list(x) for x in zip(data, truth)],
        "training_example_count": len(labeled_examples) if labeled_examples else 0,
    }
    return Codebook(
        codebook=codebook, name=name if name else hashname(), metadata=metadata
    )


def codebook_prompt_snippets(
    labeled_examples: LabeledDataset,
    item_description: str | None,
    label_description: str | None,
    scale_details: str | None = None,
    format_details: str | None = None,
):
    snippets = dict(
        item_description="- Item description: a list of items",
        label_description="- Label description: human-judged labels for the items",
        scale_details="",
        example_str="",
        format_details="",
    )

    if item_description:
        snippets["item_description"] = "- Item description: " + item_description

    if label_description:
        snippets["label_description"] = f"\n- Label description: {label_description}"

    if scale_details:
        snippets["scale_details"] = f"\n- Scale details: {scale_details}"

    if labeled_examples:
        snippets["example_str"] = "\n".join(
            [f" - `{item}`: {label}" for item, label in labeled_examples]
        )

    if format_details:
        snippets["format_details"] = f"{format_details}"

    return snippets


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
    client: openai.OpenAI | anthropic.Anthropic = openai.OpenAI(),
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

    snippets = codebook_prompt_snippets(
        list(zip(list(data), list(truth))),
        item_description,
        label_description,
        scale_details,
        format_details,
    )
    prompt_templates = dict()
    prompt_templates[
        "direct"
    ] = """
    Below is a set of human-judged `item,label` pairs. Your goal is to write a codebook for future human judges to use to label new items.

    {item_description}{label_description}{scale_details}

    Read the examples and write a detailed codebook outlining how to judge the responses.

    # EXAMPLES

    {example_str}

    # TASK

    Read all the examples and write a finely-detailed codebook describing how to label the items.

    {format_details}
    """

    prompt_templates[
        "direct-bullet"
    ] = """
    Below is a set of human-judged `item,label` pairs. Your goal is to write a codebook for future human judges to use to label new items.

    {item_description}{label_description}{scale_details}

    Read the examples and write a BULLET POINT draft of a detailed codebook outlining how to judge the responses.

    # EXAMPLES

    {example_str}

    # TASK

    Read all the examples and write a finely-detailed codebook describing how to label the items, STRUCTURED ENTIRELY AS BULLET POINTS.

    {format_details}
    """

    prompt_templates["CoT"] = (
        prompt_templates["direct"]
        + """

    # INSTRUCTIONS

    Take a deep breath and work on this problem step-by-step. After working through the steps aloud, start the final codebook by using the tag <- CODEBOOK START ->
    """
    )
    prompt_template = textwrap.dedent(prompt_templates[protocol])
    prompt = prompt_template.format(
        item_description=snippets["item_description"],
        label_description=snippets["label_description"],
        scale_details=snippets["scale_details"],
        example_str=snippets["example_str"],
        format_details=snippets["format_details"],
    ).strip()

    SYS_MSG = {
        "role": "system",
        "content": "Please write a codebook for the following examples.",
    }
    messages = [SYS_MSG, {"role": "user", "content": prompt}]
    if type(client) is anthropic.Anthropic:
        response = client.messages.create(
            model=model, messages=messages[1:], system=SYS_MSG['content'],
            temperature=temperature, max_tokens=4096
        )
        codebook = response.content[0].text
        usage = response.usage.output_tokens if response.usage else 0
    elif type(client) is openai.OpenAI:
        response = client.chat.completions.create(
            model=model, messages=messages,
            temperature=temperature, max_tokens=4096
        )
        codebook = response.choices[0].message.content
        usage = response.usage.completion_tokens if response.usage else 0
    else:
        raise ValueError("Invalid client type")

    codebook = codebook if codebook else ""
    if protocol == "CoT" and "<- CODEBOOK START ->" in codebook:
        codebook = codebook.split("<- CODEBOOK START ->")[1]
    if "<- CODEBOOK":  # if it added an 'end' tag, remove it
        codebook = codebook.split("<- CODEBOOK")[0]

    metadata: CodebookMetadata = {
        "item_description": item_description,
        "label_description": label_description,
        "scale_details": scale_details,
        "format_details": format_details,
        "model": model,
        "source": "write_codebook",
        "protocol": protocol,
        "temperature": temperature,
        "parents": [],
        "generation": 1,
        "tokens": usage,
        "timestamp": pd.Timestamp.now().isoformat(),
        # "training_examples": [list(x) for x in zip(data, truth)],
        "training_example_count": len(data),
    }
    return Codebook(
        codebook=codebook, name=name if name else hashname(), metadata=metadata
    )


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
        classresults: ClassificationEvalResults = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        return classresults
    elif type == "regression":
        # calculate regression evaluation metrics (RMSE, pearsonr)
        trutharr = np.array(truth).astype("float64")
        predictedarr = np.array(predicted).astype("float64")
        RMSE = np.sqrt(np.mean((trutharr - predictedarr) ** 2))
        pearsonr = np.corrcoef(trutharr, predictedarr)[0, 1]
        regresults: RegressionEvalResults = {"RMSE": RMSE, "pearsonr": pearsonr}
        return regresults
    else:
        raise ValueError(
            "Invalid evaluation type. Must be either "
            "'classification' or 'regression'."
        )
