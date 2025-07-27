from .base_prompter import Base_Prompter
from ..types import FullScore, ResponseTypes
import random
import pandas as pd
from typing import Literal
from .ocsai1p5_prompter import TrainProbs

class Ocsai2_Prompter(Base_Prompter):
    """The new format, introduced with Ocsai 1.5."""

    sys_msg_text = "You are a creativity judge, scoring tests of originality."
    stop_char = '\n'
    max_tokens = 1

    train_probs: TrainProbs = dict(
        action_exclude_prob=0.75,
        task_type_exclude_prob=0.3,
        prompt_exclude_prob=0,
        language_exclude_prob=0.5,
        question_exclude_prob=0.5,
        detail_exclude_prob=0.8,
    )

    def craft_prompt(
        self,
        item: str,
        response: str,
        task_type: str | None = None,
        question: str | None = None,
        language: str | None = None,
        seed=None,
        action_exclude_prob: float = 0,
        task_type_exclude_prob: float = 0,
        prompt_exclude_prob: float = 0,
        language_exclude_prob: float = 0,
        question_exclude_prob: float = 0,
        detail_exclude_prob: float = 1,
    ) -> str:
        # Initialize the random number generator with the provided seed
        if seed is not None:
            random.seed(seed)

        if not question and not task_type:
            self.logger.warning("No task_type or question provided. Assuming task_type='uses'")
            task_type = "uses"
        if not question:
            question_exclude_prob = 1
        if not language:
            language_exclude_prob = 1
        if not task_type:
            task_type_exclude_prob = 1
        if not question:
            question_exclude_prob = 1

        components = {
            "ACTION": (
                "ACTION: TAG THE ORIGINALITY OF A RESPONSE TO A CREATIVITY TEST.",
                action_exclude_prob,
            ),
            "TASK TYPE": (f"TASK: {task_type}", task_type_exclude_prob),
            "PROMPT": (f"PROMPT: {item}", prompt_exclude_prob),
            "TASK QUESTION": (f"TASK QUESTION: {question}", question_exclude_prob),
            "LANGUAGE": (f"LANGUAGE: {language}", language_exclude_prob),
            "RESPONSE": f"RESPONSE: `{response}`",
            "DETAILS": (
                (
                    "## Details\n"
                    "SCALE: 10-50, where 10 is `not original at all` and 50 is `extremely original`\n"
                    "FORMAT: Return a single token, with an originality score, 10-50, and no other text."
                ),
                detail_exclude_prob,
            ),
        }

        prompt_text = ""
        for key, value in components.items():
            if key in ["SCALE", "FORMAT"]:
                continue
            if key == "RESPONSE":
                prompt_text += value + "\n\n"
            else:
                if random.random() > value[1]:
                    prompt_text += value[0] + "\n"
                elif key == "TASK QUESTION":
                    # include anyway if task type or prompt were removed
                    if ("TASK: " not in prompt_text) or ("PROMPT: " not in prompt_text):
                        prompt_text += value[0] + "\n"
                else:
                    pass

        return prompt_text.strip()

    def craft_response(self, score: float, confidence=None, flags=None):
        """
        Just a number
        """
        if confidence is not None:
            self.logger.warning("Confidence is not supported with Classic Prompter")

        if flags is not None:
            self.logger.warning("Flags are not supported with Classic Prompter")

        return f"{int(score*10)}"

    def parse_content(self, content: str, type: ResponseTypes = "other") -> FullScore:
        score = int(content.strip()) / 10
        parsed: FullScore = {"score": score, "confidence": None, "flags": None, "n": 1, "type": type}
        return parsed

    def prepare_training_prompt(
        self, item, response, task_type, question, language, seed=None
    ):
        """Opinionated probabilities of different parts of the prompt being included"""
        return self.craft_prompt(
            item, response, task_type, question, language, seed, **self.train_probs
        )

    def prepare_example(
        self,
        item,
        response,
        task_type="uses",
        question=None,
        language=None,
        target=None,
        confidence:None=None,
        seed=None,
        action_exclude_prob:float=0,
        task_type_exclude_prob:float=0,
        prompt_exclude_prob:float=0,
        language_exclude_prob:float=0,
        question_exclude_prob:float=0,
        detail_exclude_prob:float=0
    ):
        prompt = self.craft_prompt(
            item,
            response,
            task_type,
            question,
            language,
            seed,
            action_exclude_prob=action_exclude_prob,
            task_type_exclude_prob=task_type_exclude_prob,
            prompt_exclude_prob=prompt_exclude_prob,
            language_exclude_prob=language_exclude_prob,
            question_exclude_prob=question_exclude_prob,
            detail_exclude_prob=detail_exclude_prob
        )
        msgs = [
            {"role": "system", "content": self.sys_msg_text},
            {"role": "user", "content": prompt},
        ]
        # Add the response
        if target:
            ast_msg = {
                "role": "assistant",
                "content": self.craft_response(target, None),
            }
            msgs.append(ast_msg)
        return dict(messages=msgs)


    def prepare_example_from_series(self,
                                    row: pd.Series,
                                    train_probs: TrainProbs | Literal['default'] = 'default',
                                    seed: int | None = None):
        """Parse a row of a DataFrame, with the following columns:
        prompt, response, type (or task_type), question, language, target
        """
        row = row.rename(index={"type": "task_type", "prompt": "item"})

        if train_probs == 'default':
            train_probs = self.train_probs

        # prompt, response, type (or task_type), question, language, target
        include_params = [
            "item",
            "response",
            "task_type",
            "question",
            "language",
            "target",
        ]
        kwargs = row[[p for p in include_params if p in row.index]].to_dict()
        return self.prepare_example(**kwargs, **train_probs, seed=seed)
