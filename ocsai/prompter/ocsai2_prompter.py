from .base_prompter import Base_Prompter
from ..types import LogProbPair, FullScore, ResponseTypes
import random
import numpy as np
import re


class Ocsai2_Prompter(Base_Prompter):
    """The new format, introduced with Ocsai 1.5."""

    sys_msg_text = "You are a creativity judge, scoring tests of originality."

    train_probs = dict(
        action_exclude_prob=0.75,
        task_type_exclude_prob=0.3,
        prompt_exclude_prob=0,
        language_exclude_prob=0.5,
        question_exclude_prob=0.5,
        detail_exclude_prob=0.8,
        no_flags=True,
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
        detail_exclude_prob: float = 0,
        no_flags: bool = False,
    ) -> str:
        # Initialize the random number generator with the provided seed
        # no_flags excludes the FLAGS part of the prompt altogether
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
                    "SCALE: 1-5, where 1 is `not original at all` and 5 is `extremely original`\n"
                    "FORMAT: Return in the format of newline-separated `KEY:value` pairs, with the following fields:\n"
                    "- `SCORE`: An originality score, 1-5\n"
                    "- `CONFIDENCE`: A measure of confidence in the score, 1-3, or None.\n"
                    "- `FLAGS`: A comma-separated list with content flags, such as: 'nonsense', 'violent', "
                    "'not practical'"
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

        if no_flags:
            prompt_text = re.sub(r"\n- `FLAGS`: .*", "", prompt_text)

        return prompt_text.strip()

    def craft_response(self, score, confidence=None, flags=None):
        # cast score type: it an be a float or None (written as 'null' in the dataset)
        if score is None:
            score = "null"
        else:
            score = str(score)
        response = f"SCORE: {score}\n"

        if confidence and not np.isnan(confidence):
            response += f"CONFIDENCE: {confidence}\n"

        if flags:
            if isinstance(flags, list):
                flags = ",".join(flags)
            response += f"FLAGS: {flags}"
        return response.strip()

    def parse_content(self, content: str, type: ResponseTypes = "other"
                      ) -> FullScore:
        """
        Parse the response from the OCSAI dataset into a score, confidence, and flags
        """
        confidence: int | None = None
        flags: list[str] | None = None

        try:
            score_str: str = content.split("SCORE:")[1].split("\n")[0]
        except IndexError:
            score_str = "null"

        if score_str == "null":
            score = None
        else:
            score = float(score_str)

        if "CONFIDENCE: " in content:
            try:
                confidence_str: str = content.split("CONFIDENCE: ")[1].split("\n")[0]
                confidence = int(confidence_str)
            except IndexError:
                confidence = None

        if "FLAGS: " in content:
            try:
                flags_str: str = content.split("FLAGS: ")[1].split("\n")[0]
                flags = [f.strip() for f in flags_str.split(",")]
            except IndexError:
                flags = None

        parsed: FullScore = {
            "score": score,
            "confidence": confidence,
            "flags": flags,
            "n": 1,
            "type": type
        }
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
        confidence=None,
        seed=None,
        action_exclude_prob=0,
        task_type_exclude_prob=0,
        prompt_exclude_prob=0,
        language_exclude_prob=0,
        question_exclude_prob=0,
        detail_exclude_prob=0,
        no_flags=False,
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
            detail_exclude_prob=detail_exclude_prob,
            no_flags=no_flags,
        )
        msgs = [
            {"role": "system", "content": self.sys_msg_text},
            {"role": "user", "content": prompt},
        ]
        # Add the response
        if target:
            ast_msg = {
                "role": "assistant",
                "content": self.craft_response(target, confidence),
            }
            msgs.append(ast_msg)
        return msgs

    def _extract_token_logprobs(self, choice) -> list[LogProbPair] | None:
        """Extract the token log probabilities from a response."""
        # FYI: Chat models, even with temperature=0, exhibit more
        # randomness than classic models and logprobs are less stable here
        if choice.logprobs is None:
            return None

        # the content here is trickier to parse. It will be 'SCORE: x.y\n...'
        # - the first three tokens, can be skipped - for the digit before and
        # after the colon.
        whole_numbers: list[LogProbPair] = [(x.token, x.logprob) for x in choice.logprobs.content[3].top_logprobs]
        tenths: list[LogProbPair] = [(x.token, x.logprob) for x in choice.logprobs.content[5].top_logprobs]

        # calculate joint log probs
        # \log P(X \cap Y) = \log P(X) + \log P(Y)
        combined_log_probs = []
        for w, log_prob_w in whole_numbers:
            for t, log_prob_t in tenths:
                logprobpair: LogProbPair = (
                    f"SCORE: {w}.{t}\n",
                    log_prob_w + log_prob_t
                )
                combined_log_probs.append(logprobpair)

        # sort and trim just to original size * 2
        combined_log_probs = sorted(combined_log_probs, key=lambda x: x[1], reverse=True)[:len(whole_numbers)*2]
        return combined_log_probs

    def prepare_example_from_series(self, row, seed=None):
        """Parse a row of a DataFrame, with the following columns:
        prompt, response, type (or task_type), question, language, target, confidence
        """
        if ("task_type" not in row.index) and ("task" in row.index):
            row = row.rename(index={"type": "task_type"})

        # prompt, response, type (or task_type), question, language, target, confidence
        include_params = [
            "prompt",
            "response",
            "task_type",
            "question",
            "language",
            "target",
            "confidence",
        ]
        kwargs = row[[p for p in include_params if p in row.index]].to_dict()
        return self.prepare_example(**kwargs, seed=seed)
