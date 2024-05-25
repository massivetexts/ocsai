from .llm_base_prompter import LLM_Base_Prompter
from ..types import FullScore, LogProbPair, ResponseTypes, UsageStats


class GPT_Ocsai1_Prompter(LLM_Base_Prompter):
    """The format used in the original LLM paper, Organisciak et al. 2023"""

    max_tokens: int = 2

    def craft_prompt(
        self,
        item: str,
        response: str,
        task_type: str | None = "uses",
        question: str | None = None,
        language: str | None = "eng",
    ) -> str:
        # prompt templates should take 2 args - item and response
        if task_type != "uses":
            self.logger.warning(
                "Only 'uses' task type is supported with Classic Prompter"
            )
        # the trailing space is distracting to me, but it's how it was trained!
        prompt_template = "AUT Prompt:{}\nResponse:{}\nScore:\n"

        if question:
            raise ValueError("Question is not supported with Classic Prompter")

        if language is not None and language != "eng":
            self.logger.warning(
                "Only 'eng' language is supported with Classic Prompter"
            )

        # This is format of trained models in Organisciak, Acar, Dumas, and Berthiaume
        return prompt_template.format(item, response)

    def craft_response(self, score, confidence=None, flags=None):
        """
        Just a number
        """
        if confidence is not None:
            self.logger.warning("Confidence is not supported with Classic Prompter")

        if flags is not None:
            self.logger.warning("Flags are not supported with Classic Prompter")

        return f"{int(score*10)}"

    def parse_content(self, content: str, type: ResponseTypes = "other") -> FullScore:
        score = int(content) / 10
        return {"score": score, "confidence": None, "flags": None, "n": 1, "type": type}

    def prepare_example(
        self,
        item: str,
        response: str,
        task_type: str | None = "uses",
        question: str | None = None,
        language: str | None = None,
        target=None,
        confidence=None,
        seed=None,
    ):
        """Example of format:

        ```
        {"prompt":"AUT Prompt:brick\nResponse:use as a stepping stool to get up higher\nScore:\n",
        "completion":"17"}
        ```
        """
        return {
            "prompt": self.craft_prompt(item, response, task_type, question, language),
            "completion": self.craft_response(target, confidence),
        }