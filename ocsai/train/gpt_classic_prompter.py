from .llm_base_prompter import (
    FullScore,
    LLM_Base_Prompter,
    LogProbPair,
    ResponseTypes,
    UsageStats,
)


class GPT_Classic_Prompter(LLM_Base_Prompter):
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
        prompt_template = "AUT Prompt:{}\nResponse:{}\nScore:\n "

        if question:
            self.logger.warning("Question is not supported with Classic Prompter")
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

    def _extract_content(self, choice) -> str:
        """Extract the content string from a response choice."""
        if hasattr(choice, "text"):
            content = choice.text
        elif hasattr(choice, "logprobs") and choice.logprobs is not None:
            content = "".join(choice.logprobs.tokens)
        else:
            self.logger.error(choice)
            raise ValueError(
                "Response object does not have a 'content' or 'logprobs' attribute."
            )
        return content

    def _extract_usage(self, response, divide_by: int = 1) -> UsageStats:
        """Extract usage statistics from a response."""
        return {
            "total": response.usage.total_tokens / divide_by,
            "prompt": response.usage.prompt_tokens / divide_by,
            "completion": response.usage.completion_tokens / divide_by,
        }

    def _extract_token_logprobs(self, choice) -> list[LogProbPair] | None:
        """Extract the token log probabilities from a response choice
        If there are multiple choices, return a list of lists."""
        if not hasattr(choice, "logprobs") or choice.logprobs is None:
            return None
        tokens = choice.logprobs.tokens
        toplogprobs = choice.logprobs.top_logprobs
        if len(tokens) > 1:
            self.logger.warn("Only one token expected. Trying our best to parse anyway")
            if tokens[0].strip() == "":
                tokens = tokens[1:]
                toplogprobs = toplogprobs[1:]
        score_logprobs = list(toplogprobs[0].items())
        return score_logprobs

    def parse_content(
        self, content: str, type: ResponseTypes = "other"
    ) -> FullScore:
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
