import logging
from typing import Literal
from ..types import LogProbPair, StandardAIResponse, UsageStats


class LLM_Base_Interface:

    name = "generic"
    style: Literal["chat", "completion"]

    def __init__(self, logger=None):
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)

    def _extract_choices(self, response) -> list:
        """
        Some LLMs give multiple choices, this returns them as
        an iterable, for standardization. Default assumes the
        OpenAI format.
        """
        raise NotImplementedError

    def _extract_content(self, response) -> str:
        """
        Extract the content from the response. Default assumes the
        OpenAI format.
        """
        raise NotImplementedError

    def standardize_response(self, response) -> list[StandardAIResponse]:
        """Cast a response into the standard AI response format.
        E.g. anthropic or openai responses into a common format.
        
        Returns a list of responses, unpacking when there are multiple choices.
        """
        responses = []
        n_responses = len(response.choices)
        usage = self._extract_usage(response, divide_by=n_responses)

        choices = self._extract_choices(response)
        for choice in choices:
            content = self._extract_content(choice)
            logprobs = self._extract_token_logprobs(choice)

            current: StandardAIResponse = {
                "content": content,
                "logprobs": logprobs,
                "usage": usage,
            }
            responses.append(current)

        return responses

    def _extract_usage(self, response, divide_by=1) -> UsageStats:
        """Extract usage statistics from the response."""
        raise NotImplementedError

    def _extract_token_logprobs(self, choice) -> list[LogProbPair] | None:
        """Extract the token log probabilities from a response or response choice,
        returning in a standardized format."""
        if not hasattr(choice, "logprobs"):
            return None
        elif choice.logprobs is None:
            return None
        raise NotImplementedError

    async def completion_async(
        self,
        async_client,
        model: str,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        sys_msg_text: str | None = None,
        temperature: float = 0,
        logprobs: int | None = None,
        stop_char: str | None = None,
        max_tokens: int | None = None,
    ) -> list[StandardAIResponse]:
        raise NotImplementedError

    def completion(
        self,
        async_client,
        model: str,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        sys_msg_text: str | None = None,
        temperature: float = 0,
        logprobs: int | None = None,
        stop_char: str | None = None,
        max_tokens: int | None = None,
    ) -> list[StandardAIResponse]:
        raise NotImplementedError
