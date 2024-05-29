from ..types import LogProbPair, UsageStats
from .llm_base_interface import LLM_Base_Interface


class AnthropicInterface(LLM_Base_Interface):
    name = "anthropic"
    style = "chat"

    def _extract_choices(self, response) -> list:
        return response.content

    def _extract_content(self, choice) -> str:
        """Extract the content string from a response choice."""
        return choice.text

    def _extract_usage(self, response, divide_by: int = 1) -> UsageStats:
        """Extract usage statistics from a response."""
        input = response.usage.input_tokens
        output = response.usage.output_tokens
        return {
            "total": (input+output) / divide_by,
            "prompt": input / divide_by,
            "completion": output / divide_by,
        }

    def _extract_token_logprobs(self, choice) -> list[LogProbPair] | None:
        raise NotImplementedError
