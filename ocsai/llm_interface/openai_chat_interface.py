from ..types import LogProbPair, UsageStats
from .llm_base_interface import LLM_Base_Interface


class OpenAIChatInterface(LLM_Base_Interface):
    name = "openai_chat"
    style = "chat"

    def _extract_choices(self, response) -> list:
        return response.choices

    def _extract_content(self, choice) -> str:
        """Extract the content string from a response choice."""
        return choice.message.content

    def _extract_usage(self, response, divide_by: int = 1) -> UsageStats:
        """Extract usage statistics from a response."""
        return {
            "total": response.usage.total_tokens / divide_by,
            "prompt": response.usage.prompt_tokens / divide_by,
            "completion": response.usage.completion_tokens / divide_by,
        }

    def _extract_token_logprobs(self, choice) -> list[LogProbPair] | None:
        '''Extract the token log probabilities from a response.'''
        # FYI: Chat models, even with temperature=0, exhibit more randomness "
        # than classic models.
        if not hasattr(choice, "logprobs"):
            return None
        elif choice.logprobs is None:
            return None
        score_logprobs = [(x.token, x.logprob)
                          for x in choice.logprobs.content[0].top_logprobs]
        return score_logprobs
