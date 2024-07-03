from typing import Awaitable
from ..types import LogProbPair, StandardAIResponse, UsageStats
from .llm_base_interface import LLM_Base_Interface
import openai


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
        """Extract the token log probabilities from a response."""
        # FYI: Chat models, even with temperature=0, exhibit more randomness "
        # than classic models.
        if not hasattr(choice, "logprobs"):
            return None
        elif choice.logprobs is None:
            return None
        score_logprobs = [
            # TODO fix this - it is broken for >1 token. See un-refactored code in Ocsai2_Prompter._extract_token_logprobs
            (x.token, x.logprob) for x in choice.logprobs.content[0].top_logprobs
        ]
        return score_logprobs

    def _contruct_messages(
        self, prompt: str | None, messages: list[dict] | None, sys_msg_text: str | None
    ) -> list[dict]:
        final_messages: list[dict] = []

        if not prompt and not messages:
            raise ValueError("Either prompt or messages must be provided.")

        if sys_msg_text:
            sys_msg = {"role": "system", "content": sys_msg_text}
            final_messages += [sys_msg]

        if messages:
            if len(messages) == 0:
                raise ValueError("Messages cannot be empty")
            if prompt:
                raise ValueError("Only one of prompt or messages should be provided.")

            if sys_msg_text and messages[0]["role"] == "system":
                final_messages += messages[:1]
            else:
                final_messages += messages
        elif prompt:
            final_messages += [{"role": "user", "content": prompt}]
        return final_messages

    async def completion_async(
        self,
        async_client: openai.AsyncOpenAI,
        model: str,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        sys_msg_text: str | None = None,
        temperature: float = 0,
        logprobs: int | None = None,
        stop_char: str | None = None,
        max_tokens: int | None = None,
    ) -> list[StandardAIResponse]:
        '''Completion for async client. Returns a list of StandardAIResponses - one for each choice (usually just one).'''

        final_messages: list[dict] = self._contruct_messages(
            prompt, messages, sys_msg_text
        )

        response = await async_client.chat.completions.create(
            model=model,
            messages=final_messages,
            temperature=temperature,
            n=1,
            logprobs=bool(logprobs),
            top_logprobs=logprobs,
            stop=stop_char,
            max_tokens=max_tokens,
        )
        return self.standardize_response(response)

    def completion(
        self,
        client: openai.OpenAI,
        model: str,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        sys_msg_text: str | None = None,
        temperature: float = 0,
        logprobs: int | None = None,
        stop_char: str | None = None,
        max_tokens: int | None = None,
    ) -> list[StandardAIResponse]:
        final_messages: list[dict] = self._contruct_messages(
            prompt, messages, sys_msg_text
        )

        response = client.chat.completions.create(
            model=model,
            messages=final_messages,
            temperature=temperature,
            n=1,
            logprobs=bool(logprobs),
            top_logprobs=logprobs,
            stop=stop_char,
            max_tokens=max_tokens,
        )

        return self.standardize_response(response)
