import openai
from ..types import LogProbPair, StandardAIResponse, UsageStats
from .llm_base_interface import LLM_Base_Interface


class OpenAILegacyInterface(LLM_Base_Interface):
    name = "openai_legacy"
    style = "completion"

    def _extract_choices(self, response) -> list:
        return response.choices

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
            if tokens[0].strip() == "":
                tokens = tokens[1:]
                toplogprobs = toplogprobs[1:]
            if len(tokens) > 1:
                self.logger.warn(
                    "Only one token expected, after stripping whitespace. Just using first token"
                )
        score_logprobs = list(toplogprobs[0].items())
        return score_logprobs

    def _check_prompt(self,
                      prompt: str | None,
                      messages: list[dict] | None) -> str:
        if (messages and prompt):
            raise ValueError("Only one of prompt or messages should be provided. Usually prompt.")

        if messages:
            return [m for m in messages if m["role"] == "user"][0]['content']
        else:
            if not prompt:
                raise ValueError("Prompt is required.")
            return prompt

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

        final_prompt = self._check_prompt(prompt, messages)

        response = await async_client.completions.create(
            model=model,
            prompt=final_prompt,
            temperature=temperature,
            n=1,
            logprobs=logprobs,
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
        sys_msg_text: None = None,
        temperature: float = 0,
        logprobs: int | None = None,
        stop_char: str | None = None,
        max_tokens: int | None = None,
    ) -> list[StandardAIResponse]:
        final_prompt = self._check_prompt(prompt, messages)

        response = client.completions.create(
            model=model,
            prompt=final_prompt,
            temperature=temperature,
            n=1,
            logprobs=logprobs,
            stop=stop_char,
            max_tokens=max_tokens,
        )

        return self.standardize_response(response)
