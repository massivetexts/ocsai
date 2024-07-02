from typing import Awaitable
from ocsai.types import StandardAIResponse
from .base_scorer import Base_Scorer
from ..prompter import Ocsai2_Prompter
from tqdm.auto import tqdm
# from tqdm.asyncio import tqdm_asyncio
import asyncio
from ..llm_interface import OpenAIChatInterface
import openai

GPTCHATMODELS = {
    "ocsai_1.5_full": "ft:gpt-3.5-turbo-1106:peter-organisciak:ocsai-full-1-24:8d5RLryO",
    "ocsai_1.5_small": "ft:gpt-3.5-turbo-1106:peter-organisciak:ocsai-small-1-24:8cmALwWt",
}


class Chat_Scorer(Base_Scorer):

    DEFAULT_PROMPTER = Ocsai2_Prompter
    llm_interface = OpenAIChatInterface()

    def __init__(self, *args, **kwargs):
        if "model_dict" not in kwargs or not kwargs["model_dict"]:
            kwargs["model_dict"] = GPTCHATMODELS
        if "prompter" not in kwargs or not kwargs["prompter"]:
            kwargs["prompter"] = self.DEFAULT_PROMPTER()
        super().__init__(*args, **kwargs)

        self.async_client = openai.AsyncOpenAI(api_key=openai.api_key)
        max_async_processes = 10
        self.async_semaphore = asyncio.Semaphore(max_async_processes)

    async def _score_llm_async(
        self,
        gptprompt: str | list[str],
        model_id: str,
        top_probs: int = 0,
    ) -> list[list[StandardAIResponse]]:
        gptprompt = [gptprompt] if isinstance(gptprompt, str) else gptprompt
        logprobs = self._verify_logprobs(top_probs)

        async with self.async_semaphore:
            tasks = []
            for prompt in gptprompt:
                tasks.append(
                    self.llm_interface.completion_async(
                        async_client=self.async_client,
                        model=model_id,
                        prompt=prompt,
                        temperature=0,
                        logprobs=logprobs,
                        stop_char=self.prompter.stop_char,  # STOP on newline only if aiming for score only
                        max_tokens=self.prompter.max_tokens,
                    )
                )
            all_responses = await asyncio.gather(*tasks)

        return all_responses

    def _verify_logprobs(self, top_probs: int):
        if top_probs > 0:
            max_probs = 20
            if top_probs > max_probs:
                self.logger.warning(
                    f"OpenAI API only supports {max_probs} logprobs at a time. Forcing top_probs={max_probs}."
                )
                top_probs = max_probs
            return top_probs
        return None

    def _score_llm(
        self,
        gptprompt: str | list[str],
        model_id: str,
        top_probs: int = 0
    ) -> list[list[StandardAIResponse]]:
        """
        gptprompt is the templated item+response+type+language. Use _craft_gptprompt.

        If a string is provided, returns a single response. If a list is provided,
        returns a list of responses in the same order.

        Returns:
        - An OpenAI completion object if raw=True
        - A list of strings if raw=False and top_probs=0
        - A list of (string, ProbScores) tuples if raw=False and top_probs>0
        """
        gptprompt = [gptprompt] if isinstance(gptprompt, str) else gptprompt
        logprobs: int | None = self._verify_logprobs(top_probs)

        all_responses = []
        for prompt in (tqdm(gptprompt) if len(gptprompt) > 10 else gptprompt):
            self.logger.debug(
                "Chat completions don't support batching, btw - so this is"
                "considerably slower than the classic API."
            )
            response = self.llm_interface.completion(
                client=self.client,
                model=model_id,
                prompt=prompt,
                temperature=0,
                sys_msg_text=self.prompter.sys_msg_text,
                logprobs=logprobs,
                stop_char=self.prompter.stop_char,  # STOP on newline only if aiming for score only
                max_tokens=self.prompter.max_tokens,
            )
            all_responses.append(response)

        return all_responses
