from .gpt_base_scorer import GPT_Base_Scorer
from ..train import GPT_Ocsai2_Prompter
from tqdm.auto import tqdm
# from tqdm.asyncio import tqdm_asyncio
import asyncio

GPTCHATMODELS = {
    "ocsai_1.5_small": "ft:gpt-3.5-turbo-1106:peter-organisciak:ocsai-small-1-24:8cmALwWt",
    "ocsai_1.5_full": "ft:gpt-3.5-turbo-1106:peter-organisciak:ocsai-full-1-24:8d5RLryO",
}


class GPT_Chat_Scorer(GPT_Base_Scorer):

    DEFAULT_PROMPTER = GPT_Ocsai2_Prompter

    def __init__(self, *args, **kwargs):
        if "model_dict" not in kwargs or not kwargs["model_dict"]:
            kwargs["model_dict"] = GPTCHATMODELS
        if "prompter" not in kwargs or not kwargs["prompter"]:
            kwargs["prompter"] = self.DEFAULT_PROMPTER()
        super().__init__(*args, **kwargs)

    async def _score_gpt_async(self, gptprompt, model="first", top_probs: int = 0, raw: bool = False):
        if model == "first":
            model = self.models[0]

        if type(gptprompt) is str:
            gptprompt = [gptprompt]
        if len(gptprompt) > 20:
            raise ValueError(
                "This method is not designed for large batches. Use the classic API."
            )

        SYS_MSG = {"role": "system", "content": self.prompter.sys_msg_text}

        async def process_prompt(prompt):
            async with self.async_semaphore:
                messages = [SYS_MSG, {"role": "user", "content": prompt}]
                response = await self.async_client.chat.completions.create(
                    model=self._models[model],
                    messages=messages,
                    temperature=0,
                    n=1,
                    logprobs=None,
                    max_tokens=self.prompter.max_tokens,
                )
                return response

        all_responses = [process_prompt(prompt) for prompt in gptprompt]
        final_responses = await asyncio.gather(*all_responses)
        if raw:
            return final_responses
        else:
            content = [
                response.choices[0].message.content
                for response in await final_responses
            ]
            return content

    def _score_gpt(
        self,
        gptprompt: str | list[str],
        model: str = "first",
        top_probs: int = 0,
        runasync=False,
    ):
        """
        gptprompt is the templated item+response+type+language. Use _craft_gptprompt.

        If a string is provided, returns a single response. If a list is provided,
        returns a list of responses in the same order.

        Returns:
        - An OpenAI completion object if raw=True
        - A list of strings if raw=False and top_probs=0
        - A list of (string, ProbScores) tuples if raw=False and top_probs>0
        """
        if runasync:
            raise NotImplementedError("This method is not yet complete.")
            all_responses = asyncio.run(
                self._score_gpt_async(gptprompt, model, raw=True)
            )

            content = [
                response.choices[0].message.content for response in all_responses
            ]
            return content

        if model == "first":
            model = self.models[0]
        all_responses = []
        if type(gptprompt) is str:
            gptprompt = [gptprompt]

        logprobs: int | None = None
        if top_probs > 0:
            assert top_probs <= 20, "OpenAI API only supports 20 logprobs at a time."
            logprobs = top_probs

        SYS_MSG = {"role": "system", "content": self.prompter.sys_msg_text}

        for prompt in tqdm(gptprompt):
            # May need to eventually switch
            # to multi-threading - a headache in Python - or TypeScript'
            self.logger.debug(
                "Chat completions don't support batching, btw - so this is"
                "considerably slower than the classic API."
            )

            messages = [SYS_MSG, {"role": "user", "content": prompt}]

            response = self.client.chat.completions.create(
                model=self._models[model],
                messages=messages,
                temperature=0,
                n=1,
                logprobs=bool(logprobs),
                top_logprobs=logprobs,
                # Expected token counts:
                # Just score is 6 tokens;
                # score+confidence is 13 tokens;
                # score+confidence+flags is still unknown
                # stop='\n', STOP on newline only if aiming for score only
                max_tokens=self.prompter.max_tokens,
            )
            all_responses.append(response)

        all_standardized = []
        for response in all_responses:
            standard_response = self.prompter.standardize_response(response)
            assert len(standard_response) == 1, "Only one response expected for chat model"
            all_standardized += standard_response
        return all_standardized
