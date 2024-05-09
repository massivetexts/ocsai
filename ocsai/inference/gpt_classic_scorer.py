from ..train import GPT_Classic_Prompter
from .gpt_base_scorer import GPT_Base_Scorer


GPTCLASSICMODELS = {
    "ocsai-babbage2": "ft:babbage-002:peter-organisciak:gt-main2-epochs2:8fDSqNqU",
    "ocsai-davinci2": "ft:davinci-002:peter-organisciak:gt-main2-epochs2:8fE6TmoV",
}


class GPT_Classic_Scorer(GPT_Base_Scorer):

    DEFAULT_PROMPTER = GPT_Classic_Prompter

    def __init__(self, *args, **kwargs):
        if "model_dict" not in kwargs or not kwargs["model_dict"]:
            kwargs["model_dict"] = GPTCLASSICMODELS
        if "prompter" not in kwargs or not kwargs["prompter"]:
            kwargs["prompter"] = self.DEFAULT_PROMPTER()
        super().__init__(*args, **kwargs)

    def _score_gpt(self, gptprompt: str | list[str],
                   model: str = "first",
                   top_probs: int = 0,
                   raw: bool = False):
        '''
        Score the prompt using the GPT client.

        gptprompt is the templated item+response.
            Use _craft_gptprompt. It can be a list of prompts.

        top_probs is the number of top options to return. If 0, don't return
            the log probabilities. If 1, get log probabilities for just one option.
        '''
        if model == "first":
            model = self.models[0]

        logprobs: int | None = None
        if top_probs > 0:
            logprobs = top_probs

        response = self.client.completions.create(
            model=self._models[model],
            prompt=gptprompt,
            temperature=0,
            n=1,
            logprobs=logprobs,
            stop="\n",
            # since the prompter knows the format, use its recommendation
            # for max tokens
            max_tokens=self.prompter.max_tokens,
        )

        if raw:
            return response
        else:
            return self.prompter.standardize_response(response)
