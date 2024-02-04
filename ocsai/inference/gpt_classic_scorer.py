from ..train import GPT_Classic_Prompter
from .gpt_base_scorer import GPT_Base_Scorer


GPTCLASSICMODELS = {
    'ocsai-babbage2': 'ft:babbage-002:peter-organisciak:gt-main2-epochs2:8fDSqNqU',
    'ocsai-davinci2': 'ft:davinci-002:peter-organisciak:gt-main2-epochs2:8fE6TmoV'
}

class GPT_Classic_Scorer(GPT_Base_Scorer):
    def __init__(self, *args, **kwargs):
        if 'model_dict' not in kwargs or not kwargs['model_dict']:
            kwargs['model_dict'] = GPTCLASSICMODELS
        if 'prompter' not in kwargs or not kwargs['prompter']:
            kwargs['prompter'] = GPT_Classic_Prompter()
        super().__init__(*args, **kwargs)

    def _score_gpt(self, gptprompt, model='first', just_final=False):
        # gptprompt is the templated item+response. Use _craft_gptprompt. It can be a list of prompts.
        if model == 'first':
            model = self.models[0]
        response = self.client.completions.create(
            model=self._models[model],
            prompt=gptprompt,
            temperature=0,
            n=1,
            logprobs=None,
            stop='\n',
            # since the prompter knows the format, use it's recommendation
            # for max tokens
            max_tokens=self.prompter.max_tokens
        )
        if just_final:
            return [choice.text for choice in response.choices]
        else:
            return response
