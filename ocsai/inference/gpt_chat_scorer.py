from .gpt_base_scorer import GPT_Base_Scorer
from ..train.prompts import GPT_Classic_Chat_Prompter
from tqdm.auto import tqdm
import pandas as pd

GPTCHATMODELS = {
    "ocsai_1.5_small": "ft:gpt-3.5-turbo-1106:peter-organisciak:ocsai-small-1-24:8cmALwWt",
    "ocsai_1.5_full": "ft:gpt-3.5-turbo-1106:peter-organisciak:ocsai-full-1-24:8d5RLryO"
}


class GPT_Chat_Scorer(GPT_Base_Scorer):

    def __init__(self, *args, **kwargs):
        if 'model_dict' not in kwargs or not kwargs['model_dict']:
            kwargs['model_dict'] = GPTCHATMODELS
        if 'prompter' not in kwargs or not kwargs['prompter']:
            kwargs['prompter'] = GPT_Classic_Chat_Prompter()
        super().__init__(*args, **kwargs)

    def _score_gpt(self, gptprompt, model='first', just_final=True):
        '''
        gptprompt is the templated item+response+type+language. Use _craft_gptprompt.

        If a string is provided, returns a single response. If a list is provided,
        returns a list of responses in the same order.
        '''
        if model == 'first':
            model = self.models[0]
        all_responses = []
        if type(gptprompt) is str:
            gptprompt = [gptprompt]

        SYS_MSG = {
                "role": "system",
                "content": self.prompter.sys_msg_text
        }
        
        for prompt in tqdm(gptprompt):
            # May need to eventually switch
            # to multi-threading - a headache in Python - or TypeScript'
            self.logger.debug("Chat completions don't support batching, btw - so this is"
                              "considerably slower than the classic API.")
    
            messages = [
                    SYS_MSG, {"role": "user", "content": prompt}
            ]

            response = self.client.chat.completions.create(
                model=self._models[model],
                messages=messages,
                temperature=0,
                n=1,
                logprobs=None,
                # Expected token counts:
                # Just score is 6 tokens;
                # score+confidence is 13 tokens;
                # score+confidence+flags is still unknown
                #stop='\n', STOP on newline only if aiming for score only
                max_tokens=self.prompter.max_tokens
            )
            all_responses.append(response)

        if not just_final:
            return all_responses
        else:
            content = [response.choices[0].message.content for response in all_responses]
            return content
