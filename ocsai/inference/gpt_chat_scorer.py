from .gpt_base_scorer import GPT_Base_Scorer
from ..train import prepare_prompt, GPT_SYS_MSG
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
        super().__init__(*args, **kwargs)

    def _craft_gptprompt(self, item, response, question=None, task_type='uses',
                         language='eng'):
        # prompt templates take up to four args - item, response, task_type, and language
        # prompt templates should take 2 args - item and response
        prompt = prepare_prompt(item, response, task_type=task_type, language=language,
                                question=question)
        return prompt

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

        for prompt in tqdm(gptprompt):
            # May need to eventually switch
            # to multi-threading - a headache in Python - or TypeScript'
            self.logger.debug("Chat completions don't support batching, btw - so this is"
                              "considerably slower than the classic API.")
    
            messages = [
                    GPT_SYS_MSG, {"role": "user", "content": prompt}
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
                #stop='\n', STOP on newline only if aimng for score only
                max_tokens=25
            )
            all_responses.append(response)

        if not just_final:
            return all_responses
        else:
            content = [response.choices[0].message.content for response in all_responses]
            return content

        # results = parse_ocsai_response(content.strip())
        # results.update({
        #    'prompt_tokens': response.usage.prompt_tokens,
        #    'completion_tokens': response.usage.completion_tokens,
        # })
        # return results

    def _parse_response(self, response):
        '''
        Parse the response from the OCSAI dataset into a score, confidence, and flags
        '''
        score, confidence, flags = None, None, None
        score = response.split('SCORE:')[1].split('\n')[0]
        if score == 'null':
            score = None
        else:
            score = float(score)

        if 'CONFIDENCE: ' in response:
            confidence = response.split('CONFIDENCE: ')[1].split('\n')[0]
            confidence = int(confidence)

        if 'FLAGS: ' in response:
            flags = response.split('FLAGS: ')[1].split('\n')[0].split(',')
            flags = [f.strip() for f in flags]

        return dict(score=pd.to_numeric(score, errors='ignore'),
                    confidence=pd.to_numeric(confidence, errors='ignore'),
                    flags=flags)
        # return score  # currently the base class doesn't support confidence or flags