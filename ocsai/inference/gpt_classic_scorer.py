from .gpt_base_scorer import GPT_Base_Scorer


GPTCLASSICMODELS = {
    'ocsai-babbage2': 'ft:babbage-002:peter-organisciak:gt-main2-epochs2:8fDSqNqU',
    'ocsai-davinci2': 'ft:davinci-002:peter-organisciak:gt-main2-epochs2:8fE6TmoV'
}

class GPT_Classic_Scorer(GPT_Base_Scorer):
    def __init__(self, *args, **kwargs):
        if 'model_dict' not in kwargs or not kwargs['model_dict']:
            kwargs['model_dict'] = GPTCLASSICMODELS
        super().__init__(*args, **kwargs)

    def _craft_gptprompt(self, item, response, task_type='uses', question=None, language='eng'):
        # prompt templates should take 2 args - item and response
        if task_type == 'uses':
            prompt_template = "AUT Prompt:{}\nResponse:{}\nScore:\n"
        else:
            self.logger.warning("Only 'uses' task type is supported with Classic Scorer")
        
        if question:
            self.logger.warning("Question is not supported with Classic Scorer")
        if language != 'eng':
            self.logger.warning("Only 'eng' language is supported with Classic Scorer")
        
        # This is format of trained models in Organisciak, Acar, Dumas, and Berthiaume
        return prompt_template.format(item, response)

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
            max_tokens=2
        )
        if just_final:
            return [choice.text for choice in response.choices]
        else:
            return response

    def _parse_response(self, score_raw):
        score = int(score_raw) / 10
        return dict(score=score, confidence=None, flags=None)
