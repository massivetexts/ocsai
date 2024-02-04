from .llm_base_prompter import LLM_Base_Prompter


class GPT_Classic_Prompter(LLM_Base_Prompter):
    ''' The format used in the original LLM paper, Organisciak et al. 2023'''
    max_tokens: int = 2

    def craft_prompt(self, item, response, task_type='uses', question=None, language='eng'):
        # prompt templates should take 2 args - item and response
        if task_type == 'uses':
            prompt_template = "AUT Prompt:{}\nResponse:{}\nScore:\n"
        else:
            self.logger.warning("Only 'uses' task type is supported with Classic Prompter")

        if question:
            self.logger.warning("Question is not supported with Classic Prompter")
        if language is not None and language != 'eng':
            self.logger.warning("Only 'eng' language is supported with Classic Prompter")

        # This is format of trained models in Organisciak, Acar, Dumas, and Berthiaume
        return prompt_template.format(item, response)

    def craft_response(self, score, confidence=None, flags=None):
        '''
        Just a number
        '''
        if confidence is not None:
            self.logger.warning("Confidence is not supported with Classic Prompter")

        if flags is not None:
            self.logger.warning("Flags are not supported with Classic Prompter")

        return f'{int(score*10)}'

    def parse_response(self, score_raw):
        score = int(score_raw) / 10
        return dict(score=score, confidence=None, flags=None)

    def prepare_example(self, item, response, task_type='uses', question=None,
                        language=None, target=None, confidence=None, seed=None):
        ''' Example of format:

        ```
        {"prompt":"AUT Prompt:brick\nResponse:use as a stepping stool to get up higher\nScore:\n",
        "completion":"17"}
        ```
        '''
        return {"prompt": self.craft_prompt(item, response, task_type, question, language),
                "completion": self.craft_response(target, confidence)}
