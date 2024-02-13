from typing import Optional
import logging


class LLM_Base_Prompter():
    sys_msg_text: Optional[str] = None
    max_tokens: int = 100

    def __init__(self, logger=None):
        self.logger = logger
        if not self.logger:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)

    def craft_prompt(self, item, response, task_type='uses', question=None, language='eng'):
        '''Craft a prompt for the language model, given an item, response, and task type'''
        raise NotImplementedError

    def craft_response(self, score, confidence=None, flags=None):
        '''Craft a response for the language model, given a score, confidence, and flags'''
        raise NotImplementedError

    def parse_response(self, score_raw):
        '''Parse the raw response from the language model into a dict of 
            {score, confidence, and flags}'''
        raise NotImplementedError

    def prepare_example(self, item, response, task_type='uses', question=None,
                        language=None, target=None, confidence=None, seed=None):
        '''Prepare an example for training. Not needed for inference.'''
        pass