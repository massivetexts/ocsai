import random
from typing import Optional
import numpy as np
import re
import pandas as pd
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


class GPT_Classic_Chat_Prompter(GPT_Classic_Prompter):
    ''' The format used in the original LLM paper, adjusted slightly to work with chat.'''
    sys_msg_text = "You score originality in the alternate uses task."
    max_tokens = 25

    def craft_prompt(self, item, response, task_type='uses', question=None, language='eng'):
        ''' Remove "\nScore:\n" from the legacy format.'''
        prompt = super().craft_prompt(item, response, task_type, question, language)
        return prompt.replace("\nScore:\n", "")

    def prepare_example(self, item, response, task_type='uses', question=None,
                        language=None, target=None, confidence=None, seed=None):
        prompt = self.craft_prompt(item, response, task_type, question, language)
        msgs = [
            {"role": "system", "content": self.sys_msg_text},
            {"role": "user", "content": prompt},
            ]
        # Add the response
        if target:
            ast_msg = {
                "role": "assistant",
                "content": self.craft_response(target, confidence)
            }
            msgs.append(ast_msg)
        return msgs

class GPT_Ocsai2_Prompter(LLM_Base_Prompter):
    ''' The new format, introduced with Ocsai 1.5.'''
    sys_msg_text = "You are a creativity judge, scoring tests of originality."

    def craft_prompt(self, item, response, task_type='uses', question=None, language='eng', seed=None,
                    action_exclude_prob=0, task_type_exclude_prob=0, prompt_exclude_prob=0,
                    language_exclude_prob=0, question_exclude_prob=0, detail_exclude_prob=0,
                    no_flags=False):
        # Initialize the random number generator with the provided seed
        # no_flags exluded the FLAGS part of the prompt altogether
        if seed is not None:
            random.seed(seed)

        if not question:
            question_exclude_prob = 1
        if not language:
            language_exclude_prob = 1

        components = {
            "ACTION": ("ACTION: TAG THE ORIGINALITY OF A RESPONSE TO A CREATIVITY TEST.",
                    action_exclude_prob),
            "TASK TYPE": (f"TASK: {task_type}", task_type_exclude_prob),
            "PROMPT": (f"PROMPT: {item}", prompt_exclude_prob),
            "TASK QUESTION": (f"TASK QUESTION: {question}", question_exclude_prob),
            "LANGUAGE": (f"LANGUAGE: {language}", language_exclude_prob),
            "RESPONSE": f"RESPONSE: `{response}`",
            "DETAILS": (
                ("## Details\n"
                "SCALE: 1-5, where 1 is `not original at all` and 5 is `extremely original`\n"
                "FORMAT: Return in the format of newline-separated `KEY:value` pairs, with the following fields:\n"
                "- `SCORE`: An originality score, 1-5\n"
                "- `CONFIDENCE`: A measure of confidence in the score, 1-3, or None.\n"
                "- `FLAGS`: A comma-separated list with content flags, such as: 'nonsense', 'violent', 'not practical'"
                ), detail_exclude_prob)
    }

        prompt_text = ""
        for key, value in components.items():
            if key in ['SCALE', 'FORMAT']:
                continue
            if key == "RESPONSE":
                prompt_text += value + "\n\n"
            else:
                if random.random() > value[1]:
                    prompt_text += value[0] + "\n"
                elif key == 'TASK QUESTION':
                    # include anyway if task type or prompt were removed
                    if ('TASK: ' not in prompt_text) or ('PROMPT: ' not in prompt_text):
                        prompt_text += value[0] + "\n"
                else:
                    pass

        if no_flags:
            prompt_text = re.sub(r'\n- `FLAGS`: .*', '', prompt_text)

        return prompt_text.strip()

    def craft_response(self, score, confidence=None, flags=None):
        # cast score type: it an be a float or None (written as 'null' in the dataset)
        if score is None:
            score = 'null'
        else:
            score = str(score)
        response = f'SCORE: {score}\n'

        if confidence and not np.isnan(confidence):
            response += f'CONFIDENCE: {confidence}\n'

        if flags:
            if isinstance(flags, list):
                flags = ','.join(flags)
            response += f'FLAGS: {flags}'
        return response.strip()

    def parse_response(self, response):
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

    def prepare_training_prompt(self, item, response, task_type, question, language, seed=None):
        ''' Opinionated probabilities of different parts of the prompt being included'''
        return self.prompter.craft_prompt(item, response, task_type, question, language, seed,
                                          action_exclude_prob=0.75, task_type_exclude_prob=0.3,
                                          prompt_exclude_prob=0, language_exclude_prob=0.5,
                                          question_exclude_prob=0.5, detail_exclude_prob=0.8,
                                          no_flags=True)
    
    def prepare_example(self, item, response, task_type='uses', question=None,
                        language=None, target=None, confidence=None, seed=None):
        prompt = self.prepare_training_prompt(item, response, task_type, question, language, seed)
        msgs = [
            {"role": "system", "content": self.sys_msg_text},
            {"role": "user", "content": prompt},
            ]
        # Add the response
        if target:
            ast_msg = {
                "role": "assistant",
                "content": self.craft_response(target, confidence)
            }
            msgs.append(ast_msg)
        return msgs

    def prepare_example_from_series(self, row, seed=None):
        '''Parse a row of a DataFrame, with the following columns:
            prompt, response, type (or task_type), question, language, target, confidence
        '''
        if ('task_type' not in row.index) and ('task' in row.index):
            row = row.rename(index={'type': 'task_type'})

        # prompt, response, type (or task_type), question, language, target, confidence
        include_params = ['prompt', 'response', 'task_type', 'question', 'language', 'target',
                          'confidence']
        kwargs = row[[p for p in include_params if p in row.index]].to_dict()
        return self.prepare_example(**kwargs, seed=seed)
