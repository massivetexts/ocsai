from typing import Literal
from .ocsai1_prompter import Ocsai1_Prompter


class Ocsai1_Chat_Prompter(Ocsai1_Prompter):
    ''' The format used in the original LLM paper, adjusted slightly to work with chat.'''
    sys_msg_text = "You score originality in the alternate uses task."
    max_tokens = 2
    stop_char = '\n'
    system_role_name: Literal['system', 'developer'] = 'system'

    def craft_prompt(self, item, response, task_type='uses', question=None, language='eng'):
        ''' Remove "\nScore:\n" from the legacy format.'''
        prompt = super().craft_prompt(item, response, task_type, question, language)
        return prompt.replace("\nScore:\n", "")

    def prepare_example(self, item, response, task_type='uses', question=None,
                        language=None, target=None, confidence=None, seed=None):
        prompt = self.craft_prompt(item, response, task_type, question, language)
        msgs = [
            {"role": self.system_role_name, "content": self.sys_msg_text},
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
