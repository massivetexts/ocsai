from .ocsai1_multi_prompter import Ocsai1_Multi_Prompter


class Ocsai1_Multi_Chat_Prompter(Ocsai1_Multi_Prompter):
    ''' The format used in the original LLM paper, adjusted slightly to work with chat.'''
    sys_msg_text = "You score originality in the alternate uses task."
    max_tokens = 2
    stop_char = '\n'

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
