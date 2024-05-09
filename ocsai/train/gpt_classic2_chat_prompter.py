from .gpt_classic_chat_prompter import GPT_Classic_Chat_Prompter


class GPT_Classic2_Chat_Prompter(GPT_Classic_Chat_Prompter):
    '''
    The format used in the multi-task format, adjusted to work with chat.

    There's some redundancy with GPT_Classic_Prompter - in the future I'll
    disambiguate model type parsing and prompt formats.
    '''
    sys_msg_text = "You score originality in divergent thinking tasks."

    def craft_prompt(self, item: str, response: str,
                     task_type: str | None = 'uses', question=None,
                     language='eng'):
        # prompt templates should take 2 args - item and response
        known_task_type_codes = ['Uses', 'Instances', 'Completion']
        if task_type is None:
            raise ValueError("Task type for classic prompter is necessary.")
        task_type = task_type.capitalize()
        if task_type not in known_task_type_codes:
            self.logger.debug(f"Task type {task_type} not a known type.")
        prompt_template = f'DT {task_type} Prompt:{item}\nResponse:{response}\nScore:\n'

        if question:
            self.logger.warning("Question is not supported with Classic Prompter")
        if language is not None and language != 'eng':
            self.logger.warning("Only 'eng' language is supported with Classic Prompter")

        return prompt_template.format(item, response)
