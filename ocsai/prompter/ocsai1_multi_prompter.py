from .ocsai1_prompter import Ocsai1_Prompter


class Ocsai1_Multi_Prompter(Ocsai1_Prompter):
    ''' A slightly modified training format from the 'classic' format in
    Organisciak et al. 2023, used in Acar et al. 2023, adjusted to support
    multiple task types.
    '''
    sys_msg_text = "You score originality in divergent thinking tasks."
    max_tokens: int = 2

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
        prompt_template = f'DT {task_type} Prompt:{item}\nResponse:{response}\nScore:\n '

        if question:
            self.logger.warning("Question is not supported with Classic Prompter")
        if language is not None and language != 'eng':
            self.logger.warning("Only 'eng' language is supported with Classic Prompter")

        return prompt_template.format(item, response)
