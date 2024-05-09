from .gpt_classic_prompter import GPT_Classic_Prompter, LogProbPair


class GPT_Classic_Chat_Prompter(GPT_Classic_Prompter):
    ''' The format used in the original LLM paper, adjusted slightly to work with chat.'''
    sys_msg_text = "You score originality in the alternate uses task."
    max_tokens = 25

    def _extract_content(self, choice) -> str:
        """Extract the content string from a response choice."""
        return choice.message.content

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

    def _extract_token_logprobs(self, choice) -> list[LogProbPair]:
        '''Extract the token log probabilities from a response.'''
        # FYI: Chat models, even with temperature=0, exhibit more randomness "
        # than classic models.
        score_logprobs = [(x.token, x.logprob)
                          for x in choice.logprobs.content[0].top_logprobs]
        return score_logprobs
