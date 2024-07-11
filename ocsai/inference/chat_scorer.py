from ocsai.llm_interface.openai_chat_interface import OpenAIChatInterface
from .base_scorer import Base_Scorer
from ..prompter import Ocsai2_Prompter
import openai

GPTCHATMODELS = {
    "ocsai_1.5_full": "ft:gpt-3.5-turbo-1106:peter-organisciak:ocsai-full-1-24:8d5RLryO",
    "ocsai_1.5_small": "ft:gpt-3.5-turbo-1106:peter-organisciak:ocsai-small-1-24:8cmALwWt",
}


class Chat_Scorer(Base_Scorer):

    DEFAULT_PROMPTER = Ocsai2_Prompter
    DEFAULT_INTERFACE = OpenAIChatInterface
    max_logprobs = 20
    max_async_processes = 10

    def __init__(self, *args, **kwargs):
        if "model_dict" not in kwargs or not kwargs["model_dict"]:
            kwargs["model_dict"] = GPTCHATMODELS
        if "prompter" not in kwargs or not kwargs["prompter"]:
            kwargs["prompter"] = self.DEFAULT_PROMPTER()
        super().__init__(*args, **kwargs)

        self.async_client = openai.AsyncOpenAI(api_key=openai.api_key)
