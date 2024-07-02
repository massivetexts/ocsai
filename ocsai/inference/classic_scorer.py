from ..types import StandardAIResponse
from ..prompter import Ocsai1_Prompter
from .base_scorer import Base_Scorer
from ..llm_interface import OpenAILegacyInterface
import openai

GPTCLASSICMODELS = {
    "ocsai-babbage2": "ft:babbage-002:peter-organisciak:gt-main2:8fCPIoCY",
    "ocsai-davinci2": "ft:davinci-002:peter-organisciak:gt-main2-epochs2:8fE6TmoV",
}

class Classic_Scorer(Base_Scorer):

    DEFAULT_PROMPTER = Ocsai1_Prompter
    DEFAULT_INTERFACE = OpenAILegacyInterface

    def __init__(self, *args, **kwargs):
        if "model_dict" not in kwargs or not kwargs["model_dict"]:
            kwargs["model_dict"] = GPTCLASSICMODELS
        if "prompter" not in kwargs or not kwargs["prompter"]:
            kwargs["prompter"] = self.DEFAULT_PROMPTER()
        super().__init__(*args, **kwargs)

        self.async_client = openai.AsyncOpenAI(api_key=openai.api_key)