'''
A generic interface for standardizing calls between multiple AI services.
'''

from .openai_chat_interface import OpenAIChatInterface
from .openai_legacy_interface import OpenAILegacyInterface
from .anthropic_interface import AnthropicInterface