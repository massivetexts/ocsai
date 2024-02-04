import pytest
from ocsai.inference.gpt_chat_scorer import GPT_Chat_Scorer
from ocsai.train import GPT_Classic_Chat_Prompter

# Sample data for testing
ocsai1_chat_models = {'ocsai-chatgpt': 'ft:gpt-3.5-turbo-1106:peter-organisciak::8fEKk0V6'}


# Test Initialization
def test_gpt_chat_scorer_initialization():
    chat_scorer = GPT_Chat_Scorer(cache=None, model_dict=ocsai1_chat_models, prompter=GPT_Classic_Chat_Prompter())
    assert chat_scorer is not None
    assert isinstance(chat_scorer.prompter, GPT_Classic_Chat_Prompter)


# Test score method
def test_chat_scorer_score_method():
    chat_scorer = GPT_Chat_Scorer(cache=None, model_dict=ocsai1_chat_models, prompter=GPT_Classic_Chat_Prompter())
    result = chat_scorer.score('Pants', 'makeshift flag')
    assert result == {'score': 3.3, 'confidence': None, 'flags': None}


# Test originality method
def test_chat_scorer_originality_method():
    chat_scorer = GPT_Chat_Scorer(cache=None, model_dict=ocsai1_chat_models, prompter=GPT_Classic_Chat_Prompter())
    score = chat_scorer.originality('Pants', 'makeshift flag')
    assert score == 3.3


# Test originality_batch method
def test_chat_scorer_originality_batch_method():
    chat_scorer = GPT_Chat_Scorer(cache=None, model_dict=ocsai1_chat_models, prompter=GPT_Classic_Chat_Prompter())
    results = chat_scorer.originality_batch(['Pants', 'Pants'], ['makeshift flag', 'make a parachute'], model='ocsai-chatgpt')
    expected_results = [{'score': 3.3, 'confidence': None, 'flags': None}, {'score': 3.3, 'confidence': None, 'flags': None}]
    assert results == expected_results


# Test crafting prompt
def test_chat_scorer_craft_prompt():
    chat_scorer = GPT_Chat_Scorer(cache=None, model_dict=ocsai1_chat_models, prompter=GPT_Classic_Chat_Prompter())
    prompt = chat_scorer.prompter.craft_prompt('Pants', 'makeshift flag')
    assert prompt == 'AUT Prompt:Pants\nResponse:makeshift flag'