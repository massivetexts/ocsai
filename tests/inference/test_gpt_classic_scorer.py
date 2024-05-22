from ocsai.inference.gpt_classic_scorer import GPT_Classic_Scorer
import pandas as pd

# Sample data for testing
ocsai1_classic_models = {
    'ocsai-babbage2': 'ft:babbage-002:peter-organisciak:gt-main2-epochs2:8fDSqNqU',
    'ocsai-davinci2': 'ft:davinci-002:peter-organisciak:gt-main2-epochs2:8fE6TmoV'
}


# Test Cache-less Initialization
def test_gpt_classic_scorer_initialization():
    scorer = GPT_Classic_Scorer(cache=None, model_dict=ocsai1_classic_models)
    assert scorer is not None
    assert scorer.models == list(ocsai1_classic_models.keys())


# Test score method
def test_score_method():
    scorer = GPT_Classic_Scorer(cache=None, model_dict=ocsai1_classic_models)
    result = scorer.score('Pants', 'makeshift flag')
    assert result == [{'score': 3.5, 'confidence': None, 'flags': None, 'n': 1, 'type': 'top'}]


# Test originality method
def test_originality_method():
    scorer = GPT_Classic_Scorer(cache=None, model_dict=ocsai1_classic_models)
    score = scorer.originality('Pants', 'makeshift flag')
    assert score == 3.5


# Test originality method with specific model
def test_originality_with_model_method():
    scorer = GPT_Classic_Scorer(cache=None, model_dict=ocsai1_classic_models)
    score = scorer.originality('Pants', 'makeshift flag', model='ocsai-davinci2')
    assert score == 3.3


# Test originality_batch method
def test_originality_batch_method():
    scorer = GPT_Classic_Scorer(cache=None, model_dict=ocsai1_classic_models)
    results = scorer.originality_batch(['Pants', 'Pants'], ['makeshift flag', 'make a parachute'], model='ocsai-davinci2')
    expected_results = [{'score': 3.3, 'confidence': None, 'flags': None}, {'score': 2.7, 'confidence': None, 'flags': None}]
    assert results == expected_results


# Test originality_batch with questions instead of prompts. Should throw a value error
def test_originality_batch_with_questions():
    scorer = GPT_Classic_Scorer(cache=None, model_dict=ocsai1_classic_models)
    try:
        results = scorer.originality_batch(prompts=[],
                                           responses=['makeshift flag', 'make a parachute'],
                                           questions=['What is a surprising use for pants?']*2,
                                           model='ocsai-babbage2')
        assert False
    except ValueError:
        assert True


# Test crafting prompt
def test_craft_prompt():
    scorer = GPT_Classic_Scorer(cache=None, model_dict=ocsai1_classic_models)
    prompt = scorer.prompter.craft_prompt('Pants', 'makeshift flag')
    assert prompt == 'AUT Prompt:Pants\nResponse:makeshift flag\nScore:\n'


# Test DataFrame scoring
def test_originality_df_method():
    scorer = GPT_Classic_Scorer(cache=None, model_dict=ocsai1_classic_models)
    df = pd.DataFrame([['Pants', 'makeshift flag'], ['Pants','make a parachute']], columns=['prompt_randname', 'response_randname'])
    scored_df = scorer.originality_df(df, model='ocsai-davinci2', prompt_col='prompt_randname', response_col='response_randname')
    expected_dict = {
        'prompt_randname': {0: 'Pants', 1: 'Pants'},
        'response_randname': {0: 'makeshift flag', 1: 'make a parachute'},
        'score': {0: 3.3, 1: 2.7},
        'confidence': {0: None, 1: None},
        'flags': {0: None, 1: None}
    }
    assert scored_df.to_dict() == expected_dict

# Additional tests for caching and other functionalities can be added here